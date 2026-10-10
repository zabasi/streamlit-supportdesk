"""Retrieval-only evaluation: recall@k, hit@k, MRR and packet recall, without generating answers."""
import json
import re
from pathlib import Path

from supportdesk.cli import main
from supportdesk.eval.cases import case_from_dict
from supportdesk.eval.metrics import evaluate, evaluate_retrieval, format_metrics
from supportdesk.eval.report import render_report
from supportdesk.eval.runner import run_eval, run_retrieval_eval
from supportdesk.llm.errors import BudgetExhausted, ProviderError
from supportdesk.types import CaseInput
from tests.helpers import ROOT, SAMPLE
from tests.test_config import build_world

SAMPLE_CASES = str(ROOT / "data" / "eval" / "sample_cases.jsonl")


def make_case(cid, category, gold):
    return case_from_dict({"case_id": cid, "family_id": f"f-{cid}", "split": "dev", "category": category,
                           "input": {"title": "t", "body": "b"}, "gold_evidence": gold,
                           "acceptable_outcomes": ["answer"]})


def row(retrieved_sources, packet_sources):
    def rows(ids):
        return [{"chunk_id": f"{s}:000", "source_id": s, "parent_source_id": None, "rank": i + 1}
                for i, s in enumerate(ids)]
    return {"outcome": "retrieval_only", "retrieved": rows(retrieved_sources),
            "packet": rows(packet_sources), "usage": {}}


def test_metrics_match_a_hand_computed_example():
    a, b, c = (make_case("a", "answerable", ["d1", "d2"]), make_case("b", "ambiguous", ["d3"]),
               make_case("c", "escalation", []))
    pairs = [(a, row(["d1", "x", "d2"], ["d1", "x"])),   # finds d1 at rank 1, d2 at rank 3
             (b, row(["x", "y", "z"], ["x"])),            # finds nothing
             (c, row(["x"], ["x"]))]                      # no gold: must not count
    m = evaluate_retrieval(pairs, (1, 3))
    assert (m["n_cases"], m["n_with_gold"], m["n_without_gold"]) == (3, 2, 1)
    assert m["recall_at_k"] == {"1": 0.25, "3": 0.5}      # a: 0.5 / 1.0   b: 0 / 0   -> mean
    assert m["hit_at_k"] == {"1": 0.5, "3": 0.5}          # a hits at 1; b never
    assert m["mrr"] == 0.5                                # a: 1/1, b: 0
    assert m["packet_recall"] == 0.25                     # a: d1 of {d1,d2} in the prompt = 0.5, b: 0
    assert m["by_category"]["answerable"]["recall_at_k"] == {"1": 0.5, "3": 1.0}
    assert m["by_category"]["ambiguous"]["recall_at_k"] == {"1": 0.0, "3": 0.0}
    assert "escalation" not in m["by_category"]            # no gold -> nothing to measure


def test_a_comment_counts_as_a_hit_on_its_parent_issue():
    a = make_case("a", "answerable", ["issue-9"])
    r = row(["x"], [])
    r["retrieved"].append({"chunk_id": "c:0", "source_id": "comment-1", "parent_source_id": "issue-9"})
    assert evaluate_retrieval([(a, r)], (1, 2))["recall_at_k"] == {"1": 0.0, "2": 1.0}


def test_no_gold_anywhere_gives_none_not_a_misleading_zero():
    m = evaluate_retrieval([(make_case("c", "answerable", []), row(["x"], ["x"]))], (1, 3))
    assert m["n_with_gold"] == 0 and m["recall_at_k"] == {"1": None, "3": None} and m["mrr"] is None
    assert "n/a" in format_metrics(m)


def test_retrieve_matches_run_and_never_calls_the_model(tmp_path, cases):
    _, clients, _, pipe = build_world(tmp_path, cases)
    case = CaseInput(case_id="t", title="Session state is empty after browser refresh",
                     body="st.session_state lost after F5", created_at="2024-04-20T10:00:00Z")
    r = pipe.retrieve(case)
    assert clients.chat.calls == 0  # nothing but the query embedding happened
    full = pipe.run(case)
    assert [h.chunk.chunk_id for h in r.hits] == [x["chunk_id"] for x in full.retrieved]
    assert r.packet.sha256 == full.packet_hash and r.queries and r.env == {}


def test_run_retrieval_eval_writes_artifacts_and_spends_nothing_on_answers(tmp_path, cases):
    settings, clients, index, pipe = build_world(tmp_path, cases, eval={"recall_ks": [1, 3]})
    dev = [c for c in cases if c.split == "dev"]
    summary = run_retrieval_eval(pipe, dev, clients, settings, run_id="r", out_root=tmp_path / "runs",
                                 index_manifest=index.manifest)
    assert clients.chat.calls == 0 and summary["kind"] == "retrieval"
    assert set(summary["metrics"]["recall_at_k"]) == {"1", "3"} and summary["metrics"]["n_with_gold"] == 2
    rows = [json.loads(x) for x in (tmp_path / "runs" / "r" / "results.jsonl").read_text().splitlines()]
    assert [r["case_id"] for r in rows] == ["c-001", "c-002"]
    assert rows[0]["outcome"] == "retrieval_only" and rows[0]["retrieved"] and rows[0]["packet"]
    assert (tmp_path / "runs" / "r" / "cost_ledger.csv").exists()
    assert summary["settings"]["config"]["retrieval"]["top_k"] == 6


def test_the_sample_gold_evidence_is_actually_found(tmp_path, cases):
    settings, clients, _, pipe = build_world(tmp_path, cases)
    m = run_retrieval_eval(pipe, [c for c in cases if c.split == "dev"], clients, settings,
                           run_id="r", out_root=tmp_path)["metrics"]
    assert m["recall_at_k"]["6"] == 1.0 and m["packet_recall"] == 1.0 and m["mrr"] > 0.5


def test_failures_are_recorded_per_case_and_a_budget_stop_ends_the_run(tmp_path, cases):
    settings, clients, _, pipe = build_world(tmp_path, cases)

    class Flaky:
        n = 0

        def __call__(self, case, env):
            Flaky.n += 1
            if Flaky.n == 1:
                raise ProviderError("502 key sk-abcdefghijklmnop12345")
            raise BudgetExhausted("insufficient_quota")
    broken = pipe.variant(query=Flaky())
    summary = run_retrieval_eval(broken, cases, clients, settings, run_id="b", out_root=tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["error", "budget_stopped", "budget_stopped"]
    assert "sk-abcdef" not in json.dumps(rows) and summary["metrics"]["budget_stopped"] == 2


def _cli(tmp_path, *args):
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline"]
    return main([args[0], *base, *args[1:]])


def test_cli_eval_retrieval_prints_a_readable_summary(tmp_path, capsys):
    assert _cli(tmp_path, "index", "--cases", SAMPLE_CASES) == 0
    capsys.readouterr()
    assert _cli(tmp_path, "eval-retrieval", "--cases", SAMPLE_CASES, "--run-id", "r1") == 0
    out = capsys.readouterr().out
    assert "evidence   recall @1" in out and "gold in the prompt packet" in out
    assert "have gold evidence" in out
    assert out.startswith("pipeline: baseline | retriever: dense | top_k: 6 | profile: ")
    assert re.search(r"time\s+mean \d+\.\d{3}s\s+p95 \d+\.\d{3}s", out) and "n/as" not in out
    assert (tmp_path / "art" / "runs" / "r1" / "summary.json").exists()
    assert _cli(tmp_path, "eval-retrieval", "--cases", SAMPLE_CASES, "--json", "--run-id", "r2") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["kind"] == "retrieval"  # stdout is pure JSON
    assert captured.err.startswith("pipeline: baseline")  # the header went to stderr


def test_cli_eval_retrieval_guards_the_test_split_like_eval(tmp_path, capsys):
    assert _cli(tmp_path, "index", "--cases", SAMPLE_CASES) == 0
    capsys.readouterr()
    assert _cli(tmp_path, "eval-retrieval", "--cases", SAMPLE_CASES, "--split", "test") == 2
    assert "--final" in capsys.readouterr().err
    assert _cli(tmp_path, "eval-retrieval", "--cases", SAMPLE_CASES, "--split", "test", "--final") == 0


def test_cli_eval_prints_the_same_readable_layout(tmp_path, capsys):
    assert _cli(tmp_path, "index", "--cases", SAMPLE_CASES) == 0
    capsys.readouterr()
    assert _cli(tmp_path, "eval", "--cases", SAMPLE_CASES, "--run-id", "e1") == 0
    out = capsys.readouterr().out
    for needle in ("cases: 2", "evidence   recall @1", "decisions  outcome accuracy", "grounding  citation",
                   "cost       $0.0000", "time       mean"):
        assert needle in out


def test_retrieval_and_full_runs_can_be_compared_in_one_report(tmp_path, cases):
    settings, clients, _, pipe = build_world(tmp_path, cases)
    dev = [c for c in cases if c.split == "dev"]
    run_eval(pipe, dev, clients, settings, run_id="full", out_root=tmp_path)
    run_retrieval_eval(pipe, dev, clients, settings, run_id="ret", out_root=tmp_path)
    run_retrieval_eval(pipe.variant(), dev, clients, settings, run_id="ret2", out_root=tmp_path)
    html = render_report([tmp_path / "full", tmp_path / "ret", tmp_path / "ret2"])
    assert "retrieval-only" in html and "Hit@3" in html and "Gold evidence in the prompt packet" in html
    assert "retrieval_only" in html  # per-case outcome of the retrieval runs
    only_ret = render_report([tmp_path / "ret", tmp_path / "ret2"])
    assert "recall@10" in only_ret  # the by-category cell falls back to recall at the largest k


def test_format_metrics_covers_both_kinds_and_replayed_cost(tmp_path, cases):
    settings, clients, _, pipe = build_world(tmp_path, cases)
    full = run_eval(pipe, cases, clients, settings, run_id="f", out_root=tmp_path)["metrics"]
    text = format_metrics(full)
    assert "decisions" in text and "answerable" in text
    full["cost"]["replayed_usd"] = 0.0123
    assert "+$0.0123 replayed, not spent" in format_metrics(full)
    assert evaluate([], (1,))["n_cases"] == 0


def case_with(gold, created="2024-06-01T10:00:00Z"):
    return case_from_dict({"case_id": "g", "family_id": "f", "split": "dev", "category": "answerable",
                           "input": {"title": "t", "body": "b", "created_at": created},
                           "gold_evidence": gold, "acceptable_outcomes": ["answer"]})


def test_reachable_labels_produce_no_warnings(chunks, cases):
    from supportdesk.eval.gold import check_gold
    assert check_gold(cases, chunks, historical=True) == []  # the shipped sample cases are clean


def test_a_typo_or_an_excluded_issue_is_flagged_as_not_in_the_index(chunks):
    from supportdesk.eval.gold import check_gold
    problems = check_gold([case_with(["fixture-doc-sesion-state", "fixture-issue-9004"])], chunks,
                          historical=True)
    assert len(problems) == 2 and all("not in the index" in p for p in problems)
    assert "excluded" in problems[0]  # tells the labeller the likely cause


def test_gold_dated_after_the_case_can_never_be_retrieved(chunks):
    from supportdesk.eval.gold import check_gold
    early = case_with(["fixture-release-1.33.0", "fixture-comment-3", "fixture-doc-session-state"],
                      created="2024-03-11T11:00:00Z")  # release 2024-04-04, comment 2024-03-12
    problems = check_gold([early], chunks, historical=True)
    assert len(problems) == 2 and all("dated after the case" in p for p in problems)
    assert any("fixture-release-1.33.0" in p for p in problems)
    assert check_gold([early], chunks, historical=False) == []  # only matters when the filter is on


def test_a_comment_makes_its_parent_issue_reachable(chunks):
    from supportdesk.eval.gold import check_gold
    # issue 9002 is indexed; its comment 4 is dated 2024-02-03. A case from March sees both.
    assert check_gold([case_with(["fixture-issue-9002"], "2024-03-01T00:00:00Z")], chunks,
                      historical=True) == []


def test_describe_cases_counts_splits_and_categories(cases):
    from supportdesk.eval.gold import describe_cases
    info = describe_cases(cases)
    assert info["total"] == 3 and info["families"] == 3
    assert info["by_split"]["dev"]["n"] == 2 and info["by_split"]["test"]["with_gold"] == 0


def test_cli_check_cases_reports_counts_and_fails_on_bad_labels(tmp_path, capsys):
    assert _cli(tmp_path, "index", "--cases", SAMPLE_CASES) == 0
    capsys.readouterr()
    assert _cli(tmp_path, "check-cases", "--cases", SAMPLE_CASES) == 0
    out = capsys.readouterr().out
    assert "3 cases in 3 families" in out and "dev" in out and "ok: every gold label is reachable" in out
    assert "no gold evidence" in out  # c-003 is an escalation case with none
    bad = tmp_path / "bad.jsonl"
    rows = [json.loads(x) for x in Path(SAMPLE_CASES).read_text().splitlines()]
    rows[0]["gold_evidence"] = ["fixture-doc-typo"]
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert _cli(tmp_path, "check-cases", "--cases", str(bad)) == 2
    assert "FAIL: c-001: gold 'fixture-doc-typo' is not in the index" in capsys.readouterr().out


def test_eval_warns_about_bad_labels_and_records_them_in_the_run_summary(tmp_path, capsys):
    assert _cli(tmp_path, "index", "--cases", SAMPLE_CASES) == 0
    bad = tmp_path / "bad.jsonl"
    rows = [json.loads(x) for x in Path(SAMPLE_CASES).read_text().splitlines()]
    rows[0]["gold_evidence"] = ["fixture-doc-typo"]
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    capsys.readouterr()
    assert _cli(tmp_path, "eval-retrieval", "--cases", str(bad), "--run-id", "w") == 0
    assert "WARNING gold label: c-001: gold 'fixture-doc-typo'" in capsys.readouterr().err
    summary = json.loads((tmp_path / "art" / "runs" / "w" / "summary.json").read_text())
    assert len(summary["warnings"]) == 1 and "fixture-doc-typo" in summary["warnings"][0]


def test_missing_timings_read_as_n_a_not_n_as():
    m = evaluate_retrieval([(make_case("a", "answerable", ["d1"]), row(["d1"], ["d1"]))], (1,))
    m["cost"]["wall_seconds_mean"] = m["cost"]["wall_seconds_p95"] = None
    assert "mean n/a   p95 n/a" in format_metrics(m)
