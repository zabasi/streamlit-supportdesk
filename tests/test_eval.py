import json
import re
from pathlib import Path

import pytest

from supportdesk.cli import main
from supportdesk.eval.cases import case_from_dict, exclusions_from_cases
from supportdesk.eval.leakage import check_leakage
from supportdesk.eval.metrics import evaluate, recall_at_k, reciprocal_rank
from supportdesk.eval.runner import run_eval
from supportdesk.eval.splits import assign_splits, check_no_family_overlap
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.retrieval.retriever import DenseRetriever

ROOT = Path(__file__).resolve().parent.parent
SAMPLE, CASES = str(ROOT / "data" / "sample"), str(ROOT / "data" / "eval" / "sample_cases.jsonl")

LEAK = {"overlap_threshold": 0.6, "shingle_tokens": 8}
R = [{"chunk_id": "a:000", "source_id": "a", "parent_source_id": None},
     {"chunk_id": "c:000", "source_id": "comment-1", "parent_source_id": "issue-9"},
     {"chunk_id": "d:000", "source_id": "d", "parent_source_id": None}]


def test_recall_mrr_and_comment_counts_as_parent():
    assert recall_at_k(R, ["d"], 2) == 0.0 and recall_at_k(R, ["d"], 3) == 1.0
    assert recall_at_k(R, ["issue-9", "a"], 2) == 1.0  # the comment counts as its parent issue
    assert reciprocal_rank(R, ["d"]) == pytest.approx(1 / 3)
    assert recall_at_k(R, [], 3) is None and reciprocal_rank(R, []) is None


def test_case_validation_rejects_bad_labels():
    base = {"case_id": "x", "family_id": "f", "split": "dev", "category": "answerable",
            "input": {"title": "t", "body": "b"}, "acceptable_outcomes": ["answer"]}
    assert case_from_dict(base).case_id == "x"
    for bad in ({"split": "val"}, {"category": "easy"}, {"acceptable_outcomes": ["solved"]}):
        with pytest.raises(ValueError):
            case_from_dict({**base, **bad})


def test_splits_are_family_level_and_deterministic():
    families = [f"fam-{i}" for i in range(20)]
    a = assign_splits(families, test_fraction=0.5, seed=7)
    assert a == assign_splits(list(reversed(families)), test_fraction=0.5, seed=7)
    assert sum(v == "test" for v in a.values()) == 10 and set(a.values()) == {"dev", "test"}
    cases = [case_from_dict({"case_id": f"c{i}", "family_id": "same", "split": s,
                             "category": "answerable", "input": {}, "acceptable_outcomes": ["answer"]})
             for i, s in enumerate(["dev", "test"])]
    assert check_no_family_overlap(cases) == ["family 'same' spans ['dev', 'test']"]


def test_index_excludes_every_eval_issue_and_leakage_check_agrees(corpus, cases, chunks):
    assert {9004, 9005, 9006} == set(exclusions_from_cases(cases).issue_numbers)
    assert check_leakage(chunks, cases, **LEAK).ok
    from supportdesk.data.chunking import FixedTokenChunker, build_chunks
    leaky = build_chunks(corpus.records, chunker=FixedTokenChunker(max_tokens=512, overlap_tokens=64))  # no exclusions
    report = check_leakage(leaky, cases, **LEAK)
    assert not report.ok and any("#9004" in v for v in report.violations)


def test_leakage_warns_on_links_and_verbatim_copies(chunks, cases):
    from dataclasses import replace
    linked = replace(chunks[0], text="Duplicate of https://github.com/streamlit/streamlit/issues/9004")
    copy = replace(chunks[1], text=f"{cases[0].input.title} {cases[0].input.body}")
    report = check_leakage([linked, copy], cases, **LEAK)
    assert any("links to evaluation issue #9004" in w for w in report.warnings)
    assert any("near-verbatim" in w for w in report.warnings)


def test_run_eval_writes_auditable_artifacts(index, clients, settings, cases, tmp_path):
    pipe = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)
    summary = run_eval(pipe, [c for c in cases if c.split == "dev"], clients, settings,
                       run_id="t", index_manifest=index.manifest, out_root=tmp_path)
    out = tmp_path / "t"
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert [r["case_id"] for r in rows] == ["c-001", "c-002"] and rows[0]["category"] == "answerable"
    assert (out / "cost_ledger.csv").exists() and summary["metrics"]["n_cases"] == 2
    assert summary["settings"]["api_key_present"] in (True, False) and not re.search(r"sk-[A-Za-z0-9_-]{16,}", json.dumps(summary))
    assert summary["metrics"]["by_category"]["ambiguous"]["n"] == 1


def test_metrics_split_by_category_so_refusing_everything_is_visible(cases):
    def result(outcome):
        return {"outcome": outcome, "retrieved": [], "citations": [], "violations": [], "usage": {}}
    pairs = [(c, result("escalate")) for c in cases]  # a system that refuses everything
    m = evaluate(pairs, (1, 3, 5, 6))
    assert m["by_category"]["answerable"]["outcome_accuracy"] == 0.0
    assert m["by_category"]["escalation"]["outcome_accuracy"] == 1.0
    assert 0 < m["outcome_accuracy"] < 1


def test_budget_stop_marks_remaining_cases(index, clients, settings, cases, tmp_path):
    from supportdesk.llm.errors import BudgetExhausted

    class Broke:
        def complete(self, messages, *, stage="answer", json_mode=True):
            raise BudgetExhausted("quota")
    clients.chat = Broke()
    pipe = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)
    run_eval(pipe, cases, clients, settings, run_id="b", out_root=tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["budget_stopped"] * 3


def test_cli_end_to_end_and_test_split_guard(tmp_path, capsys):
    common = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--mode", "offline"]
    assert main(["index", *common, "--cases", CASES]) == 0
    assert main(["check-cases", *common, "--cases", CASES]) == 0
    assert main(["eval", *common, "--cases", CASES, "--split", "test"]) == 2
    assert "--final" in capsys.readouterr().err
    assert main(["ask", *common, "--title", "Session state empty after refresh",
                 "--body", "st.session_state value is gone after F5"]) == 0
