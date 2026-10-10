"""Pipeline snapshots: the stages, the settings, a stage-by-stage run, and the latest results."""
import json

from supportdesk.cli import main
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.pipelines.staged import STAGE_NAMES
from supportdesk.retrieval.retriever import DenseRetriever
from supportdesk.snapshot import take_snapshot
from supportdesk.types import CaseInput
from tests.helpers import ROOT, SAMPLE

CASES = str(ROOT / "data" / "eval" / "sample_cases.jsonl")
CASE = CaseInput(case_id="s", title="Session state | empty after refresh",
                 body="st.session_state is lost after F5. Streamlit 1.31.1, Chrome.",
                 created_at="2024-04-20T10:00:00Z")


def baseline(index, clients, settings):
    return build_baseline(DenseRetriever(index, clients.embedding), clients, settings)


def test_the_observer_sees_every_step_in_order(index, clients, settings):
    seen = []
    result = baseline(index, clients, settings).run(CASE, observe=lambda step, out: seen.append(step))
    assert seen == ["environment", *STAGE_NAMES[:3], "packet", "generator", "validator", "summarizer"]
    assert result.outcome == "answer"


def test_the_generator_keeps_the_raw_model_output(index, clients, settings):
    outputs = {}
    baseline(index, clients, settings).run(CASE, observe=outputs.__setitem__)
    assert json.loads(outputs["generator"].raw_output)["outcome"] == "answer"


def test_snapshot_shows_stages_settings_walkthrough_and_cost(index, clients, settings):
    text = take_snapshot(baseline(index, clients, settings), settings, CASE,
                         gold=["fixture-doc-session-state"])
    assert text.startswith("# Pipeline snapshot: baseline")
    for implementation in ("PlainQuery", "DenseRetriever", "NoRerank", "DefaultPacketBuilder",
                           "SingleCallGenerator", "NoValidator", "build_summary"):
        assert f"`{implementation}`" in text
    assert "Embedding similarity between the query and each chunk." in text  # the stage's own docstring
    assert "| `retrieval.top_k` | 6 |" in text and "chunks allowed in the prompt 6" in text
    assert "6 candidates (pool 6)" in text and "| ✓ |" in text  # gold evidence is marked
    assert "Session state \\| empty after refresh" in text  # table-breaking characters are escaped
    assert '"outcome": "answer"' in text and "Outcome: **answer**" in text
    assert "| answer | chat | offline |" in text  # every model call, with its cost
    assert "Latest evaluation results" not in text  # no runs directory was given


def test_a_failed_run_still_gives_a_useful_snapshot(index, clients, settings):
    clients.chat.script = {1: "bad_json", 2: "bad_json"}
    text = take_snapshot(baseline(index, clients, settings), settings, CASE)
    assert "### 4. packet" in text and "### 5. generator" not in text  # stops where the run stopped
    assert "Stopped with an error: `no valid model output after 2 attempts`" in text
    assert text.count("| answer | chat |") == 2  # both failed attempts are on the record


def test_cli_snapshot_writes_one_file_per_pipeline_with_latest_results(tmp_path, capsys):
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline", "--cases", CASES]
    assert main(["index", *base]) == 0
    assert main(["eval", *base, "--pipeline", "baseline", "--run-id", "b1"]) == 0
    capsys.readouterr()
    assert main(["snapshot", *base, "--all"]) == 0
    out = capsys.readouterr().out
    assert "pipeline: baseline" in out and "pipeline: advanced" in out
    snapshots = tmp_path / "art" / "snapshots"
    base_md, adv_md = (snapshots / "baseline.md").read_text(), (snapshots / "advanced.md").read_text()
    assert "**full** run `b1`" in base_md and "outcome accuracy" in base_md
    assert "No runs yet" in adv_md and "`HybridRetriever`" not in base_md
    assert "Gold evidence for this case: `fixture-doc-session-state`" in base_md  # first dev case


def test_cli_snapshot_can_run_any_report_and_explains_when_there_is_no_case(tmp_path, capsys):
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline"]
    assert main(["index", *base, "--cases", CASES]) == 0
    assert main(["snapshot", *base, "--pipeline", "advanced", "--title", "App stuck on loading",
                 "--body", "works locally", "--output-dir", str(tmp_path / "snap")]) == 0
    assert "**App stuck on loading**" in (tmp_path / "snap" / "advanced.md").read_text()
    capsys.readouterr()
    assert main(["snapshot", *base, "--cases", str(tmp_path / "none.jsonl")]) == 1
    assert "no case to run" in capsys.readouterr().err


def test_latest_runs_picks_the_newest_run_of_each_kind_by_date_not_name(tmp_path):
    from supportdesk.snapshot import latest_runs

    def run(name, created, kind="full", pipeline="baseline"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(
            {"pipeline": pipeline, "kind": kind, "created_at": created, "metrics": {}}))
    run("aaa-old", "2020-01-01T00:00:00+00:00")
    run("zzz-new", "2026-01-01T00:00:00+00:00")
    run("aaa-ret-new", "2026-02-01T00:00:00+00:00", kind="retrieval")
    run("zzz-ret-old", "2020-02-01T00:00:00+00:00", kind="retrieval")
    run("mmm-other", "2027-01-01T00:00:00+00:00", pipeline="advanced")
    found = latest_runs(tmp_path, "baseline")
    assert found["full"][0] == "zzz-new" and found["retrieval"][0] == "aaa-ret-new"
    assert latest_runs(tmp_path, "nothing") == {}
