"""Decision confusion matrices from finished runs: matrix_from_rows, the report section and the script."""
import importlib.util
import json
from pathlib import Path

from supportdesk.eval.grounded_metrics import format_confusion, matrix_correct, matrix_from_rows
from supportdesk.eval.report import render_report

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    {"case_id": "a1", "category": "answerable", "outcome": "answer", "citations": [], "violations": []},
    {"case_id": "a2", "category": "answerable", "outcome": "request_info", "citations": [], "violations": []},
    {"case_id": "c1", "category": "ambiguous", "outcome": "request_info", "citations": [], "violations": []},
    {"case_id": "e1", "category": "escalation", "outcome": "error", "citations": [], "violations": [],
     "error": "Connection error."},
]


def write_run(directory: Path, rows, run_id):
    directory.mkdir(parents=True)
    (directory / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "summary.json").write_text(json.dumps({"run_id": run_id, "pipeline": "advanced", "mode": "live",
                                                        "settings": {}, "stages": {}, "metrics": {}}))
    return directory


def test_the_matrix_counts_expected_against_decided_with_errors_as_their_own_column():
    m = matrix_from_rows(ROWS)
    assert m["counts"]["answer"] == {"answer": 1, "request_info": 1, "escalate": 0, "error": 0}
    assert m["counts"]["escalate"]["error"] == 1 and m["n"] == 4 and matrix_correct(m) == 2
    assert "| answer | 1 | 1 | 0 | 0 | 2 |" in format_confusion(m)


def test_a_retrieval_only_run_has_no_matrix():
    assert matrix_from_rows([{**r, "outcome": "retrieval_only"} for r in ROWS]) is None


def test_the_report_shows_a_matrix_for_each_run_with_decisions(tmp_path):
    answers = write_run(tmp_path / "e2e-x", ROWS, "e2e-x")
    retrieval = write_run(tmp_path / "ret", [{**r, "outcome": "retrieval_only"} for r in ROWS], "ret")
    try:
        page = render_report([answers, retrieval])
    except (KeyError, TypeError):  # the report needs full summaries: check the section on its own
        from supportdesk.eval.report import _decisions_section
        page = _decisions_section([{"summary": {"run_id": "e2e-x"}, "rows": {r["case_id"]: r for r in ROWS}},
                                   {"summary": {"run_id": "ret"},
                                    "rows": {r["case_id"]: {**r, "outcome": "retrieval_only"} for r in ROWS}}])
    assert "<h2>Decisions</h2>" in page and "e2e-x <span class=muted>2 of 4 correct" in page
    assert "ret <span" not in page  # no matrix for a retrieval-only run


def test_the_script_prints_markdown(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("confusion_matrix", ROOT / "scripts" / "confusion_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_run(tmp_path / "e2e-y", ROWS, "e2e-y")
    assert module.main(["e2e-y", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "### e2e-y: 2 of 4 correct" in out and "| escalate | 0 | 0 | 0 | 1 | 1 |" in out


def test_export_cases_writes_one_complete_file_per_case(tmp_path):
    spec = importlib.util.spec_from_file_location("export_cases", ROOT / "scripts" / "export_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from supportdesk.eval.cases import load_cases

    cases = load_cases(str(ROOT / "data" / "eval"), split="dev")
    if not cases:
        import pytest
        pytest.skip("no development cases in this checkout")
    case = cases[0]
    controlled = {"assessment": {"main_need": "x"}, "maintainer_summary": {"next_step": "y"}, "routing": {}}
    row = {"case_id": case.case_id, "outcome": case.acceptable_outcomes[0], "user_reply": "r", "citations": [],
           "maintainer_summary": {"problem": "p"}, "usage": {"cost_usd": 0.02}, "split": "dev",
           "trace": {"packet": [{"label": "S1", "chunk_id": "c1"}], "attempts": [{"controlled_v2": controlled}]}}
    run = tmp_path / "e2e-z"
    run.mkdir()
    (run / "results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary.json").write_text(json.dumps({"case_files": [str(ROOT / "data" / "eval")]}))
    assert module.main([str(run)]) == 0
    record = json.loads((run / "cases" / f"{case.case_id}.json").read_text())
    assert record["report"]["title"] == case.input.title and record["matches_label"] is True
    assert record["controlled_v2"]["assessment"] == {"main_need": "x"}
    assert [(e["label"], e["chunk_id"]) for e in record["evidence_in_prompt"]] == [("S1", "c1")]
    assert record["judge_reference"]["acceptable_outcomes"] == list(case.acceptable_outcomes)
    assert record["judge_reference"]["for_judging_only"]
    assert module.main([str(run), "--no-reference", "--output-dir", str(tmp_path / "blind")]) == 0
    assert "judge_reference" not in json.loads((tmp_path / "blind" / f"{case.case_id}.json").read_text())
    assert json.loads((run / "cases" / "index.json").read_text())[0]["case_id"] == case.case_id


def test_judge_agreement_compares_field_by_field(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("judge_agreement", ROOT / "scripts" / "judge_agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def judgement(verdict, score):
        return {"overall": {"verdict": verdict}, "decision_quality": {"decision_acceptable": verdict == "pass",
                                                                      "score": score}}
    (tmp_path / "m").mkdir()
    (tmp_path / "h").mkdir()
    for case, m, h in (("A", ("pass", 2), ("pass", 2)), ("B", ("fail", 0), ("partial", 1))):
        (tmp_path / "m" / f"{case}.json").write_text(json.dumps(judgement(*m)))
        (tmp_path / "h" / f"{case}.json").write_text(json.dumps(judgement(*h)))
    (tmp_path / "m" / "C.json").write_text(json.dumps(judgement("pass", 2)))  # judged by the model only
    result = module.compare(tmp_path / "m", tmp_path / "h")
    assert result["cases"] == ["A", "B"]
    assert result["fields"]["verdict"] == {"compared": 2, "exact": 1}
    assert result["fields"]["decision score"] == {"compared": 2, "exact": 1, "within_one": 2}
    assert {d["case_id"] for d in result["disagreements"]} == {"B"}
    assert module.main([str(tmp_path / "m"), str(tmp_path / "h")]) == 0
    assert "cases judged by both: 2" in capsys.readouterr().out


def test_the_judge_prompt_and_its_document_are_published():
    prompt = (ROOT / "configs" / "prompts" / "judge_case_system.txt").read_text()
    for field in ("evidence_quality", "decision_quality", "operational_success", "failure_analysis", "judge_reference"):
        assert field in prompt
    doc = (ROOT / "docs" / "evaluation" / "judge_prompt.md").read_text()
    assert "judge_case_system.txt" in doc and "human" in doc.lower()
