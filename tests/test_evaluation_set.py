"""The evaluation files (schema 6.0): base cases, history variants, source-usage tasks."""
import json

import pytest

from supportdesk.eval.cases import load_cases
from supportdesk.eval.dataset import EvaluationSetError, check_spans, load_evaluation_set
from supportdesk.eval.metrics import evaluate
from tests.helpers import ROOT

EVAL_DIR = ROOT / "tests" / "data" / "eval"
REAL = ROOT / "data" / "processed" / "corpus.jsonl"


def by_id(cases):
    return {c.case_id: c for c in cases}


def test_the_three_kinds_load_and_map():
    cases = by_id(load_evaluation_set(EVAL_DIR, "development"))
    assert len(cases) == 17 and sum(c.independent for c in cases.values()) == 15
    base = cases["DEV-5604"]
    assert (base.kind, base.category, base.acceptable_outcomes) == ("issue", "ambiguous", ["request_info"])
    assert base.source_issue_number == 5604 and base.split == "dev" and base.rubric["case_category"] == "needs_clarification"
    assert base.input.created_at is None  # historical_evaluation is false: no date filter
    hist = cases["HIST-DEV-7044"]
    assert hist.kind == "history_variant" and not hist.independent and hist.family_id == "family:DEV-7044"
    assert hist.input.body.startswith("### Summary")  # the original report comes first...
    assert [m["role"] for m in hist.input.history] == ["assistant", "user"]
    assert hist.input.history[-1]["body"] == "@LukasMasuch Yes exactly."  # ...the follow-up is the newest message
    assert hist.input.environment["streamlit_version"] == "1.23.1" and hist.acceptable_outcomes == []
    src = cases["SRC-DEV-ISSUE"]
    assert src.kind == "source_usage" and src.gold_evidence == ["issue:7851"] and src.source_issue_number is None


def test_evaluation_never_auto_scores_rubric_only_cases_and_counts_only_independent_ones():
    cases = load_evaluation_set(EVAL_DIR, "development")
    from supportdesk.types import CaseResult
    pairs = [(c, CaseResult(case_id=c.case_id, pipeline="p", mode="offline", outcome="answer").to_dict()) for c in cases]
    m = evaluate(pairs, ks=(1,))
    assert m["n_cases"] == 17 and m["n_independent"] == 15
    assert m["by_category"]["history_variant"]["outcome_accuracy"] is None
    assert m["by_category"]["source_usage"]["outcome_accuracy"] is None
    assert m["by_category"]["answerable"]["outcome_accuracy"] == 1.0


@pytest.mark.skipif(not REAL.exists(), reason="real corpus not present")
def test_every_gold_span_matches_the_corpus_exactly():
    texts = {json.loads(line)["source_id"]: json.loads(line)["text"]
             for line in REAL.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert check_spans(load_evaluation_set(EVAL_DIR, "development"), texts) == []
    broken = load_evaluation_set(EVAL_DIR, "development")[0]
    broken.evidence_spans[0] = {**broken.evidence_spans[0], "start_char": broken.evidence_spans[0]["start_char"] + 1}
    assert "does not match" in check_spans([broken], texts)[0]


def test_a_folder_in_case_files_is_read_as_an_evaluation_set():
    assert {c.case_id for c in load_cases(EVAL_DIR)} == {c.case_id for c in load_evaluation_set(EVAL_DIR, "development")}


def test_missing_rubrics_and_unknown_categories_are_refused(tmp_path):
    (tmp_path / "development_inputs.jsonl").write_text((EVAL_DIR / "development_inputs.jsonl").read_text())
    with pytest.raises(EvaluationSetError, match="no rubric file"):
        load_evaluation_set(tmp_path, "development")
    rubrics = [json.loads(line) for line in (EVAL_DIR / "development_rubrics.jsonl").read_text().splitlines() if line]
    rubrics[0]["case_category"] = "maybe"
    (tmp_path / "development_rubrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rubrics))
    with pytest.raises(EvaluationSetError, match="unknown case_category"):
        load_evaluation_set(tmp_path, "development")


def test_check_cases_reads_an_evaluation_folder_from_the_cli(tmp_path, capsys):
    from supportdesk.cli import main
    from tests.helpers import SAMPLE
    code = main(["check-cases", "--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
                 "--mode", "offline", "--cases", str(EVAL_DIR)])
    out = capsys.readouterr()
    assert "not found" not in out.err and code in (0, 1)
