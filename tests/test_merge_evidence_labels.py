"""scripts/merge_evidence_labels.py: reviewed evidence labels replace the rubrics' evidence, nothing else."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "merge_evidence_labels", Path(__file__).resolve().parents[1] / "scripts" / "merge_evidence_labels.py")
merge_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_mod)

DOC = "## Alpha\nalpha text here.\n\n## Beta\nbeta text here.\n"
NOTE = "A later comment with the fix."
CORPUS = {"doc:p": SimpleNamespace(text=DOC, as_of=None), "comment:9": SimpleNamespace(text=NOTE, as_of="2024-01-01T00:00:00Z")}
B = DOC.index("## Beta")


def labels(spans):
    return [{"case_id": "C1", "input": {}, "sources": [
        {"source": {"source_id": sid, "source_type": kind, "text": CORPUS[sid].text}, "selection_scope": "partial",
         "selected_spans": [{"span_id": f"span:{sid}:{a}", "start_char": a, "end_char": b, "text": CORPUS[sid].text[a:b]}]}
        for sid, kind, a, b in spans]}]


INPUTS = [{"case_id": "C1", "created_at": "2023-01-01T00:00:00Z", "historical_evaluation": False}]
RUBRICS = [{"case_id": "C1", "case_category": "answerable", "acceptable_next_step": "keep me",
            "acceptable_retrievable_evidence": [{"evidence_id": "old:beta", "source_id": "doc:p", "start_char": B,
                                                 "end_char": len(DOC), "offset_basis": "retrieval_corpus.text",
                                                 "offset_unit": "python_unicode_code_points", "version_scope": None}],
            "claim_support": [{"claim_id": "c", "evidence_ids": ["old:beta"]}]}]


def test_new_spans_replace_the_evidence_and_everything_else_is_kept():
    spans = [("doc:p", "documentation", B, len(DOC)), ("comment:9", "comment", 0, len(NOTE))]
    inputs, rubrics, report = merge_mod.merge(labels(spans), INPUTS, RUBRICS, CORPUS, historical=False)
    r = rubrics[0]
    assert [e["source_id"] for e in r["acceptable_retrievable_evidence"]] == ["doc:p", "comment:9"]
    assert r["acceptable_next_step"] == "keep me" and r["later_evidence"] == []
    assert r["claim_support"][0]["evidence_ids"] == [f"span:doc:p:{B}"]  # re-pointed to the span covering it
    assert r["acceptable_retrievable_evidence"][1]["source_as_of"] == "2024-01-01T00:00:00Z"
    assert inputs[0]["historical_evaluation"] is False and report["unmapped_claim_refs"] == []


def test_historical_moves_evidence_dated_after_the_case_out_of_the_gold():
    spans = [("doc:p", "documentation", B, len(DOC)), ("comment:9", "comment", 0, len(NOTE))]
    inputs, rubrics, _ = merge_mod.merge(labels(spans), INPUTS, RUBRICS, CORPUS, historical=True)
    assert [e["source_id"] for e in rubrics[0]["acceptable_retrievable_evidence"]] == ["doc:p"]  # undated: visible
    assert [e["source_id"] for e in rubrics[0]["later_evidence"]] == ["comment:9"] and inputs[0]["historical_evaluation"]


def test_a_dropped_passage_drops_its_claim_reference_and_is_reported():
    _, rubrics, report = merge_mod.merge(labels([("doc:p", "documentation", 0, B)]), INPUTS, RUBRICS, CORPUS,
                                         historical=False)
    assert rubrics[0]["claim_support"][0]["evidence_ids"] == [] and report["unmapped_claim_refs"] == [("C1", "c", "old:beta")]


def test_nothing_is_merged_when_a_span_or_source_does_not_match_the_corpus():
    bad = labels([("doc:p", "documentation", 0, B)])
    bad[0]["sources"][0]["selected_spans"][0]["text"] = "not the corpus text"
    with pytest.raises(ValueError, match="not the source text"):
        merge_mod.merge(bad, INPUTS, RUBRICS, CORPUS, historical=False)
    with pytest.raises(ValueError, match="same case ids"):
        merge_mod.merge(labels([("doc:p", "documentation", 0, B)]), [{"case_id": "other"}], RUBRICS, CORPUS, historical=False)
