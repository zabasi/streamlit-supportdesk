"""Passage-level retrieval: is the gold passage itself (start_char:end_char) in the retrieved chunks,
not just some chunk of the same document?"""
import json

import pytest

from supportdesk.eval.metrics import evaluate_retrieval, passage_coverage, passage_recall

TEXT = "Intro text here.\n\n### Caveats and limitations\n\nSession State is tied to a WebSocket connection.\n"
START = TEXT.index("### Caveats")
SPAN = {"source_id": "doc:a", "start_char": START, "end_char": len(TEXT) - 1, "text": TEXT[START:len(TEXT) - 1]}
WORDS_FROM = TEXT.index("Caveats")  # the first word of the passage


def chunk(source, start, end):
    return {"source_id": source, "start_char": start, "end_char": end}


def test_a_passage_inside_one_chunk_is_fully_covered_and_markup_outside_it_does_not_count():
    assert passage_coverage([chunk("doc:a", 0, len(TEXT))], SPAN) == 1.0
    assert passage_coverage([chunk("doc:a", WORDS_FROM, len(TEXT))], SPAN) == 1.0  # "### " left outside


def test_coverage_adds_up_the_parts_and_ignores_other_documents():
    middle = TEXT.index("tied")
    first, second = chunk("doc:a", 0, middle), chunk("doc:a", middle, len(TEXT))
    assert 0 < passage_coverage([first], SPAN) < 1 and 0 < passage_coverage([second], SPAN) < 1
    assert passage_coverage([first, second], SPAN) == pytest.approx(1.0)
    assert passage_coverage([first, first], SPAN) == passage_coverage([first], SPAN)  # overlaps count once
    assert passage_coverage([chunk("doc:b", 0, len(TEXT))], SPAN) == 0.0
    assert passage_coverage([chunk("doc:a", 0, START)], SPAN) == 0.0  # the intro, not the passage


def test_a_passage_is_found_when_enough_of_it_is_covered():
    half = TEXT.index("tied")
    partial = [chunk("doc:a", 0, half)]
    share = passage_coverage(partial, SPAN)
    assert passage_recall(partial, [SPAN], threshold=share) == 1.0
    assert passage_recall(partial, [SPAN], threshold=min(1.0, share + 0.01)) == 0.0


def test_a_document_can_be_found_while_its_passage_is_not():
    from supportdesk.eval.cases import EvalCase
    from supportdesk.types import CaseInput
    case = EvalCase(case_id="c", family_id="f", split="dev", category="answerable",
                    input=CaseInput(case_id="c", title="t", body="b"), gold_evidence=["doc:a"],
                    acceptable_outcomes=["answer"], evidence_spans=[SPAN])
    intro_only = {**chunk("doc:a", 0, START), "chunk_id": "doc:a:000", "parent_source_id": None, "rank": 1}
    row = {"retrieved": [intro_only], "packet": [intro_only], "outcome": "retrieval_only", "usage": {}}
    m = evaluate_retrieval([(case, row)], (1,), passage_threshold=0.8)
    assert m["recall_at_k"]["1"] == 1.0  # the document is there...
    assert m["passage_recall_at_k"]["1"] == 0.0 and m["packet_passage_recall"] == 0.0  # ...the passage is not
    assert "passage_recall_at_k" not in evaluate_retrieval([(case, row)], (1,))  # off without a threshold


def test_retrieval_runs_record_chunk_offsets(tmp_path, capsys):
    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]
    assert main(["index", *base]) == 0
    assert main(["eval-retrieval", *base, "--run-id", "r"]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in (tmp_path / "art" / "runs" / "r" / "results.jsonl").read_text().splitlines()]
    hits = [h for row in rows for h in row["retrieved"] + row["packet"]]
    assert hits and all(isinstance(h["start_char"], int) and h["end_char"] > h["start_char"] for h in hits)
