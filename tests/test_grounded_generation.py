"""Grounded generation: parse never repairs, validation refuses, metrics are checkable."""
from __future__ import annotations

import json

from supportdesk.eval.cases import case_from_dict
from supportdesk.eval.grounded_metrics import evaluate_grounded, gold_completeness
from supportdesk.generation.grounded import (
    ExtractiveAnswerer,
    Packet,
    build_grounded_messages,
    claim_supported,
    parse_grounded,
    run_grounded_turn,
    validate_grounded,
)
from supportdesk.types import CaseInput, Chunk


def _chunk(cid="doc:a:000", text="Session state keeps widget values across reruns.",
           source="doc:a"):
    return Chunk(chunk_id=cid, source_id=source, source_type="documentation",
                 title="session_state", text=text)


def _packet(chunks=None):
    chunks = chunks or [_chunk()]
    packet = Packet(chunks=list(chunks))
    for i, chunk in enumerate(packet.chunks, start=1):
        packet.labels[f"S{i}"] = chunk.chunk_id
        packet.token_count += 8
    return packet


def _case():
    return CaseInput(case_id="DEV-1", title="session state lost",
                     body="widget value disappears after expander")


def test_parse_clean_and_not_json():
    ok = parse_grounded(json.dumps({
        "outcome": "answer", "confidence": "high",
        "answer": "Use a key.", "claims": [{"claim_id": "c1", "text": "Use a key.",
                                           "citation_ids": ["S1"]}],
        "citations": ["S1"],
    }))
    assert ok["ok"] and ok["payload"]["outcome"] == "answer"
    bad = parse_grounded("Sorry, here is prose.")
    assert not bad["ok"] and bad["violations"][0]["code"] == "not_json"
    extra = parse_grounded(json.dumps({
        "outcome": "escalate", "confidence": "low",
        "answer": "A maintainer should look.", "claims": [], "citations": [],
        "bonus": 1,
    }))
    assert not extra["ok"] and any(v["code"] == "unknown_field" for v in extra["violations"])


def test_citation_must_be_in_packet():
    packet = _packet()
    payload = {"outcome": "answer", "confidence": "high",
               "answer": "hi", "claims": [{"claim_id": "c1", "text": "hi",
                                          "citation_ids": ["S9"]}],
               "citations": ["S9"]}
    codes = [v["code"] for v in validate_grounded(payload, packet)]
    assert "citation_not_retrieved" in codes


def test_claim_support_needs_cited_words():
    assert claim_supported("session state keeps widget values",
                           ["Session state keeps widget values across reruns."])
    assert not claim_supported("refunds take fourteen days",
                               ["Session state keeps widget values across reruns."])


def test_empty_packet_abstains_without_calling():
    class Boom:
        def complete(self, *a, **k):
            raise AssertionError("empty packet must not call the model")

    turn = run_grounded_turn(_case(), Packet(), Boom(), system="s", developer="d",
                             retries=0, gold={"doc:a"})
    assert turn.outcome == "escalate" and turn.abstention_reason == "empty_packet"


def test_validation_failure_wipes_answer():
    class Chat:
        def complete(self, messages, **k):
            return json.dumps({
                "outcome": "answer", "confidence": "high",
                "answer": "Refunds take 14 days.",
                "claims": [{"claim_id": "c1", "text": "Refunds take 14 days.",
                            "citation_ids": ["S1"]}],
                "citations": ["S1"],
            }), {"cost_usd": 0.0}

    turn = run_grounded_turn(_case(), _packet(), Chat(), system="sys", developer="dev",
                             retries=0, gold={"doc:a"})
    assert turn.outcome == "validation_failed"
    assert turn.answer is None
    assert turn.abstention_reason == "validation_failure"
    assert any(v["code"] == "claim_not_supported" for v in turn.violations)
    assert turn.claims and turn.claims[0]["supported"] is False


def test_extractive_quotes_packet_and_metrics_read():
    packet = _packet()
    turn = run_grounded_turn(_case(), packet, ExtractiveAnswerer(), system="s",
                             developer="d", retries=0, gold={"doc:a"})
    assert turn.outcome == "answer"
    assert turn.citations == ["doc:a:000"]
    assert turn.claims[0]["supported"] is True
    case = case_from_dict({
        "case_id": "DEV-1", "family_id": "f", "split": "dev", "category": "answerable",
        "input": {"case_id": "DEV-1", "title": "t", "body": "b"},
        "acceptable_outcomes": ["answer"], "gold_evidence": ["doc:a"],
        "evidence_spans": [{"source_id": "doc:a", "start_char": 0, "end_char": 13,
                            "text": "Session state"}],
    })
    metrics = evaluate_grounded([(case, turn.to_dict())])
    h = metrics["headline"]
    assert h["claim_support_rate"] == 1.0
    assert h["citation_precision"] == 1.0
    assert h["citations_outside_packet"] == 0
    assert h["answered_after_failure"] == 0
    assert h["answer_when_packet_has_gold"] == 1.0
    assert h["outcome_accuracy"] == 1.0
    assert gold_completeness(turn.answer, case.evidence_spans) > 0
    messages = build_grounded_messages("q", packet, system="SYS", developer="DEV")
    assert "<question>q</question>" in messages[-1]["content"]
    assert "<evidence>" in messages[-1]["content"]


def test_abstention_f1_refuses_when_packet_has_no_gold():
    packet = _packet()
    turn = run_grounded_turn(_case(), packet, ExtractiveAnswerer(), system="s",
                             developer="d", retries=0, gold={"doc:other"})
    case = case_from_dict({
        "case_id": "DEV-1", "family_id": "f", "split": "dev", "category": "answerable",
        "input": {"case_id": "DEV-1", "title": "t", "body": "b"},
        "acceptable_outcomes": ["answer"], "gold_evidence": ["doc:other"],
    })
    h = evaluate_grounded([(case, turn.to_dict())])["headline"]
    # packet has no gold, but extractive still answered → missed a required refusal
    assert turn.outcome == "answer"
    assert h["abstention_recall"] == 0.0
    assert h["abstention_precision"] is None  # it never refused, so precision is undefined
    assert h["answer_when_packet_has_gold"] is None  # no gold in the packet, so this rate does not apply


def test_request_info_needs_no_citations():
    class Chat:
        def complete(self, messages, **k):
            return json.dumps({
                "outcome": "request_info", "confidence": "medium",
                "answer": "Which Streamlit version is installed?",
                "claims": [], "citations": [],
            }), {"cost_usd": 0.0}

    turn = run_grounded_turn(_case(), _packet(), Chat(), system="s", developer="d", retries=0)
    assert turn.outcome == "request_info"
    assert turn.answer and "version" in turn.answer.lower()
    case = case_from_dict({
        "case_id": "DEV-1", "family_id": "f", "split": "dev", "category": "ambiguous",
        "input": {"case_id": "DEV-1", "title": "t", "body": "b"},
        "acceptable_outcomes": ["request_info"], "gold_evidence": ["doc:a"],
    })
    h = evaluate_grounded([(case, turn.to_dict())])["headline"]
    assert h["outcome_accuracy"] == 1.0
    assert h["by_category"]["ambiguous"]["outcome_accuracy"] == 1.0
    matrix = h["confusion_matrix"]["counts"]
    assert matrix["request_info"]["request_info"] == 1
    assert matrix["answer"]["request_info"] == 0
    assert matrix["escalate"]["request_info"] == 0
