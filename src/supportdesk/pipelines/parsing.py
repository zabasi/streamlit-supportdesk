"""Turns the model's text into a checked answer: valid JSON, a known outcome, and citations that point
at evidence the model was actually shown. The model's labels are never taken on trust."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .. import config
from .packet import Packet


class NotJson(ValueError):
    """The output was not parseable JSON. Retryable."""


class PayloadError(ValueError):
    """Parseable JSON with the wrong shape. Not retried: asking again until it fits is guessing."""


@dataclass
class ParsedAnswer:
    outcome: str
    user_reply: str
    known_facts: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_step: str | None = None
    citations: list[str] = field(default_factory=list)  # chunk ids, all inside the packet
    violations: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)  # model-call log, filled by the generator
    raw_output: str = ""  # the model's text as received, for inspection
    confidence: str | None = None  # high | medium | low, as the model reported it


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    cleaned = _FENCE.sub("", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise NotJson("no JSON object in model output")
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise NotJson(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise NotJson("JSON root is not an object")
    return value


def _labels(value, packet: Packet, violations: list[str]) -> list[str]:
    out: list[str] = []
    for label in value or []:
        cid = packet.labels.get(str(label))
        if cid is None:
            if "citation_not_retrieved" not in violations:
                violations.append("citation_not_retrieved")
        elif cid not in out:
            out.append(cid)
    return out


def parse_answer(text: str, packet: Packet) -> ParsedAnswer:
    payload = extract_json(text)
    outcome = payload.get("outcome")
    if outcome not in config.OUTCOMES:
        raise PayloadError(f"outcome must be one of {config.OUTCOMES}, got {outcome!r}")
    reply = payload.get("user_reply")
    if not isinstance(reply, str) or not reply.strip():
        raise PayloadError("user_reply must be a non-empty string")
    violations: list[str] = []
    facts = []
    for item in payload.get("known_facts") or []:
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            raise PayloadError("known_facts items need a string 'statement'")
        facts.append({"statement": item["statement"],
                      "evidence": _labels(item.get("evidence"), packet, violations)})
    questions = [str(q) for q in (payload.get("open_questions") or [])]
    citations = _labels(payload.get("citations"), packet, violations)
    if outcome == "answer" and not citations:
        violations.append("answer_without_citation")
    if outcome == "answer" and any(not f["evidence"] for f in facts):
        violations.append("claim_without_citation")
    next_step = payload.get("next_step")
    confidence = payload.get("confidence")
    if confidence is not None and confidence not in config.CONFIDENCE:
        raise PayloadError(f"confidence must be one of {config.CONFIDENCE}, got {confidence!r}")
    if outcome == "answer" and confidence == "low":
        # Not confident enough to answer: a maintainer verifies it instead of the user receiving it.
        outcome = "escalate"
        violations.append("low_confidence_answer_escalated")
    return ParsedAnswer(outcome, reply.strip(), facts, questions,
                        next_step if isinstance(next_step, str) else None, citations, violations,
                        confidence=confidence)
