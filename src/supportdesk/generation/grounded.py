"""One grounded turn: ask, parse (never repair), validate, then deliver or refuse.

This is the grounded-generation layer on top of retrieval. Course slogan: a grounded
assistant is not one that cites; it is one that can be checked, and refuses when the
check fails.

The packet is data. Instructions live only in the system/developer messages. A validation
failure is never retried. At most one retry, and only for a provider error or non-JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .. import config
from ..llm.errors import ProviderError
from ..pipelines.packet import Packet
from ..pipelines.parsing import NotJson, extract_json
from ..types import CaseInput, Chunk

DECISIONS = config.OUTCOMES  # answer | request_info | escalate
REQUIRED = ("outcome", "confidence", "answer", "claims", "citations")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_SENTENCE = re.compile(r"(?:[^.!?\n]|\.(?=\d))*(?:[.!?]|\n)")
_WORD = re.compile(r"[a-z0-9_]{3,}")
_STOP = frozenset((
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "been",
    "have", "has", "had", "not", "but", "you", "your", "into", "than", "then", "them",
    "they", "their", "what", "when", "where", "which", "while", "will", "would",
    "could", "should", "about", "after", "also", "just", "only", "more", "most",
    "some", "any", "can", "may", "our", "out", "use", "used", "using", "via",
))
DISCLOSURE_MARKERS = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "important system message",
    "you are the knowledge assistant",
    "system prompt",
    "maintenance mode",
    "internal reason code",
)


def grounded_schema() -> dict:
    """Closed JSON shape: official support outcomes plus checkable claims."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED),
        "properties": {
            "outcome": {"enum": list(DECISIONS)},
            "confidence": {"enum": list(config.CONFIDENCE)},
            "answer": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "text", "citation_ids"],
                    "properties": {
                        "claim_id": {"type": "string"},
                        "text": {"type": "string"},
                        "citation_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "citations": {"type": "array", "items": {"type": "string"}},
        },
    }


def content_words(text: str) -> list[str]:
    """Content tokens used for claim-support and gold-completeness checks."""
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


def sentences(text: str) -> list[str]:
    found = [match.group(0).strip() for match in _SENTENCE.finditer(text or "")]
    tail = _SENTENCE.sub("", text or "").strip()
    if tail:
        found.append(tail)
    return [sentence for sentence in found if sentence]


def packet_lookup(packet: Packet) -> dict[str, Chunk]:
    """S-labels and chunk ids both resolve to the same chunk."""
    by_id = {c.chunk_id: c for c in packet.chunks}
    for label, cid in packet.labels.items():
        if cid in by_id:
            by_id[label] = by_id[cid]
    return by_id


def format_evidence(packet: Packet) -> str:
    """JSON the model is shown, so ids travel intact."""
    label_of = {cid: label for label, cid in packet.labels.items()}
    return json.dumps({
        "chunks": [
            {
                "label": label_of.get(c.chunk_id, c.chunk_id),
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "title": c.title,
                "text": c.text,
            }
            for c in packet.chunks
        ]
    }, ensure_ascii=False)


def build_grounded_messages(question: str, packet: Packet, *, system: str, developer: str) -> list[dict]:
    """Three roles (system, developer, user); developer text is folded into system so any OpenAI-compatible
    gateway accepts the request."""
    return [
        {"role": "system", "content": f"{system.strip()}\n\n{developer.strip()}"},
        {"role": "user", "content": (
            f"<question>{question}</question>\n"
            f"<evidence>{format_evidence(packet)}</evidence>"
        )},
    ]


def case_question(case: CaseInput) -> str:
    return "\n\n".join(part for part in (case.title.strip(), case.body.strip()) if part)


def _violation(code: str, detail: str, chunk_id: str | None = None) -> dict:
    return {"code": code, "chunk_id": chunk_id, "detail": detail}


def parse_grounded(text: str) -> dict:
    """Never raise, never repair. Not JSON / not an object are terminal."""
    raw = text or ""
    try:
        payload = extract_json(raw)
    except NotJson as exc:
        return {"ok": False, "payload": None, "violations": [
            _violation("not_json", str(exc))], "raw": raw}
    if not isinstance(payload, dict):
        return {"ok": False, "payload": None, "violations": [
            _violation("not_object", "JSON root is not an object")], "raw": raw}
    violations = []
    for key in REQUIRED:
        if key not in payload:
            violations.append(_violation("missing_field", f"missing {key!r}"))
    extra = sorted(set(payload) - set(REQUIRED))
    for key in extra:
        violations.append(_violation("unknown_field", f"unknown field {key!r}"))
    outcome = payload.get("outcome")
    if outcome is not None and outcome not in DECISIONS:
        violations.append(_violation("bad_outcome",
                                    f"outcome must be one of {DECISIONS}, got {outcome!r}"))
    confidence = payload.get("confidence")
    if confidence is not None and confidence not in config.CONFIDENCE:
        violations.append(_violation("bad_confidence",
                                    f"confidence must be one of {config.CONFIDENCE}, got {confidence!r}"))
    claims = payload.get("claims")
    if claims is not None and not isinstance(claims, list):
        violations.append(_violation("bad_type", "claims must be a list"))
    citations = payload.get("citations")
    if citations is not None and not isinstance(citations, list):
        violations.append(_violation("bad_type", "citations must be a list"))
    if not (isinstance(payload.get("answer"), str) and payload.get("answer").strip()):
        violations.append(_violation("empty_answer", "answer must be a non-empty string"))
    return {"ok": not violations, "payload": payload, "violations": violations, "raw": raw}


def resolve_citation(token: str, lookup: dict[str, Chunk]) -> str | None:
    chunk = lookup.get(str(token))
    return chunk.chunk_id if chunk is not None else None


def claim_supported(text: str, cited_texts: list[str], *, threshold: float = 0.5) -> bool:
    """A claim is supported when most of its content words appear in the cited passages.

    The original measures used annotated aliases + subjects. This corpus has selected_spans, not a fact
    index, so support is lexical overlap with the cited packet text only.
    """
    words = content_words(text)
    if not words or not cited_texts:
        return False
    blob = " ".join(cited_texts).lower()
    return (sum(w in blob for w in words) / len(words)) >= threshold


def validate_grounded(payload: dict, packet: Packet) -> list[dict]:
    """Every way the answer fails to be grounded. Do not repair anything."""
    lookup = packet_lookup(packet)
    packet_ids = {c.chunk_id for c in packet.chunks}
    violations: list[dict] = []
    seen: set[str] = set()
    citations = [str(c) for c in (payload.get("citations") or [])]
    for token in citations:
        cid = resolve_citation(token, lookup)
        if cid is None:
            violations.append(_violation("citation_not_retrieved",
                                        "cited id was not in this packet", token))
            continue
        if cid in seen:
            violations.append(_violation("duplicate_citation",
                                        "the same chunk cited twice", cid))
            continue
        seen.add(cid)
    outcome = payload.get("outcome")
    answer = payload.get("answer")
    if outcome == "answer" and payload.get("confidence") == "low":
        violations.append(_violation("low_confidence_answer",
                                    "do not answer with low confidence; escalate instead"))
    if outcome in ("request_info", "escalate") and citations:
        violations.append(_violation("citation_with_non_answer",
                                    f"{outcome} must not cite evidence as if it were an answer"))
    if outcome == "answer" and not citations:
        violations.append(_violation("answer_without_citation",
                                    "it answered and cited nothing"))
    if not (isinstance(answer, str) and answer.strip()):
        violations.append(_violation("empty_answer", "the user-facing message is empty"))
    if outcome in ("request_info", "escalate") and (payload.get("claims") or []):
        violations.append(_violation("claims_with_non_answer",
                                    f"{outcome} must have an empty claims list"))
    listed = {resolve_citation(t, lookup) for t in citations}
    listed.discard(None)
    if outcome != "answer":
        blob = " ".join([
            str(answer or ""),
            *[str(c.get("text") or "") for c in (payload.get("claims") or []) if isinstance(c, dict)],
        ]).lower()
        if any(marker in blob for marker in DISCLOSURE_MARKERS):
            evidence = format_evidence(packet).lower()
            if any(marker in evidence for marker in DISCLOSURE_MARKERS) or "system prompt" in blob:
                violations.append(_violation("instruction_echo",
                                            "answer repeats an instruction found in the packet"))
        return violations
    for claim in payload.get("claims") or []:
        if not isinstance(claim, dict):
            violations.append(_violation("bad_type", "claim is not an object"))
            continue
        ids = [str(x) for x in (claim.get("citation_ids") or [])]
        resolved = [resolve_citation(t, lookup) for t in ids]
        if not ids or any(c is None for c in resolved):
            violations.append(_violation("claim_without_citation",
                                        "claim has no in-packet citation",
                                        claim.get("claim_id")))
            continue
        if any(c not in listed for c in resolved if c):
            violations.append(_violation("claim_citation_not_listed",
                                        "claim cites an id missing from citations",
                                        claim.get("claim_id")))
        texts = [lookup[c].text for c in resolved if c and c in packet_ids]
        if not claim_supported(str(claim.get("text") or ""), texts):
            violations.append(_violation("claim_not_supported",
                                        "cited passages do not carry the claim's words",
                                        claim.get("claim_id")))
    blob = " ".join([
        str(answer or ""),
        *[str(c.get("text") or "") for c in (payload.get("claims") or []) if isinstance(c, dict)],
    ]).lower()
    if any(marker in blob for marker in DISCLOSURE_MARKERS):
        evidence = format_evidence(packet).lower()
        if any(marker in evidence for marker in DISCLOSURE_MARKERS) or "system prompt" in blob:
            violations.append(_violation("instruction_echo",
                                        "answer repeats an instruction found in the packet"))
    return violations


def annotate_claims(payload: dict, packet: Packet) -> list[dict]:
    """Attach support flags used by the metrics (scored even when the turn is refused)."""
    lookup = packet_lookup(packet)
    packet_ids = {c.chunk_id for c in packet.chunks}
    out = []
    for claim in payload.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        resolved = [resolve_citation(str(t), lookup) for t in (claim.get("citation_ids") or [])]
        texts = [lookup[c].text for c in resolved if c and c in packet_ids]
        text = str(claim.get("text") or "")
        out.append({
            "claim_id": claim.get("claim_id"),
            "text": text,
            "citation_ids": [c for c in resolved if c],
            "asserts_fact": bool(text.strip()),
            "supported": claim_supported(text, texts),
        })
    return out


def normalize_citations(payload: dict, packet: Packet) -> list[str]:
    lookup = packet_lookup(packet)
    out: list[str] = []
    for token in payload.get("citations") or []:
        cid = resolve_citation(str(token), lookup)
        if cid and cid not in out:
            out.append(cid)
    return out


@dataclass
class GroundedTurn:
    case_id: str
    question: str
    outcome: str
    answer: str | None
    confidence: str | None = None
    claims: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    violations: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    packet: list[dict] = field(default_factory=list)
    packet_source_ids: list[str] = field(default_factory=list)
    packet_has_gold: bool | None = None
    evidence_token_count: int = 0
    raw_output: str = ""
    attempts: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "query_id": self.case_id, "question": self.question,
            "outcome": self.outcome, "confidence": self.confidence, "answer": self.answer,
            "claims": self.claims,
            "citations": self.citations, "abstained": self.abstained,
            "abstention_reason": self.abstention_reason, "violations": self.violations,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "packet": self.packet,
            "packet_source_ids": self.packet_source_ids,
            "packet_has_gold": self.packet_has_gold,
            "evidence_token_count": self.evidence_token_count,
            "raw_output": self.raw_output, "attempts": self.attempts, "usage": self.usage,
        }


class ExtractiveAnswerer:
    """Offline model: extract a packet sentence that shares query words.

    It does not reason and it does not check its own citations. A live model is
    used when ``--mode live``. This exists so the turn and the metrics run offline.
    """

    def __init__(self, *, quirks: tuple[str, ...] = ()):
        self.quirks = tuple(quirks)
        self.calls = 0

    def complete(self, messages, *, stage="answer", json_mode=True):
        self.calls += 1
        if "malformed_output" in self.quirks:
            return '{"answer": "the packet says", "citations": [', {"mode": "offline", "cost_usd": 0.0}
        question, packet = _read_request(messages)
        payload = self.answer(question, packet)
        return json.dumps(payload), {"mode": "offline", "cost_usd": 0.0,
                                     "input_tokens": 0, "output_tokens": 0}

    def answer(self, query: str, packet: Packet) -> dict:
        if not packet.chunks:
            return _abstain("insufficient_evidence")
        q_words = set(content_words(query))
        chosen = []
        for chunk in packet.chunks:
            best, score = None, 0
            for sentence in sentences(chunk.text):
                overlap = len(q_words & set(content_words(sentence)))
                if overlap > score:
                    best, score = " ".join(sentence.split()), overlap
            if best and score:
                label = next((lab for lab, cid in packet.labels.items()
                              if cid == chunk.chunk_id), chunk.chunk_id)
                chosen.append((label, chunk.chunk_id, best))
        if not chosen:
            return _abstain("insufficient_evidence")
        claims, citations = [], []
        for i, (label, _cid, sentence) in enumerate(chosen, start=1):
            claims.append({"claim_id": f"c{i}", "text": sentence, "citation_ids": [label]})
            if label not in citations:
                citations.append(label)
        return {
            "outcome": "answer", "confidence": "high",
            "answer": " ".join(sentence for _, _, sentence in chosen),
            "claims": claims, "citations": citations,
        }


def _abstain(_reason: str) -> dict:
    return {
        "outcome": "escalate", "confidence": "low",
        "answer": "The evidence packet does not support concrete guidance; a maintainer should look.",
        "claims": [], "citations": [],
    }


def _read_request(messages: list[dict]) -> tuple[str, Packet]:
    from ..pipelines.packet import Packet as PacketType
    question, chunks = "", []
    for message in messages:
        content = str(message.get("content", ""))
        block = re.search(r"<question>(.*?)</question>", content, re.DOTALL)
        if block:
            question = block.group(1).strip()
        ev = re.search(r"<evidence>(.*?)</evidence>", content, re.DOTALL)
        if ev:
            try:
                body = json.loads(ev.group(1))
            except ValueError:
                body = {"chunks": []}
            for i, row in enumerate(body.get("chunks") or [], start=1):
                cid = row.get("chunk_id") or f"chunk:{i}"
                chunks.append(Chunk(
                    chunk_id=cid, source_id=row.get("source_id") or cid,
                    source_type="documentation", title=row.get("title") or "",
                    text=row.get("text") or ""))
    packet = PacketType(chunks=chunks)
    for i, chunk in enumerate(chunks, start=1):
        packet.labels[f"S{i}"] = chunk.chunk_id
    return question, packet


def empty_turn(case_id: str, question: str, packet: Packet, *, reason: str,
               gold: set[str] | None = None) -> GroundedTurn:
    sources = _packet_sources(packet)
    return GroundedTurn(
        case_id=case_id, question=question, outcome="escalate",
        confidence="low",
        answer="No evidence was retrieved, so a maintainer should look.",
        abstained=True, abstention_reason=reason,
        retrieved_chunk_ids=[c.chunk_id for c in packet.chunks],
        packet=_packet_rows(packet),
        packet_source_ids=sorted(sources),
        packet_has_gold=bool(sources & gold) if gold is not None else None,
        evidence_token_count=packet.token_count,
    )


def _packet_rows(packet: Packet) -> list[dict]:
    return [{"chunk_id": c.chunk_id, "source_id": c.source_id,
             "parent_source_id": c.parent_source_id} for c in packet.chunks]


def _packet_sources(packet: Packet) -> set[str]:
    out = set()
    for chunk in packet.chunks:
        out.add(chunk.source_id)
        if chunk.parent_source_id:
            out.add(chunk.parent_source_id)
    return out


def run_grounded_turn(case: CaseInput, packet: Packet, chat, *,
                      system: str, developer: str, retries: int = 1,
                      gold: set[str] | None = None) -> GroundedTurn:
    """Ask → parse → validate → deliver, or refuse. One retry only for provider / not-JSON."""
    question = case_question(case)
    sources = _packet_sources(packet)
    has_gold = bool(sources & gold) if gold is not None else None
    base = {
        "case_id": case.case_id, "question": question,
        "retrieved_chunk_ids": [c.chunk_id for c in packet.chunks],
        "packet": _packet_rows(packet),
        "packet_source_ids": sorted(sources), "packet_has_gold": has_gold,
        "evidence_token_count": packet.token_count,
    }
    if not packet.chunks:
        return empty_turn(case.case_id, question, packet, reason="empty_packet", gold=gold)
    messages = build_grounded_messages(question, packet, system=system, developer=developer)
    attempts: list[dict] = []
    parsed = None
    usage: dict = {}
    raw = ""
    for attempt in range(1 + max(0, retries)):
        try:
            raw, usage = chat.complete(messages, stage="answer", json_mode=True)
            parsed = parse_grounded(raw)
        except ProviderError as exc:
            attempts.append({"attempt": attempt, "status": "ProviderError", "detail": str(exc)})
            parsed = None
            continue
        codes = [v["code"] for v in parsed["violations"]]
        if "not_json" in codes or "not_object" in codes:
            attempts.append({"attempt": attempt, "status": codes[0]})
            continue
        attempts.append({"attempt": attempt, "status": "ok"})
        break
    if parsed is None or parsed["payload"] is None:
        return GroundedTurn(
            outcome="validation_failed", answer=None, abstained=True,
            abstention_reason="malformed_output",
            violations=[{"code": "malformed_output", "chunk_id": None,
                         "detail": "no usable JSON after retries"}],
            raw_output=raw, attempts=attempts, usage=usage or {}, **base)
    payload = parsed["payload"]
    violations = list(parsed["violations"])
    violations.extend(validate_grounded(payload, packet))
    claims = annotate_claims(payload, packet)
    citations = normalize_citations(payload, packet)
    confidence = payload.get("confidence") if payload.get("confidence") in config.CONFIDENCE else None
    decided = payload.get("outcome") if payload.get("outcome") in DECISIONS else None
    if violations:
        return GroundedTurn(
            outcome="validation_failed", confidence=confidence,
            answer=None, claims=claims, citations=citations,
            abstained=True, abstention_reason="validation_failure",
            violations=violations, raw_output=raw, attempts=attempts, usage=usage or {}, **base)
    reply = str(payload.get("answer") or "").strip() or None
    if decided in ("request_info", "escalate"):
        return GroundedTurn(
            outcome=decided, confidence=confidence, answer=reply,
            claims=[], citations=[], abstained=True,
            abstention_reason=decided,
            violations=[], raw_output=raw, attempts=attempts, usage=usage or {}, **base)
    return GroundedTurn(
        outcome="answer", confidence=confidence, answer=reply,
        claims=claims, citations=citations, abstained=False, abstention_reason=None,
        violations=[], raw_output=raw, attempts=attempts, usage=usage or {}, **base)
