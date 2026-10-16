"""Generation policy v3: one JSON support record, literal report observations, no repair.

Tested 2026-10-16 on 15 annotated development cases with gpt-5.4-mini (medium, 8000
completion tokens). That run used curated gold passages, not retrieval chunks. This
module keeps the same schema and prompt, and adds adapters so a retrieval packet can
fill the same evidence slots.

Contract: the model selects reference IDs; the controller copies original report text.
Parse never repairs. A schema failure withholds the reply.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.cases import EvalCase
from ..llm.errors import ProviderError
from ..pipelines.packet import Packet
from ..pipelines.parsing import NotJson, extract_json
from ..types import CaseInput, Chunk
from .grounded import GroundedTurn
from .policy_context import OUTCOMES, context, reference_map

ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = ROOT / "configs" / "prompts" / "policy_v3_system.txt"
PROMPT_V3B_PATH = ROOT / "configs" / "prompts" / "policy_v3b_system.txt"
PROMPT_V3C_PATH = ROOT / "configs" / "prompts" / "policy_v3c_system.txt"
FROZEN_INPUTS = ROOT / "experiments" / "generation_policy_v3" / "inputs" / "model_inputs.jsonl"
ALLOWED_MODELS = ("gpt-4o-mini", "gpt-5.4-mini", "gpt-5-mini")
INLINE_REF = re.compile(r"\[((?:S|R|H)\d+)\]")
CODE_FENCE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.I | re.S)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_prompt(path: str | Path | None = None) -> str:
    file = Path(path) if path else PROMPT_PATH
    if not file.is_file():
        raise FileNotFoundError(f"policy v3 prompt not found: {file}")
    return file.read_text(encoding="utf-8")


def load_frozen_rows(path: str | Path | None = None) -> list[dict]:
    file = Path(path) if path else FROZEN_INPUTS
    if not file.is_file():
        raise FileNotFoundError(f"policy v3 frozen inputs not found: {file}")
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _text(values=None) -> dict:
    return {"type": "string", **({"enum": values} if values is not None else {"pattern": r"\S"})}


def _array(item: dict) -> dict:
    return {"type": "array", "items": item}


def schema(row: dict) -> dict:
    refs = reference_map(row)
    reports = [k for k, v in refs.items() if v["kind"] == "report" or v.get("role") == "user"]
    evidence = [k for k, v in refs.items() if v["kind"] == "evidence"]
    branches = []
    for outcome in OUTCOMES:
        props = {"outcome": _text([outcome]), "message": _text()}
        if outcome == "request_info":
            props["question"] = _text()
        branches.append(_obj(props))
    return _obj({
        "case_id": _text([row["case_id"]]),
        "reported_observation_refs": {**_array(_text(reports)), "maxItems": 6},
        "workflow_findings": {**_array(_obj({
            "report_ref": _text(reports), "problem": _text(),
            "correction": _text(),
            "evidence_refs": {**_array(_text(evidence)), "minItems": 1},
        })), "maxItems": 4},
        "action": {"anyOf": branches},
        "maintainer_summary": _obj({
            "problem": _text(), "reported_environment": _text(),
            "uncertainties": _array(_text()), "next_step": _text(),
        }),
    })


def check(value, spec: dict, path: str = "$") -> None:
    """Validate the JSON Schema subset this module generates, including anyOf."""
    if "anyOf" in spec:
        for branch in spec["anyOf"]:
            try:
                check(value, branch, path)
                return
            except ValueError:
                pass
        raise ValueError(path + ": no action branch matches")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str)}.get(spec["type"], False)
    if not valid:
        raise ValueError(path + ": wrong type")
    if "enum" in spec and value not in spec["enum"]:
        raise ValueError(path + ": unknown value/reference")
    if "pattern" in spec and not re.search(spec["pattern"], value):
        raise ValueError(path + ": empty text")
    if isinstance(value, str):
        if len(value) < spec.get("minLength", 0) or len(value) > spec.get("maxLength", len(value)):
            raise ValueError(path + ": text outside length allowance")
    if isinstance(value, dict):
        if set(value) != set(spec["required"]):
            raise ValueError(path + ": missing or extra fields")
        for key, item in value.items():
            check(item, spec["properties"][key], path + "." + key)
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            raise ValueError(path + ": too few items")
        if len(value) > spec.get("maxItems", len(value)):
            raise ValueError(path + ": too many items")
        for i, item in enumerate(value):
            check(item, spec["items"], path + f"[{i}]")


def validate(value: dict, row: dict) -> dict:
    check(value, schema(row))
    allowed = set(reference_map(row))
    for ref in INLINE_REF.findall(json.dumps(value, ensure_ascii=False)):
        if ref not in allowed:
            raise ValueError("Unknown inline reference")
    selected = value["reported_observation_refs"]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate reported observation reference")
    return value


def issues(value: dict) -> list[str]:
    """Deterministic warnings. They never rewrite the payload or change the outcome."""
    result = []
    message = value["action"]["message"]
    if value["action"]["outcome"] == "answer" and not re.search(r"\[S\d+\]", message):
        result.append("Answer has no inline evidence citation")
    for code in CODE_FENCE.findall(message):
        try:
            ast.parse(code)
        except SyntaxError:
            result.append("Proposed Python has invalid syntax (not executed)")
    return result


def request(row: dict, model: str, *, prompt: str | None = None,
            prompt_path: str | Path | None = None) -> dict:
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model!r} outside the allowed set {ALLOWED_MODELS}")
    result = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt if prompt is not None else load_prompt(prompt_path)},
            {"role": "user", "content": context(row)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "supportdesk_action_policy_v3", "strict": True, "schema": schema(row)}},
        "max_completion_tokens": 8000,
    }
    if model.startswith("gpt-5"):
        result["reasoning_effort"] = "medium"
    else:
        result["temperature"] = 0
    return result


def resolve_report_observations(value: dict, row: dict) -> list[dict]:
    """Controller-owned literal text; the model only selected allowed reference IDs."""
    validate(value, row)
    refs = reference_map(row)
    result = []
    for ref in value["reported_observation_refs"]:
        item = refs[ref]
        result.append({
            "reference": ref,
            "quote": item["text"],
            "text_sha256": sha(item["text"].encode("utf8")),
            "origin": "original_report" if item["kind"] == "report" else "user_conversation",
        })
    return result


def rendered(value: dict, row: dict) -> dict:
    validate(value, row)
    action = value["action"]
    reply = action["message"]
    if action["outcome"] == "request_info":
        reply += "\n\n" + action["question"]
    return {
        "case_id": value["case_id"],
        "outcome": action["outcome"],
        "user_reply": reply,
        "reported_observations": resolve_report_observations(value, row),
        "workflow_findings": value["workflow_findings"],
        "maintainer_summary": value["maintainer_summary"],
        "warnings": issues(value),
        "tracker_writes": 0,
        "status": "proposal_requires_content_review",
    }


def sources_from_packet(packet: Packet) -> list[dict]:
    """Turn a retrieval packet into the labelled-passage list policy v3 expects."""
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for chunk in packet.chunks:
        sid = chunk.source_id
        if sid not in grouped:
            grouped[sid] = {
                "source_id": sid,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "url": chunk.url or "",
                "version": chunk.version,
                "passages": [],
            }
            order.append(sid)
        grouped[sid]["passages"].append({
            "span_id": chunk.chunk_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text": chunk.text,
        })
    return [grouped[sid] for sid in order]


def sources_from_spans(case: EvalCase) -> list[dict]:
    """Reviewer-selected gold spans (the original v3 evidence, not retrieval)."""
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for i, span in enumerate(case.evidence_spans or [], 1):
        sid = span.get("source_id")
        if not sid:
            continue
        if sid not in grouped:
            grouped[sid] = {
                "source_id": sid,
                "source_type": "documentation",
                "title": sid,
                "passages": [],
            }
            order.append(sid)
        grouped[sid]["passages"].append({
            "span_id": span.get("evidence_id") or f"{sid}:{i}",
            "start_char": span.get("start_char"),
            "end_char": span.get("end_char"),
            "text": span.get("text") or "",
        })
    return [grouped[sid] for sid in order]


def row_from_case(case: EvalCase | CaseInput, sources: list[dict]) -> dict:
    incoming = case.input if isinstance(case, EvalCase) else case
    return {
        "case_id": incoming.case_id,
        "input": {
            "case_id": incoming.case_id,
            "title": incoming.title,
            "body": incoming.body,
            "created_at": incoming.created_at,
            "history": list(incoming.history or []),
            "environment": dict(incoming.environment or {}),
        },
        "sources": sources,
    }


def evidence_lookup(row: dict) -> dict[str, dict]:
    """S-label → passage + source_id, in the same order the prompt numbered them."""
    out = {}
    n = 0
    for source in row["sources"]:
        for passage in source["passages"]:
            n += 1
            out[f"S{n}"] = {
                "source_id": source["source_id"],
                "span_id": passage.get("span_id"),
                "text": passage.get("text") or "",
            }
    return out


def citations_from_message(message: str, row: dict) -> list[str]:
    lookup = evidence_lookup(row)
    seen: list[str] = []
    for ref in INLINE_REF.findall(message or ""):
        item = lookup.get(ref)
        if not item:
            continue
        cid = item.get("span_id") or ref
        if cid not in seen:
            seen.append(cid)
    return seen


def packet_source_ids(packet: Packet) -> list[str]:
    out = []
    for chunk in packet.chunks:
        if chunk.source_id not in out:
            out.append(chunk.source_id)
        if chunk.parent_source_id and chunk.parent_source_id not in out:
            out.append(chunk.parent_source_id)
    return out


@dataclass
class PolicyTurn:
    case_id: str
    question: str
    outcome: str
    answer: str | None
    user_reply: str | None = None
    reported_observations: list[dict] = field(default_factory=list)
    workflow_findings: list[dict] = field(default_factory=list)
    maintainer_summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    packet: list[dict] = field(default_factory=list)
    packet_source_ids: list[str] = field(default_factory=list)
    packet_has_gold: bool | None = None
    evidence_token_count: int = 0
    raw_output: str = ""
    attempts: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    evidence_kind: str = "packet"

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "query_id": self.case_id, "question": self.question,
            "outcome": self.outcome, "answer": self.answer, "user_reply": self.user_reply,
            "confidence": None, "claims": self.claims, "citations": self.citations,
            "reported_observations": self.reported_observations,
            "workflow_findings": self.workflow_findings,
            "maintainer_summary": self.maintainer_summary,
            "warnings": self.warnings, "violations": self.violations,
            "retrieved_chunk_ids": self.retrieved_chunk_ids, "packet": self.packet,
            "packet_source_ids": self.packet_source_ids,
            "packet_has_gold": self.packet_has_gold,
            "evidence_token_count": self.evidence_token_count,
            "raw_output": self.raw_output, "attempts": self.attempts, "usage": self.usage,
            "evidence_kind": self.evidence_kind,
            "abstained": self.outcome != "answer",
            "abstention_reason": None if self.outcome == "answer" else self.outcome,
        }

    def to_grounded(self) -> GroundedTurn:
        body = self.to_dict()
        return GroundedTurn(
            case_id=self.case_id, question=self.question, outcome=self.outcome,
            answer=self.answer, claims=self.claims, citations=self.citations,
            abstained=body["abstained"], abstention_reason=body["abstention_reason"],
            violations=self.violations, retrieved_chunk_ids=self.retrieved_chunk_ids,
            packet=self.packet, packet_source_ids=self.packet_source_ids,
            packet_has_gold=self.packet_has_gold,
            evidence_token_count=self.evidence_token_count,
            raw_output=self.raw_output, attempts=self.attempts, usage=self.usage,
        )


def _packet_rows(packet: Packet) -> list[dict]:
    return [{"chunk_id": c.chunk_id, "source_id": c.source_id,
             "parent_source_id": c.parent_source_id} for c in packet.chunks]


def parse_policy(text: str) -> dict:
    """Never raise, never repair. Not JSON / not an object are terminal."""
    raw = text or ""
    try:
        payload = extract_json(raw)
    except NotJson as exc:
        return {"ok": False, "payload": None,
                "violations": [{"code": "not_json", "detail": str(exc)}], "raw": raw}
    if not isinstance(payload, dict):
        return {"ok": False, "payload": None,
                "violations": [{"code": "not_object", "detail": "JSON root is not an object"}],
                "raw": raw}
    return {"ok": True, "payload": payload, "violations": [], "raw": raw}


def run_policy_turn(case: EvalCase | CaseInput, sources: list[dict], chat, *,
                    packet: Packet | None = None, model: str = "gpt-4o-mini",
                    retries: int = 0, gold: set[str] | None = None,
                    evidence_kind: str = "packet",
                    prompt: str | None = None,
                    prompt_path: str | Path | None = None) -> PolicyTurn:
    """Ask → parse (never repair) → validate schema → hydrate report quotes, or withhold."""
    incoming = case.input if isinstance(case, EvalCase) else case
    question = "\n\n".join(p for p in (incoming.title.strip(), incoming.body.strip()) if p)
    row = row_from_case(case, sources)
    packet = packet or Packet()
    sources_ids = packet_source_ids(packet) if packet.chunks else [
        s["source_id"] for s in sources if s.get("source_id")]
    has_gold = bool(set(sources_ids) & gold) if gold is not None else None
    base = dict(
        case_id=incoming.case_id, question=question,
        retrieved_chunk_ids=[c.chunk_id for c in packet.chunks],
        packet=_packet_rows(packet),
        packet_source_ids=list(sources_ids),
        packet_has_gold=has_gold,
        evidence_token_count=packet.token_count,
        evidence_kind=evidence_kind,
    )
    if not sources or not any(s.get("passages") for s in sources):
        return PolicyTurn(
            outcome="escalate",
            answer="No evidence was supplied, so a maintainer should look.",
            user_reply="No evidence was supplied, so a maintainer should look.",
            **base)
    built = request(row, model, prompt=prompt, prompt_path=prompt_path)
    messages = built["messages"]
    extra = {k: built[k] for k in ("reasoning_effort", "response_format") if k in built}
    attempts: list[dict] = []
    parsed = None
    usage: dict = {}
    raw = ""
    for attempt in range(1 + max(0, retries)):
        try:
            try:
                raw, usage = chat.complete(messages, stage="answer", json_mode=True, extra=extra)
            except TypeError:
                raw, usage = chat.complete(messages, stage="answer", json_mode=True)
            parsed = parse_policy(raw)
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
        return PolicyTurn(
            outcome="validation_failed", answer=None,
            violations=[{"code": "malformed_output",
                         "detail": "no usable JSON after retries"}],
            raw_output=raw, attempts=attempts, usage=usage or {}, **base)
    try:
        record = rendered(parsed["payload"], row)
    except ValueError as exc:
        return PolicyTurn(
            outcome="validation_failed", answer=None,
            violations=[{"code": "contract_invalid", "detail": str(exc)}],
            raw_output=raw, attempts=attempts, usage=usage or {}, **base)
    reply = record["user_reply"]
    citations = citations_from_message(parsed["payload"]["action"]["message"], row)
    claims = []
    for i, finding in enumerate(record["workflow_findings"], 1):
        claims.append({
            "claim_id": f"wf{i}",
            "text": f"{finding.get('problem', '')} {finding.get('correction', '')}".strip(),
            "citation_ids": list(finding.get("evidence_refs") or []),
            "asserts_fact": True,
            "supported": None,
        })
    return PolicyTurn(
        outcome=record["outcome"], answer=reply, user_reply=reply,
        reported_observations=record["reported_observations"],
        workflow_findings=record["workflow_findings"],
        maintainer_summary=record["maintainer_summary"],
        warnings=record["warnings"], citations=citations, claims=claims,
        raw_output=raw, attempts=attempts, usage=usage or {}, **base)
