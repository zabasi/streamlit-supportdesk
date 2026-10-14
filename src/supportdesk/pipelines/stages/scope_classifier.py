"""Classify a report before retrieval so clearly unrelated requests never enter the RAG path."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ...types import CaseInput
from ..parsing import NotJson, extract_json
from .clean_query import clean_report

_STREAMLIT_API = re.compile(r"\bst\.[A-Za-z_]\w*\b")
_STREAMLIT_TERMS = re.compile(
    r"\b(streamlit|session[_ ]state|st\.session_state|widget|multipage|rerun|fragment|"
    r"data_editor|selectbox|text_input|text_area|number_input|slider|st\.form|callback)\b",
    re.IGNORECASE,
)
_GREETING_ONLY = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|سلام|درود|ممنون|مرسی)[.!?\s]*$", re.IGNORECASE
)


@dataclass(frozen=True)
class ScopeDecision:
    """Plain-data classification result safe to store in a trace or checkpoint."""

    label: str  # IN_SCOPE | OUT_OF_SCOPE
    reason: str
    method: str  # rule | llm | fail_open

    @property
    def in_scope(self) -> bool:
        return self.label == "IN_SCOPE"

    def to_dict(self) -> dict:
        return asdict(self)


class RuleScopeClassifier:
    """Free conservative baseline: reject only obvious non-Streamlit input; ambiguous input passes."""

    def __call__(self, case: CaseInput) -> ScopeDecision:
        text = "\n".join(x for x in (case.title, clean_report(case.body)) if x).strip()
        if _STREAMLIT_API.search(text):
            return ScopeDecision("IN_SCOPE", "contains a Streamlit st.* API", "rule")
        if _STREAMLIT_TERMS.search(text):
            return ScopeDecision("IN_SCOPE", "contains Streamlit/support terminology", "rule")
        if not text or _GREETING_ONLY.fullmatch(text):
            return ScopeDecision("OUT_OF_SCOPE", "empty or greeting-only request", "rule")
        # False negatives are more costly than false positives here: uncertain requests continue.
        return ScopeDecision("IN_SCOPE", "ambiguous; fail open to retrieval", "rule")


def scope_from_text(text: str, method: str = "llm") -> ScopeDecision:
    """The scope decision in a model's JSON answer ({"label": ..., "reason": ...}); unusable output fails
    open to IN_SCOPE, since sending an unrelated report to retrieval costs less than refusing a real one."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return ScopeDecision("IN_SCOPE", "classifier output was unusable; fail open", "fail_open")
    if not isinstance(payload, dict):
        return ScopeDecision("IN_SCOPE", "classifier output was not an object; fail open", "fail_open")
    label = str(payload.get("label", "")).strip().upper()
    reason = str(payload.get("reason", "")).strip()
    if label not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
        return ScopeDecision("IN_SCOPE", "classifier label was invalid; fail open", "fail_open")
    return ScopeDecision(label, reason or "model classification", method)


class LLMScopeClassifier:
    """Binary Streamlit support classifier using the shared chat gateway and cost ledger."""

    def __init__(self, clients, prompt_path: Path):
        self.clients = clients
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def __call__(self, case: CaseInput) -> ScopeDecision:
        report = "\n".join(x for x in (case.title, clean_report(case.body)) if x).strip()
        # Preserve Streamlit APIs from code: clean_report deliberately keeps st.* lines.
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": report},
        ]
        text, _ = self.clients.chat.complete(messages, stage="scope_classify", json_mode=True)
        return scope_from_text(text, "llm")


__all__ = ["LLMScopeClassifier", "RuleScopeClassifier", "ScopeDecision", "scope_from_text"]
