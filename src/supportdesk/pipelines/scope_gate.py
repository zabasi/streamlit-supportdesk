"""Pre-retrieval Streamlit scope gate that wraps the current advanced pipeline."""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import START, StateGraph

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .advanced import build_advanced
from .packet import Packet
from .parsing import ParsedAnswer
from .staged import STAGE_NAMES, StagedPipeline
from .state import answer_to_dict, case_from_dict, packet_to_dict
from .stages.scope_classifier import LLMScopeClassifier, RuleScopeClassifier

_SCOPE_PROMPT = Path("configs/prompts/scope_system.txt")
_OUT_OF_SCOPE_REPLY = (
    "This support assistant is scoped to Streamlit technical issues. "
    "Please send a Streamlit question, error, or minimal Streamlit code example."
)


class ScopeGatedPipeline(StagedPipeline):
    """Run a scope classifier after intake and before the current retrieval pipeline."""

    def __init__(self, *, classifier, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def describe(self) -> dict:
        return {"classifier": type(self.classifier).__name__, **super().describe()}

    def _scope(self, state) -> dict:
        decision = self.classifier(case_from_dict(state["case"]))
        return {"extra": {**state.get("extra", {}), "scope": decision.to_dict()}}

    def _scope_route(self, state) -> str:
        if state["extra"]["scope"]["label"] != "IN_SCOPE":
            return "scope_reject"
        # Scope runs immediately after intake. Preserve any future advanced extra node inserted there.
        return self.nodes()[1][0]

    def _scope_reject(self, state) -> dict:
        packet = Packet()
        answer = ParsedAnswer(
            outcome="request_info",
            user_reply=_OUT_OF_SCOPE_REPLY,
            open_questions=["What Streamlit issue are you trying to solve?"],
            next_step="Ask for a Streamlit-specific problem or minimal example.",
        )
        case = case_from_dict(state["case"])
        summary = self.summarizer(case, state["env"], packet, answer)
        return {
            "queries": [],
            "candidates": [],
            "hits": [],
            "packet": packet_to_dict(packet),
            "generated": answer_to_dict(answer),
            "answer": answer_to_dict(answer),
            "summary": summary,
        }

    @staticmethod
    def _scope_done(state) -> dict:
        return {}

    def add_to(self, graph: StateGraph, *, until: str = "summarize") -> str:
        """Add the gate before query/retrieval while preserving the wrapped pipeline stages."""
        linear = self.nodes()
        names = [name for name, _ in linear]
        if until not in names:
            raise ValueError(f"unknown terminal node {until!r}; nodes: {names}")
        linear = linear[: names.index(until) + 1]

        graph.add_node("intake", self._intake)
        graph.add_node("scope", self._scope)
        graph.add_node("scope_reject", self._scope_reject)
        graph.add_node("scope_done", self._scope_done)
        rest = linear[1:]
        for name, fn in rest:
            graph.add_node(name, fn)

        graph.add_edge(START, "intake")
        graph.add_edge("intake", "scope")
        in_scope_target = rest[0][0]
        graph.add_conditional_edges("scope", self._scope_route, [in_scope_target, "scope_reject"])
        for (a, _), (b, _) in zip(rest, rest[1:]):
            graph.add_edge(a, b)
        graph.add_edge(linear[-1][0], "scope_done")
        graph.add_edge("scope_reject", "scope_done")
        return "scope_done"


def _wrap(base: StagedPipeline, classifier, *, name: str) -> ScopeGatedPipeline:
    """Wrap an already-built pipeline so its current stages remain exactly intact."""
    stages = {stage: getattr(base, stage) for stage in STAGE_NAMES}
    return ScopeGatedPipeline(
        name=name,
        clients=base.clients,
        settings=base.settings,
        classifier=classifier,
        extra_nodes=base.extra_nodes,
        **stages,
    )


def build_advanced_scoped(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Current advanced pipeline with an LLM pre-retrieval scope classifier."""
    return _wrap(
        build_advanced(retriever, clients, settings),
        LLMScopeClassifier(clients, _SCOPE_PROMPT),
        name="advanced",
    )


def build_scope_gate(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Named comparison pipeline: current advanced retrieval plus the LLM scope gate."""
    return _wrap(
        build_advanced(retriever, clients, settings),
        LLMScopeClassifier(clients, _SCOPE_PROMPT),
        name="scope_gate",
    )


def build_scope_gate_rules(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Rule-based comparison using the same current advanced pipeline underneath."""
    return _wrap(
        build_advanced(retriever, clients, settings),
        RuleScopeClassifier(),
        name="scope_gate_rules",
    )


__all__ = [
    "ScopeGatedPipeline",
    "build_advanced_scoped",
    "build_scope_gate",
    "build_scope_gate_rules",
]
