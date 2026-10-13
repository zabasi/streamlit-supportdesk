"""Experimental pre-retrieval scope gate for the advanced pipeline."""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import START, StateGraph

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .packet import Packet
from .parsing import ParsedAnswer
from .staged import StagedPipeline
from .state import answer_to_dict, case_from_dict, packet_to_dict
from .stages import default_stages
from .stages.clean_query import CleanQuery
from .stages.llm_rerank import LLMReranker
from .stages.scope_classifier import LLMScopeClassifier, RuleScopeClassifier

_SCOPE_PROMPT = Path("configs/prompts/scope_system.txt")
_OUT_OF_SCOPE_REPLY = (
    "This support assistant is scoped to Streamlit technical issues. "
    "Please send a Streamlit question, error, or minimal Streamlit code example."
)


class ScopeGatedPipeline(StagedPipeline):
    """Run a scope classifier after intake and skip retrieval for clearly unrelated reports."""

    def __init__(self, *, classifier, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def describe(self) -> dict:
        return {"classifier": type(self.classifier).__name__, **super().describe()}

    def _scope(self, state) -> dict:
        decision = self.classifier(case_from_dict(state["case"]))
        return {"extra": {**state.get("extra", {}), "scope": decision.to_dict()}}

    @staticmethod
    def _scope_route(state) -> str:
        return "query" if state["extra"]["scope"]["label"] == "IN_SCOPE" else "scope_reject"

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
        """Add a conditional scope gate while leaving the baseline graph untouched."""
        linear = self._base_nodes()
        names = [name for name, _ in linear]
        if until not in names:
            raise ValueError(f"unknown terminal node {until!r}; nodes: {names}")
        linear = linear[: names.index(until) + 1]

        # intake is always first. The classifier then routes to the ordinary pipeline or reject path.
        graph.add_node("intake", self._intake)
        graph.add_node("scope", self._scope)
        graph.add_node("scope_reject", self._scope_reject)
        graph.add_node("scope_done", self._scope_done)
        for name, fn in linear[1:]:  # query ... requested terminal
            graph.add_node(name, fn)

        graph.add_edge(START, "intake")
        graph.add_edge("intake", "scope")
        graph.add_conditional_edges("scope", self._scope_route, ["query", "scope_reject"])
        for (a, _), (b, _) in zip(linear[1:], linear[2:]):
            graph.add_edge(a, b)
        graph.add_edge(linear[-1][0], "scope_done")
        graph.add_edge("scope_reject", "scope_done")
        return "scope_done"


def _from_advanced(retriever: Retriever, clients: Clients, settings: Settings, classifier, name: str):
    # Recreate advanced's stage composition here instead of importing build_advanced.
    # This keeps scope_gate independent and lets advanced itself opt into the same gate
    # without a circular import.
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return ScopeGatedPipeline(
        name=name, clients=clients, settings=settings, classifier=classifier, extra_nodes=(), **stages
    )


def build_scope_gate(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Advanced retrieval behind an LLM scope classifier; baseline remains unchanged."""
    return _from_advanced(
        retriever, clients, settings, LLMScopeClassifier(clients, _SCOPE_PROMPT), "scope_gate")


def build_scope_gate_rules(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Zero-cost rule baseline for the same pre-retrieval routing task."""
    return _from_advanced(retriever, clients, settings, RuleScopeClassifier(), "scope_gate_rules")


__all__ = ["ScopeGatedPipeline", "build_scope_gate", "build_scope_gate_rules"]
