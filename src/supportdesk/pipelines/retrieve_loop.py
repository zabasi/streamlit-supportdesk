"""Corrective retrieval inside the corpus: judge 6 chunks, rewrite the query, try at most retrieval.max_loops times.

The baseline graph is unchanged. This pipeline builds its own graph, with one cycle, and then uses
the same packet, generator, validator and summarizer. There is no web search.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import Retrieval, StagedPipeline, _ignore
from .stages import default_stages
from .state import SupportState, case_from_dict, hit_from_dict, hit_to_dict, packet_from_dict


def next_step(sufficient: str, attempt: int, max_loops: int) -> Literal["generate", "rewrite"]:
    """Stop when the chunks can answer, or when the try budget is spent."""
    if sufficient == "yes" or attempt >= max_loops:
        return "generate"
    return "rewrite"


class RetrieveLoopPipeline(StagedPipeline):
    """Retrieve top_k chunks, ask whether they are enough, and rewrite the query until they are
    or retrieval.max_loops searches have run. The generator sees the last chunks only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_loop: dict = {}
        self._packet_graph = None
        self.judge_prompt = Path(self.settings.prompts.judge).read_text(encoding="utf-8")
        self.rewrite_prompt = Path(self.settings.prompts.loop_rewrite).read_text(encoding="utf-8")

    def describe(self) -> dict:
        limit = self.settings.retrieval.max_loops
        return {**super().describe(), "loop": f"judge, then rewrite, at most {limit} searches"}

    def _query_node(self, state) -> dict:
        out = self._query(state)
        extra = {"question": out["queries"][0], "attempt": 0, "previous_queries": [], "searched": []}
        return {**out, "extra": extra}

    def _loop_retrieve(self, state) -> dict:
        extra = dict(state.get("extra") or {})
        query = state["queries"][-1]
        case = case_from_dict(state["case"])
        as_of = case.created_at if self.settings.retrieval.historical_filter else None
        hits = self.retriever.search(query, k=self.settings.retrieval.top_k, as_of=as_of)
        extra["attempt"] = int(extra.get("attempt") or 0) + 1
        extra["last_query"] = query
        extra["searched"] = [*extra.get("searched", []), query]
        rows = [hit_to_dict(hit) for hit in hits]
        return {"candidates": rows, "hits": rows, "extra": extra}

    def _judge(self, state) -> dict:
        extra = dict(state.get("extra") or {})
        question = extra.get("question") or state["queries"][0]
        passages = _passages(state.get("hits") or [])
        messages = [{"role": "system", "content": self.judge_prompt},
                    {"role": "user", "content": f"Question:\n{question}\n\nPassages:\n{passages}"}]
        text, _ = self.clients.chat.complete(messages, stage="judge", json_mode=False)
        extra["sufficient"] = "yes" if text.strip().lower().startswith("yes") else "no"
        return {"extra": extra}

    def _rewrite(self, state) -> dict:
        extra = dict(state.get("extra") or {})
        previous = list(extra.get("previous_queries") or [])
        current = extra.get("last_query") or state["queries"][-1]
        previous.append(current)
        question = extra.get("question") or current
        passages = _passages(state.get("hits") or [], limit=400)
        messages = [{"role": "system", "content": self.rewrite_prompt},
                    {"role": "user", "content": (
                        f"Question:\n{question}\n\nPrevious queries:\n" + "\n".join(previous)
                        + f"\n\nPassages already retrieved:\n{passages}")}]
        text, _ = self.clients.chat.complete(messages, stage="loop_rewrite", json_mode=False)
        query = " ".join(text.strip().split()) or " ".join(question.split())
        seen = {" ".join(item.split()) for item in previous}
        if query in seen:
            query = f"{query} streamlit session state evidence"
        extra["previous_queries"] = previous
        return {"queries": [query], "extra": extra}

    def _route(self, state) -> Literal["packet", "rewrite"]:
        extra = state.get("extra") or {}
        step = next_step(extra.get("sufficient") or "no", int(extra.get("attempt") or 0),
                         self.settings.retrieval.max_loops)
        return "packet" if step == "generate" else "rewrite"

    def _packet_node(self, state) -> dict:
        extra = state.get("extra") or {}
        sufficient = extra.get("sufficient") or "no"
        if sufficient == "yes":
            reason = "passages were sufficient"
        else:
            reason = f"reached maximum tries ({self.settings.retrieval.max_loops})"
        self._last_loop = {"attempt": int(extra.get("attempt") or 0), "sufficient": sufficient,
                           "stop_reason": reason, "queries": list(extra.get("searched") or [])}
        return self._packet(state)

    def _wire(self, graph: StateGraph, until: str) -> str:
        graph.add_node("intake", self._intake)
        graph.add_node("query", self._query_node)
        graph.add_node("retrieve", self._loop_retrieve)
        graph.add_node("judge", self._judge)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("packet", self._packet_node)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "query")
        graph.add_edge("query", "retrieve")
        graph.add_edge("retrieve", "judge")
        graph.add_conditional_edges("judge", self._route, {"packet": "packet", "rewrite": "rewrite"})
        graph.add_edge("rewrite", "retrieve")
        if until == "packet":
            return "packet"
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_node("summarize", self._summarize)
        graph.add_edge("packet", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "summarize")
        return "summarize"

    def _compile(self, until: str):
        graph = StateGraph(SupportState)
        graph.add_edge(self._wire(graph, until), END)
        return graph.compile()

    def add_to(self, graph: StateGraph, *, until: str = "summarize") -> str:
        return self._wire(graph, until)

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._compile("summarize")
        return self._graph

    def retrieve(self, case) -> Retrieval:
        if self._packet_graph is None:
            self._packet_graph = self._compile("packet")
        state: dict = {}
        self._stream(self._packet_graph, case, state, _ignore)
        return Retrieval(state["env"], state["queries"], [hit_from_dict(h) for h in state["hits"]],
                         packet_from_dict(state["packet"]))

    def run(self, case, observe=_ignore):
        self._last_loop = {}
        result = super().run(case, observe)
        if self._last_loop:
            result.trace["loop"] = self._last_loop
            result.trace["n_queries"] = self._last_loop["attempt"]
        return result


def _passages(hits: list[dict], limit: int = 1200) -> str:
    blocks = []
    for row in hits:
        chunk = row["chunk"]
        cite = chunk.get("chunk_id") or "chunk"
        title = chunk.get("title") or ""
        blocks.append(f"[{cite}] {title}\n{(chunk.get('text') or '')[:limit]}")
    return "\n\n".join(blocks) if blocks else "(no passages)"


def build_retrieve_loop(retriever: Retriever, clients: Clients, settings: Settings) -> RetrieveLoopPipeline:
    """Retrieve 6 chunks, judge them, and rewrite the query once (max_loops = 2 searches) before answering."""
    return RetrieveLoopPipeline(name="retrieve_loop", clients=clients, settings=settings,
                                **default_stages(retriever, clients, settings))
