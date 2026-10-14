"""The support pipeline as a LangGraph graph. Every stage is a node; the class owns what must not
be copied between pipelines: timing, per-case usage, failure outcomes, redaction and the trace."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from langgraph.graph import END, START, StateGraph

from ..config import Settings, redact
from ..data.environment import extract_environment
from ..llm.errors import BudgetExhausted, CallLimitReached, GatewayError, InputTooLarge
from ..llm.factory import Clients
from ..retrieval.retriever import Hit, Retriever, fuse_hits
from ..types import CaseInput, CaseResult
from .packet import Packet
from .parsing import PayloadError
from .stages.protocols import GenerationFailed
from .state import (
    SupportState,
    answer_from_dict,
    answer_to_dict,
    case_from_dict,
    case_to_dict,
    hit_from_dict,
    hit_to_dict,
    packet_from_dict,
    packet_to_dict,
)

STAGE_NAMES = ("query", "retriever", "reranker", "packet", "generator", "validator", "summarizer")

#: Graph node that runs each stage. "intake" (environment) comes first and is not replaceable.
NODES = {"query": "query", "retriever": "retrieve", "reranker": "rerank", "packet": "packet",
         "generator": "generate", "validator": "validate", "summarizer": "summarize"}

#: ``observe(step, output)`` is told what each step produced; used by ``supportdesk snapshot``.
Observer = Callable[[str, object], None]


def _ignore(step: str, output: object) -> None:
    pass


@dataclass
class Retrieval:
    """The retrieval half of a run: what the pipeline would hand to the model."""

    env: dict
    queries: list[str]
    hits: list[Hit]
    packet: Packet


def _stage_name(stage) -> str:
    return getattr(stage, "__name__", None) or type(stage).__name__


class StagedPipeline:
    def __init__(self, *, name: str, clients: Clients, settings: Settings, query, retriever,
                 reranker, packet, generator, validator, summarizer, extra_nodes=()):
        """``extra_nodes``: ``(name, fn, after)`` entries that add a node to this pipeline's graph
        right after the node named ``after``. ``fn(state) -> dict`` reads the plain-data state and
        returns the keys it updates: existing ones (for example ``hits``) or its own data under ``extra``."""
        self.name, self.clients, self.settings = name, clients, settings
        self.extra_nodes = tuple(extra_nodes)
        self.query, self.retriever, self.reranker, self.packet = query, retriever, reranker, packet
        self.generator, self.validator, self.summarizer = generator, validator, summarizer
        for stage in STAGE_NAMES:
            target = getattr(self, stage)
            ok = hasattr(target, "search") if stage == "retriever" else callable(target)
            if not ok:
                raise TypeError(f"stage {stage!r} is not usable: {target!r}")
        known = [n for n, _ in self._base_nodes()]
        for extra_name, fn, after in self.extra_nodes:
            if extra_name in known:
                raise ValueError(f"extra node {extra_name!r}: that name is already a node")
            if after not in known:
                raise ValueError(f"extra node {extra_name!r}: no node {after!r} to put it after; nodes: {known}")
            if not callable(fn):
                raise TypeError(f"extra node {extra_name!r} is not callable")
            known.append(extra_name)
        self._graph = self._retrieval_graph = None

    def describe(self) -> dict:
        """Which implementation fills each stage; recorded in every run summary and trace."""
        stages = {stage: _stage_name(getattr(self, stage)) for stage in STAGE_NAMES}
        return {**stages, **{f"node:{name}": f"{_stage_name(fn)} (after {after})"
                             for name, fn, after in self.extra_nodes}}

    def variant(self, name: str | None = None, **stages) -> StagedPipeline:
        """A copy with some stages replaced; the original is untouched. For ablations."""
        unknown = set(stages) - set(STAGE_NAMES)
        if unknown:
            raise TypeError(f"unknown stage(s): {sorted(unknown)}; valid: {STAGE_NAMES}")
        kwargs = {stage: getattr(self, stage) for stage in STAGE_NAMES}
        kwargs.update(stages)
        return StagedPipeline(name=name or f"{self.name}+{'+'.join(sorted(stages))}",
                              clients=self.clients, settings=self.settings, extra_nodes=self.extra_nodes,
                              **kwargs)

    # ---- nodes: plain data in, plain data out (see pipelines/state.py) ----

    def _intake(self, state: SupportState) -> dict:
        case = case_from_dict(state["case"])
        return {"env": {**extract_environment(f"{case.title}\n{case.body}"), **case.environment}}

    def _query(self, state: SupportState) -> dict:
        queries = list(self.query(case_from_dict(state["case"]), state["env"]))
        if not queries:
            raise PayloadError("the query stage returned no queries")
        return {"queries": queries}

    def _retrieve_node(self, state: SupportState) -> dict:
        case = case_from_dict(state["case"])
        as_of = case.created_at if self.settings.retrieval.historical_filter else None
        pool = self.settings.pool_size
        rankings = [self.retriever.search(q, k=pool, as_of=as_of) for q in state["queries"]]
        hits = (rankings[0] if len(rankings) == 1
                else fuse_hits(rankings, pool, rrf_k=self.settings.retrieval.rrf_k))
        return {"candidates": [hit_to_dict(h) for h in hits]}

    def _rerank(self, state: SupportState) -> dict:
        hits = self.reranker(case_from_dict(state["case"]), state["queries"][0],
                             [hit_from_dict(h) for h in state["candidates"]])
        return {"hits": [hit_to_dict(h) for h in hits[: self.settings.retrieval.top_k]]}

    def _packet(self, state: SupportState) -> dict:
        return {"packet": packet_to_dict(self.packet([hit_from_dict(h) for h in state["hits"]]))}

    def _generate(self, state: SupportState) -> dict:
        answer = self.generator(case_from_dict(state["case"]), state["env"], packet_from_dict(state["packet"]))
        return {"generated": answer_to_dict(answer)}

    def _validate(self, state: SupportState) -> dict:
        answer = self.validator(case_from_dict(state["case"]), packet_from_dict(state["packet"]),
                                answer_from_dict(state["generated"]))
        return {"answer": answer_to_dict(answer)}

    def _summarize(self, state: SupportState) -> dict:
        summary = self.summarizer(case_from_dict(state["case"]), state["env"],
                                  packet_from_dict(state["packet"]), answer_from_dict(state["answer"]))
        return {"summary": summary}

    def nodes(self) -> list[tuple[str, Callable]]:
        """The support graph's nodes, in order, extra nodes included. The case workflow reuses them."""
        nodes = self._base_nodes()
        for name, fn, after in self.extra_nodes:
            at = [n for n, _ in nodes].index(after) + 1
            nodes.insert(at, (name, fn))
        return nodes

    def _base_nodes(self) -> list[tuple[str, Callable]]:
        return [("intake", self._intake), ("query", self._query), ("retrieve", self._retrieve_node),
                ("rerank", self._rerank), ("packet", self._packet), ("generate", self._generate),
                ("validate", self._validate), ("summarize", self._summarize)]

    def add_to(self, graph: StateGraph, *, until: str = "summarize") -> str:
        """Add the nodes from intake up to ``until`` to ``graph``, chained from START. Returns the
        last node's name, for the caller to continue from."""
        steps = self.nodes()
        names = [name for name, _ in steps]
        steps = steps[: names.index(until) + 1]
        for name, fn in steps:
            graph.add_node(name, fn)
        for a, b in zip([START, *[n for n, _ in steps]], [n for n, _ in steps]):
            graph.add_edge(a, b)
        return steps[-1][0]

    def _build(self, last: str):
        graph = StateGraph(SupportState)
        graph.add_edge(self.add_to(graph, until=last), END)
        return graph.compile()

    @property
    def graph(self):
        """The compiled LangGraph graph (built once, on first use)."""
        if self._graph is None:
            self._graph = self._build("summarize")
        return self._graph

    # ---- running ----

    _OBSERVED: ClassVar[dict] = {"intake": ("environment", "env", None),
                 "query": ("query", "queries", None),
                 "retrieve": ("retriever", "candidates", lambda rows: [hit_from_dict(h) for h in rows]),
                 "rerank": ("reranker", "hits", lambda rows: [hit_from_dict(h) for h in rows]),
                 "packet": ("packet", "packet", packet_from_dict),
                 "generate": ("generator", "generated", answer_from_dict),
                 "validate": ("validator", "answer", answer_from_dict),
                 "summarize": ("summarizer", "summary", None)}

    def _stream(self, graph, case: CaseInput, state: dict, observe: Observer) -> None:
        """Run the graph, merging each node's output into ``state`` as it arrives, so a failure
        part-way still leaves everything produced before it (for the trace and the result)."""
        state["case"] = case_to_dict(case)
        for update in graph.stream({"case": state["case"]}, stream_mode="updates"):
            for node, output in update.items():
                state.update(output or {})
                step, key, convert = self._OBSERVED.get(node, (None, None, None))
                if step is not None:
                    value = output[key]
                    observe(step, convert(value) if convert else value)

    def retrieve(self, case: CaseInput) -> Retrieval:
        """Run everything before the model call: query, retrieval, reranking, packet. This is what
        ``supportdesk eval-retrieval`` measures, at the cost of the query embedding only (plus any
        model call a query or rerank stage makes). Errors propagate; ``run`` maps them for you."""
        if self._retrieval_graph is None:
            self._retrieval_graph = self._build("packet")
        state: dict = {}
        self._stream(self._retrieval_graph, case, state, _ignore)
        return Retrieval(state["env"], state["queries"], [hit_from_dict(h) for h in state["hits"]],
                         packet_from_dict(state["packet"]))

    def run(self, case: CaseInput, observe: Observer = _ignore) -> CaseResult:
        ledger = self.clients.ledger
        mark, started = ledger.mark(), time.perf_counter()
        result = CaseResult(case_id=case.case_id, pipeline=self.name, mode=self.clients.mode,
                            outcome="error")
        attempts: list[dict] = []
        state: dict = {}
        try:
            self._stream(self.graph, case, state, observe)
        except GenerationFailed as exc:
            attempts = exc.attempts
            result.error = str(exc)
        except (BudgetExhausted, CallLimitReached) as exc:
            result.outcome, result.error = "budget_stopped", redact(exc)
        except PayloadError as exc:  # wrong shape is a validation failure: never retried
            result.error = f"invalid_payload: {redact(exc)}"
        except (GatewayError, InputTooLarge) as exc:
            result.error = redact(exc)
        if "packet" in state:
            packet = state["packet"]
            result.retrieved = [{"chunk_id": h["chunk"]["chunk_id"], "source_id": h["chunk"]["source_id"],
                                 "parent_source_id": h["chunk"]["parent_source_id"], "rank": h["rank"],
                                 "score": round(h["score"], 6), "found_by": h["source"]}
                                for h in state["hits"]]
            result.packet_hash = packet["sha256"]
            result.trace["packet"] = [{"label": label, "chunk_id": cid} for label, cid in packet["labels"].items()]
        if "answer" in state:
            answer = state["answer"]
            attempts = answer["attempts"]
            result.outcome, result.user_reply = answer["outcome"], answer["user_reply"]
            result.known_facts, result.open_questions = answer["known_facts"], answer["open_questions"]
            result.next_step, result.citations = answer["next_step"], answer["citations"]
            result.violations = answer["violations"]
            result.confidence = answer.get("confidence")
        if "summary" in state:
            result.maintainer_summary = state["summary"]
        queries = state.get("queries", [])
        result.trace["attempts"] = attempts
        result.trace["query_sha256"] = (hashlib.sha256(queries[0].encode()).hexdigest()[:16]
                                        if queries else None)
        result.trace["n_queries"] = len(queries)
        result.trace["stages"] = self.describe()
        result.usage = {**ledger.summarize(mark),
                        "wall_seconds": round(time.perf_counter() - started, 4)}
        return result


__all__ = ["NODES", "STAGE_NAMES", "Observer", "Retrieval", "Retriever", "StagedPipeline"]
