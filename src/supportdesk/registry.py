"""Where pipelines and retrievers are registered. The CLI and the web UI both build through here,
so a new pipeline or retriever becomes available everywhere by adding one line."""
from __future__ import annotations

from .config import ConfigError, Settings
from .llm.factory import Clients
from .pipelines.advanced import build_advanced
from .pipelines.baseline import build_baseline
from .retrieval.index import VectorIndex
from .retrieval.retriever import BM25Retriever, DenseRetriever, HybridRetriever

#: Pipelines by name: each builds the pipeline from (retriever, clients, settings); its docstring is its one-line description.
PIPELINES = {"baseline": build_baseline, "advanced": build_advanced}

RETRIEVERS = {
    "dense": lambda index, clients, s: DenseRetriever(index, clients.embedding),
    "bm25": lambda index, clients, s: BM25Retriever(
        index.chunks, k1=s.retrieval.bm25_k1, b=s.retrieval.bm25_b),
    "hybrid": lambda index, clients, s: HybridRetriever(
        index, clients.embedding, pool=s.retrieval.hybrid_pool, rrf_k=s.retrieval.rrf_k,
        bm25_k1=s.retrieval.bm25_k1, bm25_b=s.retrieval.bm25_b),
}


def pipeline_description(name: str) -> str:
    return (PIPELINES[name].__doc__ or "").strip().splitlines()[0]


def _require(kind: str, name: str, options) -> None:
    if name not in options:
        hint = " (run `supportdesk pipelines` to see them)" if kind == "pipeline" else ""
        raise ConfigError(f"unknown {kind} {name!r}; choose from {sorted(options)}{hint}")


def require(pipeline: str, retriever: str) -> None:
    """Fail fast, before anything is loaded, when a name is not registered."""
    _require("pipeline", pipeline, PIPELINES)
    _require("retriever", retriever, RETRIEVERS)


def make_retriever(name: str, index: VectorIndex, clients: Clients, settings: Settings):
    _require("retriever", name, RETRIEVERS)
    return RETRIEVERS[name](index, clients, settings)


def make_pipeline(name: str, retriever: str, index: VectorIndex, clients: Clients,
                  settings: Settings):
    require(name, retriever)
    return PIPELINES[name](make_retriever(retriever, index, clients, settings), clients, settings)
