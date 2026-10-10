"""The indexing graph: load -> screen -> chunk -> audit -> embed -> store.

Every step is a node, so a different chunker (an LLM chunker, say) or an extra step is one node to
swap or add. Text is never modified: every chunk is an exact span of the delivered source
(``source.text[start_char:end_char]``), checked in the chunk node. The screen node refuses a corpus
that contains e-mail addresses or secrets instead of editing it, and the audit node runs the
leakage check; both run BEFORE embedding, so a refused corpus costs nothing. This graph never pauses, so its
state may hold objects (only the support graph is checkpointed)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .data.chunking import (
    Exclusions,
    build_chunks,
    check_chunks,
    make_chunker,
    sources_without_chunks,
)
from .data.clean import sensitive_findings
from .data.corpus import load_corpus
from .eval.leakage import check_leakage
from .llm.factory import Clients
from .retrieval.index import IndexUnusable, index_from_vectors

NODES = ("load", "screen", "chunk", "audit", "embed", "store")


class IndexingState(TypedDict, total=False):
    corpus: Any
    records: list
    sensitive: list  # [(source_id, ["email" | "secret", ...])]: indexing stops if non-empty
    chunks: list
    sources_without_chunks: list[str]
    leakage: Any
    vectors: list
    index: Any


def build_indexing_graph(settings: Settings, clients: Clients, *, cases: list, exclusions: Exclusions,
                         save_to: Path | None = None):
    chunker = make_chunker(settings)

    def load(state):
        corpus = load_corpus(settings.paths.data_dir)
        return {"corpus": corpus, "records": corpus.records}

    def screen(state):
        found = [(r.source_id, kinds) for r in state["records"] if (kinds := sensitive_findings(r.text))]
        return {"sensitive": found}

    def chunk(state):
        chunks = build_chunks(state["records"], exclusions, chunker=chunker)
        check_chunks(state["records"], chunks)
        return {"chunks": chunks,
                "sources_without_chunks": sources_without_chunks(state["records"], chunks, exclusions)}

    def audit(state):
        return {"leakage": check_leakage(state["chunks"], cases,
                                         overlap_threshold=settings.eval.leakage_overlap_threshold,
                                         shingle_tokens=settings.eval.leakage_shingle_tokens)}

    def embed(state):
        if not state["chunks"]:
            raise IndexUnusable("nothing to index")
        vectors, _ = clients.embedding.embed([c.text for c in state["chunks"]], stage="index")
        return {"vectors": vectors}

    def store(state):
        index = index_from_vectors(
            state["chunks"], state["vectors"], embedding_model=clients.embedding_model,
            chunking=settings.chunking, corpus=state["corpus"], exclusions=exclusions, mode=clients.mode,
            notes={"sources_without_chunks": state["sources_without_chunks"],
                   "config_sha256": settings.fingerprint(), "chunker": type(chunker).__name__})
        if save_to is not None:
            index.save(save_to)
        return {"index": index}

    graph = StateGraph(IndexingState)
    for name, fn in zip(NODES, (load, screen, chunk, audit, embed, store), strict=True):
        graph.add_node(name, fn)
    graph.add_edge(START, "load")
    graph.add_edge("load", "screen")
    graph.add_conditional_edges("screen", lambda s: END if s["sensitive"] else "chunk", ["chunk", END])
    graph.add_edge("chunk", "audit")
    graph.add_conditional_edges("audit", lambda s: "embed" if s["leakage"].ok else END, ["embed", END])
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)
    return graph.compile()


def run_indexing(settings: Settings, clients: Clients, *, cases: list, exclusions: Exclusions,
                 save_to: Path | None = None) -> dict:
    """Run the indexing graph; returns its final state (``index`` is absent if the audit failed)."""
    graph = build_indexing_graph(settings, clients, cases=cases, exclusions=exclusions, save_to=save_to)
    return graph.invoke({})
