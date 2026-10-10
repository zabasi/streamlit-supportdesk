"""Retrievers share one interface so pipelines can swap them without changing anything else."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..llm.protocols import EmbeddingClient
from ..types import Chunk
from .bm25 import BM25
from .filters import visible_mask
from .fusion import rrf
from .index import VectorIndex


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int
    source: str  # dense | bm25 | hybrid


class Retriever(Protocol):
    def search(self, query: str, *, k: int, as_of: str | None = None) -> list[Hit]: ...


#: Scores are compared at this precision when ranking. Different CPUs and math libraries return
#: slightly different floats for the same inputs (0.0 here, 1e-10 there); without rounding, that
#: noise decides near-ties and the same query ranks differently on different machines.
RANK_DECIMALS = 6


def _top(scores, mask, k):
    """Best first; ties (at RANK_DECIMALS precision) go to the earlier chunk, on every machine."""
    order = sorted((i for i, ok in enumerate(mask) if ok),
                   key=lambda i: (-round(float(scores[i]), RANK_DECIMALS), i))
    return order[:k]


class DenseRetriever:
    """Embedding similarity between the query and each chunk."""

    name = "dense"

    def __init__(self, index: VectorIndex, embedder: EmbeddingClient):
        self.index, self.embedder = index, embedder

    def search(self, query, *, k, as_of=None):
        vec, _ = self.embedder.embed([query], stage="query")
        # Ask the vector store for extra candidates, then rank them with the same fixed rule
        # as everywhere else: the store's own order among (near-)ties is arbitrary.
        limit = min(len(self.index.chunks), max(k * 4, k + 20))
        found = dict(self.index.search(vec[0], limit=limit, as_of=as_of))
        order = sorted(found, key=lambda i: (-round(found[i], RANK_DECIMALS), i))[:k]
        return [Hit(self.index.chunks[i], found[i], r, "dense") for r, i in enumerate(order, start=1)]


class DocSlotsRetriever:
    """Embedding similarity, with places kept for documentation so similar issues cannot crowd it out."""

    name = "dense_doc_slots"

    def __init__(self, index: VectorIndex, embedder: EmbeddingClient, *, doc_slots: int):
        self.index, self.embedder, self.doc_slots = index, embedder, doc_slots

    def search(self, query, *, k, as_of=None):
        vec, _ = self.embedder.embed([query], stage="query")
        limit = min(len(self.index.chunks), max(k * 4, k + 20))
        docs = dict(self.index.search(vec[0], limit=limit, as_of=as_of, source_types=("documentation",)))
        every = dict(self.index.search(vec[0], limit=limit, as_of=as_of))
        rank = lambda found: sorted(found, key=lambda i: (-round(found[i], RANK_DECIMALS), i))
        chosen = rank(docs)[: min(self.doc_slots, k)]  # the reserved places
        chosen += [i for i in rank(every) if i not in chosen][: k - len(chosen)]  # the rest, from everything
        scores = {**every, **docs}
        order = sorted(chosen, key=lambda i: (-round(scores[i], RANK_DECIMALS), i))  # best first, as usual
        return [Hit(self.index.chunks[i], scores[i], r, "dense") for r, i in enumerate(order, start=1)]


class BM25Retriever:
    """Keyword match (BM25); finds exact identifiers and error strings."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], *, k1: float, b: float):
        self.chunks, self.bm25 = chunks, BM25(chunks, k1=k1, b=b)

    def search(self, query, *, k, as_of=None):
        scores = self.bm25.scores(query)
        mask = visible_mask(self.chunks, as_of)
        picked = [i for i in _top(scores, mask, k) if scores[i] > 0]
        return [Hit(self.chunks[i], float(scores[i]), r, "bm25") for r, i in enumerate(picked, 1)]


class HybridRetriever:
    """Dense and BM25 results merged with Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(self, index: VectorIndex, embedder: EmbeddingClient, *, pool: int, rrf_k: int,
                 bm25_k1: float, bm25_b: float):
        self.dense = DenseRetriever(index, embedder)
        self.lexical = BM25Retriever(index.chunks, k1=bm25_k1, b=bm25_b)
        self.pool, self.rrf_k = pool, rrf_k
        self.by_id = {c.chunk_id: c for c in index.chunks}

    def search(self, query, *, k, as_of=None):
        d = self.dense.search(query, k=self.pool, as_of=as_of)
        b = self.lexical.search(query, k=self.pool, as_of=as_of)
        fused = rrf([[h.chunk.chunk_id for h in d], [h.chunk.chunk_id for h in b]], k=self.rrf_k)[:k]
        return [Hit(self.by_id[cid], score, r, "hybrid") for r, (cid, score) in enumerate(fused, 1)]


def fuse_hits(rankings: list[list[Hit]], k: int, *, rrf_k: int) -> list[Hit]:
    """Merge several per-query rankings into one with RRF (used when a query stage returns more
    than one query). The first occurrence of a chunk supplies the chunk object."""
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for hit in ranking:
            chunks.setdefault(hit.chunk.chunk_id, hit.chunk)
    fused = rrf([[h.chunk.chunk_id for h in ranking] for ranking in rankings], k=rrf_k)[:k]
    return [Hit(chunks[cid], score, rank, "fused") for rank, (cid, score) in enumerate(fused, 1)]
