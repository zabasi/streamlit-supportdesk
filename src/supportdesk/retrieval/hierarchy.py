"""Hierarchical retrieval: search the section summaries first, then the chunks inside the best sections.

Level 1 compares the query with a short summary of every documentation section
(``scripts/summarize_sections.py``). Level 2 ranks the chunks that lie inside the best-matching
sections by their own similarity to the query. A few places in the result (``places``) go to those
chunks; the rest come from the ordinary search (``base``), so issues, discussions and release notes
are still found. The same summaries file serves ``SummaryReranker``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..llm.protocols import EmbeddingClient
from ..types import Chunk
from .filters import parse_ts
from .index import COLLECTION, IndexUnusable, VectorIndex
from .retriever import RANK_DECIMALS, Hit, Retriever

SOURCE = "hierarchy"  # what a chunk found inside a best-matching section is marked with (found_by)


class SectionSummaries:
    """The summary of every documentation section, embedded once on first use."""

    def __init__(self, path: str | Path, embedder: EmbeddingClient, known_pages: set[str]):
        self.path, self.embedder = Path(path), embedder
        if not self.path.exists():  # reported like an index that is not built
            raise IndexUnusable(f"section summaries {self.path} not found: run scripts/summarize_sections.py first")
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.sections = [r for r in rows if r["source_id"] in known_pages]
        self._vectors: np.ndarray | None = None

    def _matrix(self) -> np.ndarray:
        if self._vectors is None:
            texts = [f"{r['title']} > {r['heading']}: {r['summary']}" for r in self.sections]
            vectors = self.embedder.embed(texts, stage="summary")[0] if texts else []
            matrix = np.asarray(vectors, dtype=np.float64).reshape(len(texts), -1)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._vectors = matrix / np.where(norms == 0, 1.0, norms)
        return self._vectors

    def best(self, query_vector, n: int) -> list[dict]:
        """The ``n`` sections whose summaries are most similar to the query, best first."""
        if not self.sections or n < 1:
            return []
        q = np.asarray(query_vector, dtype=np.float64)
        norm = np.linalg.norm(q)
        scores = self._matrix() @ (q / (norm if norm else 1.0))
        order = sorted(range(len(self.sections)), key=lambda i: (-round(float(scores[i]), RANK_DECIMALS), i))
        return [{**self.sections[i], "score": float(scores[i])} for i in order[:n]]


class HierarchicalRetriever:
    """``base`` search, with ``places`` of the results taken from inside the ``sections`` best sections."""

    name = "hierarchical"

    def __init__(self, base: Retriever, index: VectorIndex, embedder: EmbeddingClient,
                 summaries: str | Path, *, sections: int, places: int):
        self.base, self.index, self.embedder = base, index, embedder
        self.sections_n, self.places = sections, places
        self.by_page: dict[str, list[int]] = {}
        for position, chunk in enumerate(index.chunks):
            if chunk.source_type == "documentation" and chunk.start_char is not None:
                self.by_page.setdefault(chunk.source_id, []).append(position)
        self.level = SectionSummaries(summaries, embedder, set(self.by_page))

    def _inside(self, section: dict, as_of: str | None) -> list[int]:
        """Positions of the chunks that overlap the section and are visible at ``as_of``."""
        cutoff = parse_ts(as_of)
        found = []
        for position in self.by_page.get(section["source_id"], []):
            chunk: Chunk = self.index.chunks[position]
            if chunk.start_char < section["end_char"] and chunk.end_char > section["start_char"]:
                stamp = parse_ts(chunk.as_of)
                if cutoff is None or stamp is None or stamp < cutoff:
                    found.append(position)
        return found

    def _similarity(self, vector, positions: list[int]) -> dict[int, float]:
        """Cosine similarity between the query and the stored vector of each chunk."""
        if not positions:
            return {}
        records = self.index.client.retrieve(COLLECTION, ids=positions, with_vectors=True)
        q = np.asarray(vector, dtype=np.float64)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = {}
        for record in records:
            v = np.asarray(record.vector, dtype=np.float64)
            scores[int(record.id)] = float(v @ q / (np.linalg.norm(v) or 1.0))
        return scores

    def search(self, query, *, k, as_of=None):
        base_hits = self.base.search(query, k=k, as_of=as_of)
        if self.places < 1 or self.sections_n < 1:
            return base_hits
        vector = self.embedder.embed([query], stage="query")[0][0]
        positions: list[int] = []
        for section in self.level.best(vector, self.sections_n):
            positions += [p for p in self._inside(section, as_of) if p not in positions]
        scores = self._similarity(vector, positions)
        order = sorted(scores, key=lambda p: (-round(scores[p], RANK_DECIMALS), p))[: min(self.places, k)]
        inside = [Hit(self.index.chunks[p], scores[p], 0, SOURCE) for p in order]
        taken = {h.chunk.chunk_id for h in inside}
        merged = inside + [h for h in base_hits if h.chunk.chunk_id not in taken][: k - len(inside)]
        merged.sort(key=lambda h: (-round(h.score, RANK_DECIMALS), h.chunk.chunk_id))
        return [Hit(h.chunk, h.score, rank, h.source) for rank, h in enumerate(merged, start=1)]
