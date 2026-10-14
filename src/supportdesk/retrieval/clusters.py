"""Cluster routing: search only the topic clusters a query is about (an experiment).

Offline, scripts/build_clusters.py groups the index's chunks into topic clusters of about
retrieval.cluster_size chunks and labels each cluster with a name and keywords. retrieval.cluster_method
chooses how: kmeans on the embeddings as they are; centered, with each source type's average embedding
removed first, since issues, comments, release notes and documentation are each written in their own style
and plain k-means groups by that style rather than by topic; or doc_anchored, which clusters only the
documentation (centered) and attaches every other chunk to its nearest documentation cluster, so every
cluster is a topic that holds documentation. At search time, ClusterRoutedRetriever first routes each query to its best clusters
(retrieval.cluster_top), by the keywords (cluster_route = keywords), by meaning against each cluster's description (embedding) or against the
average of its own chunks (centroid), then
searches with places kept for documentation, either inside those clusters only (cluster_mode = filter) or
over everything with the routed clusters ranked higher (boost).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ..tokenizer import lexical_tokens
from .index import COLLECTION, IndexUnusable, VectorIndex
from .retriever import RANK_DECIMALS, Hit


def index_vectors(index: VectorIndex) -> np.ndarray:
    """Every chunk's vector, one row per chunk in ``index.chunks`` order, read back from the vector store
    (kept here, not in VectorIndex, so the code that defines the baseline stays unchanged)."""
    matrix = np.zeros((len(index.chunks), index.dimension), dtype=np.float64)
    offset = None
    while True:
        points, offset = index.client.scroll(COLLECTION, limit=1024, offset=offset, with_vectors=True,
                                             with_payload=["pos"])
        for point in points:
            matrix[int(point.payload["pos"])] = point.vector
        if offset is None:
            return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def kmeans(vectors: np.ndarray, k: int, *, seed: int = 0, iterations: int = 50) -> np.ndarray:
    """Cluster rows by cosine similarity (spherical k-means, k-means++ start); the same input and seed
    always give the same clusters. Returns each row's cluster number, 0..k-1, numbered by first row."""
    x = _normalize(np.asarray(vectors, dtype=np.float64))
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):  # k-means++: each next centre far from those chosen
        distance = 1.0 - np.max(x @ np.array(centres).T, axis=1)
        distance = np.clip(distance, 0.0, None)
        total = distance.sum()
        centres.append(x[rng.choice(n, p=distance / total)] if total > 0 else x[rng.integers(n)])
    centres = np.array(centres)
    labels = np.full(n, -1)
    for _ in range(iterations):
        new = np.argmax(x @ centres.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
            else:  # an empty cluster restarts at the row farthest from its centre
                far = int(np.argmin(np.max(x @ centres.T, axis=1)))
                centres[c] = x[far]
        centres = _normalize(centres)
    return _renumber(labels)  # numbered by first row, so numbering does not depend on the random start


def remove_type_means(vectors: np.ndarray, types: list[str]) -> np.ndarray:
    """Unit vectors with each source type's average removed: what all issues (or all comments, all
    release notes...) share is their style, so what remains is mostly the subject."""
    x = _normalize(np.asarray(vectors, dtype=np.float64))
    out = x.copy()
    for kind in set(types):
        rows = [i for i, t in enumerate(types) if t == kind]
        out[rows] = x[rows] - x[rows].mean(axis=0)
    return _normalize(out)


def _renumber(labels) -> np.ndarray:
    order = {}
    for label in labels:
        order.setdefault(int(label), len(order))
    return np.array([order[int(label)] for label in labels])


def cluster_chunks(vectors: np.ndarray, types: list[str], k: int, *, method: str = "kmeans",
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(each row's cluster, the vectors the clusters were found in), by retrieval.cluster_method:
    kmeans (the embeddings as they are), centered (each source type's average removed first) or
    doc_anchored (centered; the documentation is clustered alone, every other row joins its nearest
    documentation cluster, so every cluster holds documentation)."""
    if method == "kmeans":
        return kmeans(vectors, k, seed=seed), _normalize(np.asarray(vectors, dtype=np.float64))
    space = remove_type_means(vectors, types)
    docs = [i for i, t in enumerate(types) if t == "documentation"]
    if method == "centered" or not docs:
        return kmeans(space, k, seed=seed), space
    doc_labels = kmeans(space[docs], min(k, len(docs)), seed=seed)
    centres = _normalize(np.array([space[docs][doc_labels == c].mean(axis=0)
                                   for c in range(int(doc_labels.max()) + 1)]))
    labels = np.argmax(space @ centres.T, axis=1)
    labels[docs] = doc_labels  # documentation keeps the cluster it was found in
    return _renumber(labels), space


def distinctive_words(texts_in: list[str], texts_out: list[str], count: int = 15) -> list[str]:
    """The words most typical of ``texts_in`` against ``texts_out`` (frequency inside, weighted by how
    rare the word is outside): a cluster's fallback keywords when the model's label is unusable."""
    inside = Counter(t for text in texts_in for t in set(lexical_tokens(text)) if len(t) > 2)
    outside = Counter(t for text in texts_out for t in set(lexical_tokens(text)))
    scored = {t: n / len(texts_in) * math.log((1 + len(texts_out)) / (1 + outside[t])) for t, n in inside.items()}
    return [t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:count]]


def load_clusters(path: str | Path, index: VectorIndex) -> list[dict]:
    """The clusters written by scripts/build_clusters.py, checked against ``index``."""
    path = Path(path)
    if not path.exists():  # like an index that is not built: commands report it
        raise IndexUnusable(f"clusters {path} not found: run scripts/build_clusters.py first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {c.chunk_id for c in index.chunks}
    listed = [cid for row in rows for cid in row["chunk_ids"]]
    if set(listed) != known or len(listed) != len(known):
        raise IndexUnusable(f"clusters {path} were built for another index: rebuild them with "
                            "scripts/build_clusters.py")
    return rows


def cluster_text(row: dict) -> str:
    """What a query is matched against when routing: the cluster's name and keywords."""
    return f"{row['label']}. {', '.join(row['keywords'])}"


class ClusterRoutedRetriever:
    """Route the query to its best topic clusters, then search with places kept for documentation:
    only inside those clusters (filter), or everywhere with their chunks ranked higher (boost)."""

    name = "cluster_doc_slots"

    def __init__(self, index: VectorIndex, embedder, *, doc_slots: int, clusters: str | Path, top: int,
                 route: str, mode: str, boost: float):
        self.index, self.embedder, self.doc_slots = index, embedder, doc_slots
        self.top, self.route, self.mode, self.boost = top, route, mode, boost
        self.clusters = load_clusters(clusters, index)
        position = {c.chunk_id: i for i, c in enumerate(index.chunks)}
        self.members = [{position[cid] for cid in row["chunk_ids"]} for row in self.clusters]
        self.last_route: list[int] = []  # the clusters the last query was routed to
        self._vectors: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        words = [set(lexical_tokens(cluster_text(row))) for row in self.clusters]
        df = Counter(t for w in words for t in w)
        self._words = words
        self._idf = {t: math.log(1 + len(words) / n) for t, n in df.items()}

    def _cluster_vectors(self) -> np.ndarray:
        if self._vectors is None:  # one embedding call for all the cluster descriptions, then cached
            vectors, _ = self.embedder.embed([cluster_text(row) for row in self.clusters], stage="cluster")
            self._vectors = _normalize(np.asarray(vectors, dtype=np.float64))
        return self._vectors

    def _cluster_centroids(self) -> np.ndarray:
        if self._centroids is None:  # the average of each cluster's own chunk vectors (no model call)
            unit = _normalize(index_vectors(self.index))
            self._centroids = _normalize(np.array([unit[sorted(m)].mean(axis=0) for m in self.members]))
        return self._centroids

    def routes(self, query: str, vector) -> list[int]:
        """The ``top`` clusters for this query, best first."""
        if self.route == "keywords":  # the query's words found among each cluster's keywords, rare ones weighing more
            tokens = set(lexical_tokens(query))
            scores = [sum(self._idf[t] for t in tokens & w) for w in self._words]
        else:  # embedding: against each cluster's description; centroid: against its own chunks' average
            q = np.asarray(vector, dtype=np.float64)
            matrix = self._cluster_centroids() if self.route == "centroid" else self._cluster_vectors()
            scores = list(matrix @ (q / (np.linalg.norm(q) or 1.0)))
        order = sorted(range(len(scores)), key=lambda c: (-round(float(scores[c]), RANK_DECIMALS), c))
        return order[: self.top]

    def search(self, query, *, k, as_of=None):
        vec, _ = self.embedder.embed([query], stage="query")
        self.last_route = self.routes(query, vec[0])
        routed = set().union(*(self.members[c] for c in self.last_route)) if self.last_route else set()
        everything = len(self.index.chunks)  # small enough to score every chunk, then filter or boost
        docs = dict(self.index.search(vec[0], limit=everything, as_of=as_of, source_types=("documentation",)))
        every = dict(self.index.search(vec[0], limit=everything, as_of=as_of))
        if self.mode == "filter":  # only the routed clusters can be found
            docs = {i: s for i, s in docs.items() if i in routed}
            every = {i: s for i, s in every.items() if i in routed}
        else:  # boost: everything can be found, the routed clusters' chunks rank higher
            docs = {i: s + (self.boost if i in routed else 0.0) for i, s in docs.items()}
            every = {i: s + (self.boost if i in routed else 0.0) for i, s in every.items()}
        rank = lambda found: sorted(found, key=lambda i: (-round(found[i], RANK_DECIMALS), i))
        chosen = rank(docs)[: min(self.doc_slots, k)]  # the reserved places for documentation
        chosen += [i for i in rank(every) if i not in chosen][: k - len(chosen)]
        scores = {**every, **docs}
        order = sorted(chosen, key=lambda i: (-round(scores[i], RANK_DECIMALS), i))
        return [Hit(self.index.chunks[i], scores[i], r, "cluster" if i in routed else "dense")
                for r, i in enumerate(order, start=1)]


__all__ = ["ClusterRoutedRetriever", "cluster_chunks", "cluster_text", "distinctive_words", "index_vectors",
           "kmeans", "load_clusters", "remove_type_means"]
