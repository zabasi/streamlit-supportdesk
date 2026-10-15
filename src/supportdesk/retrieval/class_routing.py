"""Topic-class routing: search only the classes a query is about.

Two ways to label chunks: spherical k-means (see ``retrieval.clusters``) or an LLM with a
fixed Streamlit topic list. At search time the query is routed to the best ``top`` classes
(by centroid, or by one LLM call) and ``ClassRoutedRetriever`` looks only there, keeping
documentation places the same way ``dense_doc_slots`` does.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..llm.errors import ProviderError
from ..pipelines.parsing import NotJson, extract_json
from .clusters import _normalize, kmeans, remove_type_means
from .index import VectorIndex
from .retriever import RANK_DECIMALS, Hit

TOPICS = (
    "session_state",
    "widgets_keys",
    "callbacks_forms",
    "data_editor",
    "caching",
    "multipage_navigation",
    "expanders_layout",
    "other",
)

_KEYWORDS = (
    ("data_editor", ("data_editor", "dataframe", "data editor", "column_config")),
    ("caching", ("cache_data", "cache_resource", "@st.cache", "st.cache")),
    ("multipage_navigation", ("multipage", "pages/", "st.navigation", "st.page_link")),
    ("callbacks_forms", ("on_change", "on_click", "st.form", "form_submit")),
    ("expanders_layout", ("st.expander", "expander", "st.sidebar")),
    ("session_state", ("session_state", "session state")),
    ("widgets_keys", ("widget", "key=", "st.text_input", "st.slider", "st.number_input")),
)


def heuristic_label(text: str) -> str:
    """Keyword fallback used offline and when the model answer is unusable."""
    lowered = text.lower()
    for label, keys in _KEYWORDS:
        if any(key in lowered for key in keys):
            return label
    return "other"


def _valid_label(value) -> str | None:
    label = str(value or "").strip()
    return label if label in TOPICS else None


def classify_chunks_llm(chunks, chat, *, prompt: str, batch: int = 16) -> list[str]:
    """One label per chunk, in chunk order. Unusable batches fall back to keywords."""
    labels = [heuristic_label(c.text) for c in chunks]
    for start in range(0, len(chunks), batch):
        group = chunks[start:start + batch]
        listing = "\n".join(f"{i}. [{c.source_type}] {c.title}: {' '.join(c.text.split())[:360]}"
                            for i, c in enumerate(group))
        try:
            raw, _ = chat.complete(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": listing}],
                stage="topic_chunk", json_mode=True)
            payload = extract_json(raw)
            rows = payload.get("assignments") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = _valid_label(row.get("label"))
                if label is None or not isinstance(row.get("i"), int):
                    continue
                if 0 <= row["i"] < len(group):
                    labels[start + row["i"]] = label
        except (NotJson, ValueError, TypeError, OSError, ProviderError):
            continue
    return labels


def classify_query_llm(query: str, chat, *, prompt: str, top: int = 2) -> list[str]:
    """Best topic labels for a report, most relevant first."""
    try:
        raw, _ = chat.complete(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": query[:4000]}],
            stage="topic_query", json_mode=True)
        payload = extract_json(raw)
        found = []
        for value in (payload.get("labels") if isinstance(payload, dict) else None) or []:
            label = _valid_label(value)
            if label and label not in found:
                found.append(label)
        if found:
            return found[:top]
    except (NotJson, ValueError, TypeError, OSError, ProviderError):
        pass
    first = heuristic_label(query)
    return [first] if first != "other" else [first, "session_state"][:top]


def kmeans_labels(vectors: np.ndarray, types: list[str], *, k: int = 8,
                  method: str = "doc_anchored", seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Cluster ids as strings, plus the space the clusters were found in."""
    from .clusters import cluster_chunks
    assigned, space = cluster_chunks(vectors, types, k, method=method, seed=seed)
    return [str(int(x)) for x in assigned], space


def class_centroids(vectors: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    unit = _normalize(np.asarray(vectors, dtype=np.float64))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    return {label: _normalize(unit[idx].mean(axis=0, keepdims=True))[0]
            for label, idx in groups.items() if idx}


def centroid_routes(query_vector, centroids: dict[str, np.ndarray], *, top: int = 2) -> list[str]:
    q = np.asarray(query_vector, dtype=np.float64)
    q = q / (np.linalg.norm(q) or 1.0)
    scored = [(label, float(vec @ q)) for label, vec in centroids.items()]
    scored.sort(key=lambda kv: (-round(kv[1], RANK_DECIMALS), kv[0]))
    return [label for label, _ in scored[:top]]


class ClassRoutedRetriever:
    """Route the query to its best topic classes, then search with documentation places kept."""

    name = "class_doc_slots"

    def __init__(self, index: VectorIndex, embedder, *, labels: list[str], doc_slots: int,
                 top: int = 2, mode: str = "filter", boost: float = 0.05, chat=None,
                 prompt: str = "", route: str = "centroid", vectors: np.ndarray | None = None):
        if len(labels) != len(index.chunks):
            raise ValueError(f"{len(labels)} labels for {len(index.chunks)} chunks")
        self.index, self.embedder, self.labels = index, embedder, labels
        self.doc_slots, self.top, self.mode, self.boost = doc_slots, top, mode, boost
        self.chat, self.prompt, self.route = chat, prompt, route
        self.members = defaultdict(set)
        for i, label in enumerate(labels):
            self.members[label].add(i)
        if vectors is None:
            from .clusters import index_vectors
            vectors = index_vectors(index)
        self.centroids = class_centroids(vectors, labels)
        self.last_route: list[str] = []

    def routes(self, query: str, vector) -> list[str]:
        if self.route == "llm" and self.chat is not None:
            return classify_query_llm(query, self.chat, prompt=self.prompt, top=self.top)
        return centroid_routes(vector, self.centroids, top=self.top)

    def search(self, query, *, k, as_of=None):
        vec, _ = self.embedder.embed([query], stage="query")
        self.last_route = self.routes(query, vec[0])
        routed = set().union(*(self.members[c] for c in self.last_route if c in self.members))
        everything = len(self.index.chunks)
        docs = dict(self.index.search(vec[0], limit=everything, as_of=as_of,
                                      source_types=("documentation",)))
        every = dict(self.index.search(vec[0], limit=everything, as_of=as_of))
        if self.mode == "filter":
            docs = {i: s for i, s in docs.items() if i in routed}
            every = {i: s for i, s in every.items() if i in routed}
        else:
            docs = {i: s + (self.boost if i in routed else 0.0) for i, s in docs.items()}
            every = {i: s + (self.boost if i in routed else 0.0) for i, s in every.items()}
        rank = lambda found: sorted(found, key=lambda i: (-round(found[i], RANK_DECIMALS), i))
        chosen = rank(docs)[: min(self.doc_slots, k)]
        chosen += [i for i in rank(every) if i not in chosen][: k - len(chosen)]
        scores = {**every, **docs}
        order = sorted(chosen, key=lambda i: (-round(scores[i], RANK_DECIMALS), i))
        return [Hit(self.index.chunks[i], scores[i], r, "class" if i in routed else "dense")
                for r, i in enumerate(order, start=1)]


__all__ = ["TOPICS", "ClassRoutedRetriever", "centroid_routes", "class_centroids",
           "classify_chunks_llm", "classify_query_llm", "heuristic_label", "kmeans_labels",
           "remove_type_means", "kmeans"]
