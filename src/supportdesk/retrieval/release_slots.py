"""Release notes newer than the reporter's version, added to the candidates (retrieval.release_slots).

The documentation in the corpus is one current snapshot, so it cannot be filtered to the reporter's version.
Release notes can: each states the version it belongs to. With release_slots > 0, after the normal search,
up to that many release notes for versions NEWER than the one the report states, best matching the query,
are added to the candidates (none are displaced). They describe changes the reporter does not have yet;
the reranker decides whether one is useful, and with rerank_versions on it sees each labelled as such.
"""
from __future__ import annotations

import dataclasses

from ..pipelines.stages.version_notes import versions
from .retriever import Hit


def _parts(retriever):
    """The index and embedder of a retriever, or of the base retriever it wraps; (None, None) if neither."""
    for r in (retriever, getattr(retriever, "base", None)):
        if r is not None and hasattr(r, "index") and hasattr(r, "embedder"):
            return r.index, r.embedder
    return None, None


def newer_release_notes(retriever, query: str, reported_version: str | None, n: int, *,
                        as_of: str | None = None) -> list[Hit]:
    """Up to ``n`` release-note chunks for versions newer than the reported one, best matching ``query``."""
    reported = versions(reported_version)
    index, embedder = _parts(retriever)
    if n <= 0 or not reported or index is None:
        return []
    newest_reported = max(reported)
    vector, _ = embedder.embed([query], stage="query")  # the same text as the main search: a cache hit
    found = index.search(vector[0], limit=min(len(index.chunks), 300), as_of=as_of,
                         source_types=("release_section",))
    out = []
    for position, score in found:
        chunk = index.chunks[position]
        own = versions(chunk.version)
        if own and own[0] > newest_reported:
            out.append(Hit(chunk, score, 0, "release_slot"))
            if len(out) == n:
                break
    return out


class ReleaseSlots:
    """Adds release notes newer than the reported version to the candidates, then reranks them all."""

    def __init__(self, reranker, retriever, settings):
        self.reranker, self.retriever = reranker, retriever
        self.n, self.historical = settings.retrieval.release_slots, settings.retrieval.historical_filter

    def __call__(self, case, query: str, hits: list[Hit]) -> list[Hit]:
        have = {h.chunk.chunk_id for h in hits}
        as_of = case.created_at if self.historical else None
        extra = [h for h in newer_release_notes(self.retriever, query, (case.environment or {}).get("streamlit_version"),
                                                self.n, as_of=as_of) if h.chunk.chunk_id not in have]
        hits = list(hits) + [dataclasses.replace(h, rank=len(hits) + n) for n, h in enumerate(extra, start=1)]
        return self.reranker(case, query, hits)


__all__ = ["ReleaseSlots", "newer_release_notes"]
