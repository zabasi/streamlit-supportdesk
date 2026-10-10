"""Reciprocal Rank Fusion: merge rankings without having to calibrate their raw scores."""
from __future__ import annotations


def rrf(rankings: list[list[str]], *, k: int) -> list[tuple[str, float]]:
    """Each ranking is an ordered list of chunk ids. Returns (chunk_id, fused_score), best first.
    Ties break on chunk_id, so the order is the same on every run."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
