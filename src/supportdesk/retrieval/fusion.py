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


def weighted_rrf(rankings: list[list[str]], weights: list[float], *, k: int) -> list[tuple[str, float]]:
    """rrf with a weight per ranking: each ranking adds weight / (k + rank). With every weight 1 it gives
    exactly rrf's scores and order."""
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
