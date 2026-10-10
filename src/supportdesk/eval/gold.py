"""Gold-label checks. A label that can never be retrieved silently caps recall: it may be a typo, an
excluded evaluation issue, or dated after the case (the historical filter hides it)."""
from __future__ import annotations

from ..retrieval.filters import parse_ts
from ..types import Chunk
from .cases import EvalCase


def check_gold(cases: list[EvalCase], chunks: list[Chunk], *, historical: bool) -> list[str]:
    """Human-readable problems, one per bad label. Empty means every label is reachable."""
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_id, []).append(chunk)
        if chunk.parent_source_id:  # a comment stands for its parent issue
            by_source.setdefault(chunk.parent_source_id, []).append(chunk)
    problems = []
    for case in cases:
        filed = parse_ts(case.input.created_at) if historical else None
        for gold in case.gold_evidence:
            found = by_source.get(gold)
            if not found:
                problems.append(f"{case.case_id}: gold {gold!r} is not in the index "
                                "(typo, or it is an evaluation issue that was excluded)")
            elif filed is not None and not any(
                    (c.as_of is None) or parse_ts(c.as_of) < filed for c in found):
                problems.append(f"{case.case_id}: gold {gold!r} is dated after the case "
                                f"({case.input.created_at}); the historical filter hides it, so it "
                                "can never be retrieved")
    return problems


def describe_cases(cases: list[EvalCase]) -> dict:
    """Counts a person needs before trusting an evaluation set."""
    by_split: dict = {}
    for case in cases:
        bucket = by_split.setdefault(case.split, {"n": 0, "with_gold": 0, "categories": {}})
        bucket["n"] += 1
        bucket["with_gold"] += bool(case.gold_evidence)
        bucket["categories"][case.category] = bucket["categories"].get(case.category, 0) + 1
    return {"total": len(cases), "families": len({c.family_id for c in cases}), "by_split": by_split}
