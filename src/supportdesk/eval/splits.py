"""Splitting evaluation cases into dev and test. All cases of one family (duplicate reports of one
problem) go to the same split."""
from __future__ import annotations

import hashlib


def assign_splits(family_ids: list[str], *, test_fraction: float, seed: int) -> dict[str, str]:
    """Map each family to "dev" or "test". The same seed always gives the same split: families are
    shuffled by a hash of the seed and their name, and the first ``test_fraction`` of them go to test."""
    unique = sorted(set(family_ids))
    ranked = sorted(unique, key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest())
    n_test = round(len(ranked) * test_fraction)
    test = set(ranked[:n_test])
    return {f: ("test" if f in test else "dev") for f in unique}


def check_no_family_overlap(cases) -> list[str]:
    """Problems: any family that appears in more than one split."""
    seen: dict[str, set[str]] = {}
    for case in cases:
        seen.setdefault(case.family_id, set()).add(case.split)
    return [f"family {f!r} spans {sorted(s)}" for f, s in sorted(seen.items()) if len(s) > 1]
