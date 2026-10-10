"""The refactored baseline must reproduce the pre-refactor outputs exactly.

Scores are compared with a tolerance (float32 matmul can differ in the last digit across CPUs);
everything else is exact. If this fails after an *intended* behaviour change, regenerate with
``PYTHONPATH=src:. python -m tests.golden.generate`` and review the diff.
"""
import json

import pytest

from supportdesk.pipelines.baseline import build_baseline
from tests.golden.scenarios import GOLDEN, run_scenarios

GOLD = json.loads(GOLDEN.read_text())
ACTUAL = run_scenarios(build_baseline)


def test_same_scenarios():
    assert set(ACTUAL) == set(GOLD) and len(GOLD) >= 15


@pytest.mark.parametrize("name", sorted(GOLD))
def test_baseline_matches_golden(name):
    got, want = dict(ACTUAL[name]), dict(GOLD[name])
    assert got.pop("scores") == pytest.approx(want.pop("scores"), abs=1e-4)
    assert got == want


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ranking_does_not_depend_on_float_noise(seed, monkeypatch):
    """Different CPUs and math libraries (Apple Accelerate, AVX2, AVX-512...) return slightly
    different floats for the same embeddings: 0.0 on one machine, 1e-10 on another. Ranking must
    not depend on that noise, or the golden test passes on one laptop and fails on the next."""
    import numpy as np

    from supportdesk.llm.fake import FakeEmbeddingClient

    original = FakeEmbeddingClient.embed
    rng = np.random.default_rng(seed)

    def noisy(self, texts, *, stage="embed"):
        vectors, usage = original(self, texts, stage=stage)
        return [list(np.asarray(v, dtype=np.float64) + rng.uniform(-1e-9, 1e-9, len(v))) for v in vectors], usage

    monkeypatch.setattr(FakeEmbeddingClient, "embed", noisy)
    noisy_results = run_scenarios(build_baseline)
    for name in sorted(GOLD):
        got, want = dict(noisy_results[name]), dict(GOLD[name])
        assert got.pop("scores") == pytest.approx(want.pop("scores"), abs=1e-4), name
        assert got == want, name
