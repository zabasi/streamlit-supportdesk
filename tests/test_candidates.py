"""The four candidate retrieval designs are advanced except for their stated differences."""
import dataclasses

import pytest

from supportdesk.config import Settings
from supportdesk.registry import PIPELINES

EXPECTED = {
    "candidate-1-dense-rerank-section": {},
    "candidate-2-doc2query15-rerank-section": {"pipeline.retriever": "question_doc_slots",
                                               "retrieval.question_sections": 15},
    "candidate-3-dense-dualrerank-section": {"prompts.rerank_second": "configs/prompts/rerank_explain_system.txt",
                                             "retrieval.rerank_merge": "alternate"},
    "candidate-4-dense-rerank-section-6chunks": {"retrieval.top_k": 6, "retrieval.rerank_keep_docs": 3,
                                                 "context.max_tokens": 3200},
    "candidate-5-dense-rerank-section-controlled-v2": {"pipeline.generator": "controlled_v2"},
    "candidate-6-doc2query15-rerank-section-controlled-v2": {"pipeline.retriever": "question_doc_slots",
                                                             "retrieval.question_sections": 15,
                                                             "pipeline.generator": "controlled_v2"},
    "candidate-7-dense-dualrerank-section-controlled-v2": {
        "prompts.rerank_second": "configs/prompts/rerank_explain_system.txt", "retrieval.rerank_merge": "alternate",
        "pipeline.generator": "controlled_v2"},
    "candidate-9-dense-rerank-section-policy-v3b": {"pipeline.generator": "policy_v3",
                                                    "prompts.policy": "configs/prompts/policy_v3b_system.txt"},
    "candidate-8-dense-rerank-section-6chunks-controlled-v2": {"retrieval.top_k": 6, "retrieval.rerank_keep_docs": 3,
                                                               "context.max_tokens": 3200,
                                                               "pipeline.generator": "controlled_v2"},
}
SECTIONS = ("chunking", "retrieval", "context", "prompts", "pipeline")


def flat(settings: Settings) -> dict:
    out = {}
    for section in SECTIONS:
        for key, value in dataclasses.asdict(getattr(settings, section)).items():
            out[f"{section}.{key}"] = str(value)
    out.pop("pipeline.name", None)
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_candidate_is_advanced_but_for_its_differences(name, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert name in PIPELINES
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=advanced"]))
    candidate = flat(Settings.load(env={}, overrides=[f"pipeline.name={name}"]))
    differs = {k: candidate[k] for k in candidate if candidate[k] != advanced[k]}
    assert differs == {k: str(v) for k, v in EXPECTED[name].items()}
