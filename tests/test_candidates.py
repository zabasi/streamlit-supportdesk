"""The candidate designs are candidate 1 except for their stated differences; advanced is candidate 6."""
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
def test_a_candidate_is_candidate_1_but_for_its_differences(name, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert name in PIPELINES
    reference = flat(Settings.load(env={}, overrides=["pipeline.name=candidate-1-dense-rerank-section"]))
    candidate = flat(Settings.load(env={}, overrides=[f"pipeline.name={name}"]))
    differs = {k: candidate[k] for k in candidate if candidate[k] != reference[k]}
    assert differs == {k: str(v) for k, v in EXPECTED[name].items()}


def test_advanced_is_the_previous_design_plus_the_two_fixes(monkeypatch):
    """advanced is candidate 6 (the previous advanced) with version notes and complete questions, the same
    in every setting as advanced-notes-questions."""
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=advanced"]))
    previous = flat(Settings.load(env={}, overrides=["pipeline.name=candidate-6-doc2query15-rerank-section-controlled-v2"]))
    named = flat(Settings.load(env={}, overrides=["pipeline.name=advanced-notes-questions"]))
    assert {k: advanced[k] for k in advanced if advanced[k] != previous[k]} == {
        "pipeline.version_notes": "True", "pipeline.complete_questions": "True"}
    assert {k: advanced[k] for k in advanced if k != "pipeline.name" and advanced[k] != named[k]} == {}


def test_advanced_version_notes_is_advanced_with_the_notes_on(monkeypatch):
    """The version-notes pipeline differs from advanced in that one setting, so advanced stays untouched."""
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=candidate-6-doc2query15-rerank-section-controlled-v2"]))  # the previous advanced
    notes = flat(Settings.load(env={}, overrides=["pipeline.name=advanced-version-notes"]))
    assert {k: notes[k] for k in notes if notes[k] != advanced[k]} == {"pipeline.version_notes": "True"}
    assert advanced["pipeline.version_notes"] == "False"


def test_the_notes_and_questions_pipelines_differ_from_advanced_only_in_those_settings(monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=candidate-6-doc2query15-rerank-section-controlled-v2"]))  # the previous advanced
    fixes = {"pipeline.version_notes": "True", "pipeline.complete_questions": "True"}
    for name, extra in (("advanced-notes-questions", {}),
                        ("advanced-notes-questions-gpt5mini", {"pipeline.answer_model": "gpt-5-mini"}),
                        ("advanced-notes-questions-v4-gpt5mini",
                         {"pipeline.answer_model": "gpt-5-mini", "pipeline.controlled_version": "v4"})):
        other = flat(Settings.load(env={}, overrides=[f"pipeline.name={name}"]))
        assert {k: other[k] for k in other if other[k] != advanced[k]} == {**fixes, **extra}, name
