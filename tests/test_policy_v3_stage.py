"""pipeline.generator = policy_v3: the policy-v3 answer step with the prompt in prompts.policy."""
import dataclasses
from pathlib import Path

import pytest

from supportdesk.generation.policy_v3 import PolicyTurn
from supportdesk.pipelines.packet import Packet
from supportdesk.pipelines.stages import policy as stage
from supportdesk.pipelines.stages.protocols import GenerationFailed
from supportdesk.registry import make_pipeline
from supportdesk.types import CaseInput

ROOT = Path(__file__).resolve().parents[1]
V3B = "configs/prompts/policy_v3b_system.txt"
CASE = CaseInput(case_id="c", title="Value lost", body="The widget value is lost after a rerun.")


def with_policy(settings, prompt=V3B):
    return dataclasses.replace(settings, prompts=dataclasses.replace(settings.prompts, policy=str(ROOT / prompt)))


def test_the_chosen_prompt_reaches_the_model_and_a_bad_answer_is_an_error(clients, settings, index):
    gen = stage.PolicyV3Generator(clients, with_policy(settings))
    hits = index.chunks[:2]
    packet = Packet(chunks=list(hits), labels={f"S{n}": c.chunk_id for n, c in enumerate(hits, start=1)})
    with pytest.raises(GenerationFailed, match="policy_v3"):  # the fake model's answer is not a v3 record
        gen(CASE, {}, packet)
    system = gen.chat.seen[-1][0]["content"]
    assert system == (ROOT / V3B).read_text(encoding="utf-8")


def test_a_valid_action_becomes_the_pipelines_answer(clients, settings, monkeypatch):
    turn = PolicyTurn(case_id="c", question="q", outcome="request_info", answer=None,
                      user_reply="Which Streamlit version do you use?", citations=["c1"],
                      maintainer_summary={"next_step": "ask for the version"}, raw_output="{}")
    monkeypatch.setattr(stage, "run_policy_turn", lambda *a, **k: turn)
    answer = stage.PolicyV3Generator(clients, with_policy(settings))(CASE, {}, Packet())
    assert (answer.outcome, answer.user_reply, answer.next_step, answer.citations) == (
        "request_info", "Which Streamlit version do you use?", "ask for the version", ["c1"])
    assert answer.attempts[0]["prompt"].endswith("policy_v3b_system.txt")


def test_an_app_case_gets_the_extracted_environment(clients, settings, monkeypatch):
    seen = []
    monkeypatch.setattr(stage, "run_policy_turn",
                        lambda case, *a, **k: seen.append(case.environment) or PolicyTurn(
                            case_id="c", question="q", outcome="answer", answer="a", user_reply="a"))
    gen = stage.PolicyV3Generator(clients, with_policy(settings))
    gen(CASE, {"streamlit_version": "1.40.0"}, Packet())
    gen(dataclasses.replace(CASE, environment={"streamlit_version": "1.13.0"}), {"streamlit_version": "1.40.0"}, Packet())
    assert seen == [{"streamlit_version": "1.40.0"}, {"streamlit_version": "1.13.0"}]


def test_advanced_uses_it_when_selected_and_the_baseline_never_does(index, clients, settings):
    s = dataclasses.replace(with_policy(settings), pipeline=dataclasses.replace(
        settings.pipeline, name="advanced", scope_gate="off", generator="policy_v3"))
    assert isinstance(make_pipeline("advanced", "dense_doc_slots", index, clients, s).generator, stage.PolicyV3Generator)
    base = dataclasses.replace(s, pipeline=dataclasses.replace(s.pipeline, name="baseline"))
    assert not isinstance(make_pipeline("baseline", "dense", index, clients, base).generator, stage.PolicyV3Generator)
