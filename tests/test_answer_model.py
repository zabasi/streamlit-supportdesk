"""pipeline.answer_model: the package's request, sent unchanged to a cheaper model when chosen."""
import dataclasses
import json

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.packet import Packet
from supportdesk.pipelines.stages.controlled import MODEL, ControlledGenerator
from supportdesk.types import CaseInput, Chunk
from tests.test_controlled_v2 import assessment


class Recording:
    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, *, model, messages, max_output_tokens, temperature, json_mode, extra=None):
        self.calls.append({"model": model, "extra": extra or {}, "messages": messages})
        return {"text": self.text, "finish": {"finish_reason": "stop", "refusal": None},
                "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 300}}

    def embed(self, *, model, texts):
        return {"vectors": [[1.0] for _ in texts], "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 0}}


def live(settings, tmp_path, model):
    return dataclasses.replace(
        settings, run=dataclasses.replace(settings.run, mode="live"),
        pipeline=dataclasses.replace(settings.pipeline, answer_model=model),
        paths=dataclasses.replace(settings.paths, ledger_path=tmp_path / "l.jsonl", cache_path=tmp_path / "c.sqlite"),
        budget=dataclasses.replace(settings.budget, ceiling_usd=5.0, reserve_usd=0.0))


def test_the_default_answer_model_is_unchanged():
    assert Settings.load(env={}).pipeline.answer_model == MODEL == "gpt-5.4-mini"


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-4.1-mini"])
def test_the_packages_request_goes_unchanged_to_the_chosen_model(settings, tmp_path, model):
    s = live(settings, tmp_path, model)
    stub = Recording(json.dumps(assessment()))
    gen = ControlledGenerator(build_clients(s, transport=stub), s)
    case = CaseInput(case_id="C-1", title="Value lost", body="The widget value is lost after a rerun.")
    doc = Chunk(chunk_id="doc:0#0", source_id="doc:0", source_type="documentation", title="Doc", text="Widgets keep state.")
    answer = gen(case, {}, Packet(chunks=[doc], labels={"S1": doc.chunk_id}))
    call = stub.calls[-1]
    assert call["model"] == model  # sent to the chosen model
    assert call["extra"]["reasoning_effort"] == "medium" and call["extra"]["response_format"]["json_schema"]["strict"]
    assert answer.attempts[0]["model"] == model  # and the answer says which model wrote it


def test_an_unknown_answer_model_is_refused():
    with pytest.raises(ConfigError, match="answer_model"):
        Settings.load(env={}, overrides=["pipeline.answer_model=gpt-4o-mini"])


def test_advanced_gpt5mini_is_advanced_with_only_the_answer_model_changed(monkeypatch):
    from pathlib import Path

    from tests.test_candidates import flat
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=advanced"]))
    cheaper = flat(Settings.load(env={}, overrides=["pipeline.name=advanced-gpt5mini"]))
    assert {k: cheaper[k] for k in cheaper if cheaper[k] != advanced[k]} == {"pipeline.answer_model": "gpt-5-mini"}
