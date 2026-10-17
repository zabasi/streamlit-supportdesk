"""The controlled answer step retries a dropped connection, and nothing else."""
import json

import pytest

from supportdesk.llm.errors import BudgetExhausted, ProviderError
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.packet import Packet
from supportdesk.pipelines.stages.controlled import ControlledGenerator
from supportdesk.pipelines.stages.protocols import GenerationFailed
from supportdesk.types import CaseInput, Chunk
from tests.test_answer_model import live
from tests.test_controlled_v2 import assessment


class Flaky:
    """A provider that fails the first calls with the given errors, then answers with ``text``."""

    def __init__(self, failures, text):
        self.failures, self.text, self.calls = list(failures), text, 0

    def chat(self, **_):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"text": self.text, "finish": {"finish_reason": "stop", "refusal": None},
                "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 300}}

    def embed(self, *, model, texts):
        return {"vectors": [[1.0] for _ in texts], "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 0}}


def run(settings, tmp_path, provider):
    s = live(settings, tmp_path, "gpt-5.4-mini")
    gen = ControlledGenerator(build_clients(s, transport=provider), s)
    doc = Chunk(chunk_id="doc:0#0", source_id="doc:0", source_type="documentation", title="Doc", text="Widgets keep state.")
    case = CaseInput(case_id="C-1", title="Value lost", body="The widget value is lost after a rerun.")
    return gen(case, {}, Packet(chunks=[doc], labels={"S1": doc.chunk_id}))


def test_a_dropped_connection_is_retried_once_and_recorded(settings, tmp_path):
    provider = Flaky([ProviderError("Connection error.")], json.dumps(assessment()))
    answer = run(settings, tmp_path, provider)
    assert answer.outcome == "answer" and provider.calls == 2
    assert answer.attempts[0]["retries"] == 1


def test_a_second_drop_is_raised_after_one_retry(settings, tmp_path):
    provider = Flaky([ProviderError("Connection error."), ProviderError("Connection error.")], json.dumps(assessment()))
    with pytest.raises(ProviderError):
        run(settings, tmp_path, provider)
    assert provider.calls == 2


def test_a_budget_stop_is_not_retried(settings, tmp_path):
    provider = Flaky([BudgetExhausted("quota")], json.dumps(assessment()))
    with pytest.raises(BudgetExhausted):
        run(settings, tmp_path, provider)
    assert provider.calls == 1


def test_an_invalid_assessment_is_not_retried(settings, tmp_path):
    provider = Flaky([], "not json")
    with pytest.raises(GenerationFailed):
        run(settings, tmp_path, provider)
    assert provider.calls == 1
