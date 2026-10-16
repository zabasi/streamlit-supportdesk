"""models.rerank: the reranking call can use its own (priced) chat model."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.llm.factory import Clients, chat_client_for
from supportdesk.pipelines.stages.llm_rerank import LLMReranker


def with_rerank(settings, model):
    return dataclasses.replace(settings, models=dataclasses.replace(settings.models, rerank=model))


def test_by_default_the_reranker_uses_the_chat_model(clients, settings):
    assert LLMReranker(clients, settings).chat is clients.chat


def test_a_rerank_model_gets_its_own_client_on_the_same_ledger(clients, settings):
    reranker = LLMReranker(clients, with_rerank(settings, "gpt-4.1-nano"))
    assert reranker.chat is not clients.chat and reranker.chat.ledger is clients.ledger


def test_a_model_without_a_price_is_refused(clients, settings, tmp_path):
    s = dataclasses.replace(settings, paths=dataclasses.replace(settings.paths, cache_path=tmp_path / "cache.sqlite"))
    live = Clients("live", clients.chat, clients.embedding, clients.ledger, clients.embedding_model)
    with pytest.raises(ConfigError, match="no price"):  # checked before any connection is made
        chat_client_for(live, s, "a-model-with-no-price")


def test_rerank_model_defaults_to_empty():
    assert Settings.load(env={}).models.rerank == ""
