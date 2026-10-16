"""prompts.rerank_second (two rankings merged) and retrieval.bm25_extra."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.llm_rerank import LLMReranker, merge_orders
from supportdesk.retrieval.retriever import DocSlotsRetriever, Hit
from supportdesk.retrieval.section_questions import QuestionExpandedRetriever
from supportdesk.types import CaseInput
from tests.test_section_questions import bank_for


def test_merging_two_rankings():
    a, b = [0, 1, 2, 3], [3, 2, 1, 0]
    assert merge_orders(a, b, 4, mode="alternate", k=10) == [0, 3, 1, 2]
    assert set(merge_orders(a, b, 4, mode="rrf", k=10)) == {0, 1, 2, 3}
    assert merge_orders([0, 1, 2], [0, 2, 1], 3, mode="rrf", k=10)[0] == 0  # first in both: first
    assert merge_orders([], b, 4, mode="rrf", k=10) == b and merge_orders(a, [], 4, mode="rrf", k=10) == a


def test_split_takes_exactly_n_from_the_first_and_the_rest_from_the_second():
    first = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    second = [1, 9, 0, 8, 7, 6, 2, 3, 4, 5]  # its best two overlap the first ranking's best four
    merged = merge_orders(first, second, 10, mode="split", k=10, top_k=8, split=4)
    assert merged[:4] == [0, 1, 2, 3]  # the first ranking's best four
    assert merged[4:8] == [9, 8, 7, 6]  # the second's best four not already taken
    assert sorted(merged) == list(range(10))  # every candidate exactly once
    capped = merge_orders(first, second, 10, mode="split", k=10, top_k=3, split=4)
    assert capped[:3] == [0, 1, 2]  # never more than top_k from the first


def test_a_second_prompt_makes_a_second_ranking_with_its_own_instructions(index, clients, settings, tmp_path):
    second = tmp_path / "second.txt"
    second.write_text("SECOND PROMPT: rank by cause or fix")
    hits = [Hit(c, 1.0, n, "dense") for n, c in enumerate(index.chunks[:4], start=1)]
    case = CaseInput(case_id="c", title="t", body="widget value lost")
    one = LLMReranker(clients, settings)
    before = clients.chat.calls
    one(case, "q", hits)
    assert clients.chat.calls - before == 1
    s = dataclasses.replace(settings, prompts=dataclasses.replace(settings.prompts, rerank_second=str(second)))
    two = LLMReranker(clients, s)
    before = clients.chat.calls
    out = two(case, "q", hits)
    assert clients.chat.calls - before == 2
    systems = [m[0]["content"] for m in clients.chat.seen[-2:]]
    assert systems[0] != systems[1] and systems[1] == "SECOND PROMPT: rank by cause or fix"
    assert sorted(h.chunk.chunk_id for h in out) == sorted(h.chunk.chunk_id for h in hits)


def test_bm25_extras_are_new_documentation_chunks(index, clients):
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=3)
    r = QuestionExpandedRetriever(base, bank_for(index, clients), extra=0, bm25_extra=2)
    plain = [h.chunk.chunk_id for h in base.search("session state widget value", k=4)]
    hits = r.search("session state widget value", k=4)
    assert [h.chunk.chunk_id for h in hits[: len(plain)]] == plain
    extra = hits[len(plain):]
    assert len(extra) <= 2 and all(h.source == "bm25" and h.chunk.source_type == "documentation" for h in extra)
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_the_settings_are_checked():
    s = Settings.load(env={})
    assert (s.prompts.rerank_second, s.retrieval.rerank_merge, s.retrieval.bm25_extra) == ("", "rrf", 0)
    for bad in ("retrieval.rerank_merge=vote", "retrieval.bm25_extra=-1", "retrieval.rerank_split=-1"):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=[bad])
