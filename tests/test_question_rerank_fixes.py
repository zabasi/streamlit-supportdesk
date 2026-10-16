"""question_snippet (A), question_order (B) and question_keep for the question_doc_slots retriever."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.llm_rerank import LLMReranker, keep_question_places
from supportdesk.retrieval.retriever import DocSlotsRetriever, Hit
from supportdesk.retrieval.section_questions import QuestionExpandedRetriever
from supportdesk.types import CaseInput, Chunk
from tests.test_section_questions import bank_for

FRONT = "multiselect callback selection " * 30  # longer than one 400-character window
BACK = "\n\nforms batch submit button together " * 12


def chunk(n, source_type="documentation", question=None, text=None):
    meta = {"matched_question": question} if question else {}
    return Chunk(chunk_id=f"c{n}", source_id=f"doc:{n}", source_type=source_type, title=f"T{n}",
                 text=text or f"text {n}", start_char=0, end_char=len(text or f"text {n}"), meta=meta)


def test_a_question_found_candidate_is_shown_by_its_questions_words(clients, settings):
    found = chunk(1, question="How do I submit forms together with one button?", text=FRONT + BACK)
    case = CaseInput(case_id="c", title="multiselect callback", body="callback on every selection")
    for mode, expect, avoid in (("question", "forms batch submit", "multiselect callback selection"),
                                ("report", "multiselect callback selection", "forms batch submit")):
        s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, question_snippet=mode,
                                                                        rerank_snippet="focused"))
        LLMReranker(clients, s).model_text(case, [Hit(found, 1.0, 1, "question"), Hit(chunk(2), 0.9, 2, "dense")])
        listing = clients.chat.seen[-1][1]["content"].split("C1:")[1].split("C2:")[0]
        assert expect in listing and avoid not in listing, mode


def test_interleave_shuffle_and_append(index, clients):
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=3)
    bank = bank_for(index, clients)
    query = "widget value lost"
    appended = QuestionExpandedRetriever(base, bank, extra=3, order="append").search(query, k=4)
    sources = [h.source for h in appended]
    assert sources == sorted(sources, key=lambda s: s == "question")  # the base's first
    mixed = QuestionExpandedRetriever(base, bank, extra=3, order="interleave").search(query, k=4)
    found = [h.source == "question" for h in mixed]
    if sum(found) >= 1 and len(found) > 2:
        assert found[1]  # a question-found candidate in the second position
    shuffled = QuestionExpandedRetriever(base, bank, extra=3, order="shuffle")
    a, b = shuffled.search(query, k=4), shuffled.search(query, k=4)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]  # reproducible
    assert sorted(h.chunk.chunk_id for h in a) == sorted(h.chunk.chunk_id for h in appended)
    assert [h.rank for h in a] == list(range(1, len(a) + 1))


def test_reserved_places_promote_question_found_candidates():
    ranked = [Hit(chunk(n, "comment" if n % 2 else "documentation"), 1.0, n, "dense") for n in range(6)]
    ranked += [Hit(chunk(10 + n, question=f"q{n}"), 0.5, 7 + n, "question") for n in range(3)]
    kept = keep_question_places(ranked, top_k=4, minimum=2)[:4]
    assert sum(bool(h.chunk.meta.get("matched_question")) for h in kept) == 2
    assert kept[0].chunk.chunk_id == "c0"  # the best-ranked stay; the lowest-ranked others make room


def test_the_settings_are_checked():
    s = Settings.load(env={})
    assert (s.retrieval.question_snippet, s.retrieval.question_order, s.retrieval.question_keep) == ("report", "append", 0)
    for bad in ("retrieval.question_snippet=title", "retrieval.question_order=random", "retrieval.question_keep=99"):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=[bad])
