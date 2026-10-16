"""rerank_fuse_weight (#1), question_filter_top (#2, Doc2Query--) and question_chunk (#3)."""

import numpy as np
import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.llm_rerank import fuse_with_search
from supportdesk.retrieval.retriever import DocSlotsRetriever, Hit
from supportdesk.retrieval.section_questions import QuestionBank, QuestionExpandedRetriever
from supportdesk.types import Chunk
from tests.test_section_questions import bank_for


def hit(n, rank, question=None):
    c = Chunk(chunk_id=f"c{n}", source_id=f"s{n}", source_type="documentation", title="t", text="x",
              start_char=0, end_char=1, meta={"matched_question": question} if question else {})
    return Hit(c, 1.0, rank, "question" if question else "dense")


def test_fusion_blends_the_reranker_order_with_each_sources_own_rank():
    hits = [hit(0, 1), hit(1, 2), hit(2, 3), hit(3, 4, "q"), hit(4, 5, "q")]
    reranked = list(reversed(hits))  # the reranker put everything upside down
    assert [h.chunk.chunk_id for h in fuse_with_search(reranked, hits, weight=0.0, k=10)] == \
        [h.chunk.chunk_id for h in reranked]
    strong = [h.chunk.chunk_id for h in fuse_with_search(reranked, hits, weight=100.0, k=10)]
    assert strong.index("c0") < strong.index("c1") < strong.index("c2")  # the base's own order
    assert strong.index("c3") < strong.index("c4")  # the question-found candidates' own order
    assert {strong[0], strong[1]} == {"c0", "c3"}  # each source's best comes first


def test_the_filter_keeps_questions_whose_nearest_chunk_is_in_their_section(index, clients):
    bank = bank_for(index, clients)  # each section's first question is its own chunk's text
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=3)
    r = QuestionExpandedRetriever(base, bank, extra=3, filter_top=1)
    first_rows = [row for row in range(len(bank.owner)) if row == 0 or bank.owner[row] != bank.owner[row - 1]]
    assert all(r.allowed[row] for row in first_rows)
    assert r.search("widget value lost", k=4)


def test_the_chunk_most_similar_to_the_question_is_chosen(clients):
    from types import SimpleNamespace

    page = [Chunk(chunk_id=f"p:{n}", source_id="doc:p", source_type="documentation", title="P", text="x" * 10,
                  start_char=10 * n, end_char=10 * (n + 1)) for n in range(2)]
    base = SimpleNamespace(index=SimpleNamespace(chunks=page), embedder=None)
    section = {"source_id": "doc:p", "start_char": 0, "end_char": 20, "title": "P", "heading": "h"}
    vectors = np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]])  # the second chunk is about what the question asks
    bank = QuestionBank([section], [["q"]], np.array([[0, 0, 0, 1.0]]), clients.embedding_model)
    by_question = QuestionExpandedRetriever(base, bank, extra=1, chunk_mode="question", chunk_vectors=vectors)
    by_overlap = QuestionExpandedRetriever(base, bank, extra=1, chunk_mode="overlap", chunk_vectors=vectors)
    assert by_question._chunk_for(section, 0) == 1
    assert by_overlap._chunk_for(section, 0) == 0  # equal overlap: the first


def test_the_settings_are_checked():
    s = Settings.load(env={})
    assert (s.retrieval.rerank_fuse_weight, s.retrieval.question_filter_top, s.retrieval.question_chunk) == \
        (0.0, 0, "overlap")
    for bad in ("retrieval.question_chunk=best", "retrieval.rerank_fuse_weight=-1", "retrieval.question_filter_top=-2"):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=[bad])
