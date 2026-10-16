"""retrieval.rerank_mode (no reranking call) and the union pool keeping question-found candidates."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.llm_rerank import LLMReranker, search_order
from supportdesk.pipelines.state import case_to_dict, hit_from_dict
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

CASE = CaseInput(case_id="c", title="Widget value lost", body="st.text_input value lost after rerun")


def hit(n, score, rank, source="dense", kind="documentation"):
    c = Chunk(chunk_id=f"c{n}", source_id=f"s{n}", source_type=kind, title="t", text="x", start_char=0, end_char=1)
    return Hit(c, score, rank, source)


def test_score_and_rrf_orders_need_no_model_call(clients, settings):
    hits = [hit(0, 0.5, 1), hit(1, 0.4, 2), hit(2, 0.9, 3, "question"), hit(3, 0.3, 4, "question")]
    assert search_order(hits, mode="score", k=10) == [2, 0, 1, 3]
    assert search_order(hits, mode="rrf", k=10)[:2] == [0, 2]  # each source's best first
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, rerank_mode="score",
                                                                    rerank_keep_docs=0))
    before = clients.chat.calls
    out = LLMReranker(clients, s)(CASE, "q", hits)
    assert clients.chat.calls == before  # no reranking call
    assert [h.chunk.chunk_id for h in out] == ["c2", "c0", "c1", "c3"]


def test_without_a_model_call_the_documentation_minimum_still_applies(clients, settings):
    hits = [hit(0, 0.9, 1, kind="comment"), hit(1, 0.8, 2, kind="comment"), hit(2, 0.1, 3)]
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, rerank_mode="score",
                                                                    top_k=2, rerank_keep_docs=1))
    out = LLMReranker(clients, s)(CASE, "q", hits)[:2]
    assert any(h.chunk.source_type == "documentation" for h in out)


def test_the_union_pool_keeps_question_found_candidates(index, clients, settings, tmp_path):
    from tests.test_section_questions import bank_for

    bank_for(index, clients).save(tmp_path / "q")
    s = dataclasses.replace(
        settings, paths=dataclasses.replace(settings.paths, section_questions=tmp_path / "q"),
        pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
        retrieval=dataclasses.replace(settings.retrieval, pool_mode="union", union_report=3, union_per_concept=2,
                                      union_max=40, question_sections=3))
    p = make_pipeline("advanced", "question_doc_slots", index, clients, s)
    state = {"case": case_to_dict(CASE), "queries": ["widget value lost", "session state reruns"]}
    hits = [hit_from_dict(h) for h in p._retrieve_node(state)["candidates"]]
    assert "question" in {h.source for h in hits}  # not cut at union_report
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_rerank_mode_is_checked():
    assert Settings.load(env={}).retrieval.rerank_mode == "llm"
    with pytest.raises(ConfigError, match="rerank_mode"):
        Settings.load(env={}, overrides=["retrieval.rerank_mode=none"])
