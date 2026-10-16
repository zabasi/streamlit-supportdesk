"""dual_doc_slots: two embedding models, two indexes, rankings fused with weighted RRF."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.llm.factory import second_embedding_client
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.dual import DualDocSlotsRetriever, build_dual
from supportdesk.retrieval.index import IndexUnusable, VectorIndex
from supportdesk.retrieval.retriever import DocSlotsRetriever
from supportdesk.types import CaseInput


def dual_settings(settings, index, tmp_path, weight=0.5):
    index.save(tmp_path / "second")
    model = VectorIndex.load(tmp_path / "second").manifest["embedding_model"]
    return dataclasses.replace(
        settings, models=dataclasses.replace(settings.models, second_embedding=model),
        paths=dataclasses.replace(settings.paths, second_index_dir=tmp_path / "second"),
        retrieval=dataclasses.replace(settings.retrieval, second_weight=weight),
        pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"))


def test_the_dual_search_fuses_both_rankings(index, clients, settings, tmp_path):
    retriever = build_dual(index, clients, dual_settings(settings, index, tmp_path))
    hits = retriever.search("session state lost after rerun", k=6)
    assert 0 < len(hits) <= 6 and {h.source for h in hits} == {"dual"}
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_weight_zero_is_the_main_models_ranking(index, clients, settings, tmp_path):
    s = dual_settings(settings, index, tmp_path, weight=0.0)
    main = DocSlotsRetriever(index, clients.embedding, doc_slots=s.retrieval.doc_slots)
    query = "widget value lost after button click"
    assert [h.chunk.chunk_id for h in build_dual(index, clients, s).search(query, k=6)] == \
        [h.chunk.chunk_id for h in main.search(query, k=6)]


def test_a_second_index_with_other_chunks_is_refused(index, clients, settings, tmp_path):
    s = dual_settings(settings, index, tmp_path)
    second = VectorIndex.load(s.paths.second_index_dir)
    second.chunks = second.chunks[:-1]
    with pytest.raises(IndexUnusable, match="different chunks"):
        DualDocSlotsRetriever(DocSlotsRetriever(index, clients.embedding, doc_slots=3),
                              DocSlotsRetriever(second, clients.embedding, doc_slots=3), weight=0.5, rrf_k=60)


def test_the_second_model_shares_the_ledger(clients, settings):
    assert second_embedding_client(clients, settings, "text-embedding-3-large").ledger is clients.ledger


def test_advanced_runs_with_the_dual_retriever(index, clients, settings, tmp_path):
    p = make_pipeline("advanced", "dual_doc_slots", index, clients, dual_settings(settings, index, tmp_path))
    assert p.run(CaseInput(case_id="c", title="Session state lost", body="st.session_state lost")).retrieved


def test_second_weight_is_checked():
    assert Settings.load(env={}).retrieval.second_weight == 0.5
    with pytest.raises(ConfigError, match="second_weight"):
        Settings.load(env={}, overrides=["retrieval.second_weight=2"])
