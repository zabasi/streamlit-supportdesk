import numpy as np
import pytest

from supportdesk.retrieval.bm25 import BM25
from supportdesk.retrieval.filters import visible_mask
from supportdesk.retrieval.fusion import rrf
from supportdesk.retrieval.index import IndexUnusable, VectorIndex, build_index
from supportdesk.retrieval.retriever import BM25Retriever, DenseRetriever, HybridRetriever

QUERY = "session state lost after browser refresh F5 logged-in user"


def test_dense_finds_the_session_doc(index, clients):
    hits = DenseRetriever(index, clients.embedding).search(QUERY, k=3)
    assert "fixture-doc-session-state" in [h.chunk.source_id for h in hits]
    assert [h.rank for h in hits] == [1, 2, 3]


def test_bm25_and_hybrid_find_identifiers(index, clients):
    hits = BM25Retriever(index.chunks, k1=1.5, b=0.75).search("WebSocket Upgrade Connection headers nginx", k=3)
    assert hits[0].chunk.source_id in {"fixture-doc-deploy-proxy", "fixture-comment-4"}
    hybrid = HybridRetriever(index, clients.embedding, pool=30, rrf_k=60, bm25_k1=1.5,
                             bm25_b=0.75).search(QUERY, k=3)
    assert hybrid[0].source == "hybrid" and len(hybrid) == 3


def test_as_of_hides_the_future(index, clients):
    early = "2024-03-11T11:00:00Z"
    hits = DenseRetriever(index, clients.embedding).search(
        "widget state dropped copy value session_state different key", k=12, as_of=early)
    ids = {h.chunk.source_id for h in hits}
    assert "fixture-comment-3" not in ids  # posted 2024-03-12
    assert "fixture-release-1.33.0" not in ids  # released 2024-04-04
    assert "fixture-doc-session-state" in ids  # undated docs stay visible
    mask = visible_mask(index.chunks, None)
    assert all(mask)


def test_rrf_is_deterministic_and_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert [cid for cid, _ in fused][:2] == ["a", "c"]
    assert rrf([["x"], ["x"]], k=60) == rrf([["x"], ["x"]], k=60)


def test_bm25_scores_zero_for_unrelated(index):
    scores = BM25(index.chunks, k1=1.5, b=0.75).scores("zzzzunknownterm")
    assert max(scores) == 0.0


def test_index_roundtrip_and_model_guard(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx", expect_model="fake-embedding")
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    probe = [1.0] + [0.0] * (index.dimension - 1)
    for as_of in (None, "2024-04-20T10:00:00Z", "2000-01-01T00:00:00Z"):  # same answers, with the date filter
        before, after = index.search(probe, limit=50, as_of=as_of), loaded.search(probe, limit=50, as_of=as_of)
        assert [i for i, _ in before] == [i for i, _ in after]
        assert np.allclose([s for _, s in before], [s for _, s in after])
    with pytest.raises(IndexUnusable, match="rebuild"):
        VectorIndex.load(tmp_path / "idx", expect_model="some-other-model")
    with pytest.raises(IndexUnusable, match="no index"):
        VectorIndex.load(tmp_path / "nowhere")


def test_manifest_records_what_was_built(index):
    m = index.manifest
    assert m["chunk_count"] == len(index.chunks) and m["embedding_model"] == "fake-embedding"
    assert m["chunks_by_type"]["documentation"] == 3 and len(m["chunk_ids_sha256"]) == 64


def test_embedder_length_mismatch_is_rejected(chunks, settings):
    from supportdesk.llm.factory import build_clients
    clients = build_clients(settings)
    clients.embedding.script = {1: "short_batch"}
    with pytest.raises(IndexUnusable, match="vectors for"):
        build_index(chunks, clients.embedding, embedding_model="fake-embedding",
                    chunking=settings.chunking)


def test_vector_store_date_filter_matches_the_visibility_rule(index):
    """Qdrant does the date filtering now; it must agree exactly with visible_mask: undated chunks
    always visible, dated ones only if strictly before the case."""
    from supportdesk.retrieval.filters import visible_mask
    probe = [1.0] + [0.0] * (index.dimension - 1)
    stamps = sorted({c.as_of for c in index.chunks if c.as_of})
    for as_of in [None, "1990-01-01T00:00:00Z", *stamps, "2100-01-01T00:00:00Z"]:  # each stamp: strictness
        found = {i for i, _ in index.search(probe, limit=len(index.chunks), as_of=as_of)}
        expected = {i for i, ok in enumerate(visible_mask(index.chunks, as_of)) if ok}
        assert found == expected, as_of


def test_a_built_index_can_be_copied_and_shared_without_reindexing(index, tmp_path):
    """Build once, share the folder: a copy on another machine loads and answers identically, and loading
    spends nothing (no embedding calls)."""
    import shutil
    index.save(tmp_path / "built")
    shutil.copytree(tmp_path / "built", tmp_path / "elsewhere" / "index")
    copy = VectorIndex.load(tmp_path / "elsewhere" / "index", expect_model="fake-embedding")
    probe = [0.5, 1.0] + [0.0] * (index.dimension - 2)
    for as_of in (None, "2024-04-20T10:00:00Z"):
        mine, theirs = index.search(probe, limit=50, as_of=as_of), copy.search(probe, limit=50, as_of=as_of)
        assert [i for i, _ in mine] == [i for i, _ in theirs]
        assert np.allclose([s for _, s in mine], [s for _, s in theirs], atol=1e-6)  # float noise only
    assert copy.manifest["chunk_ids_sha256"] == index.manifest["chunk_ids_sha256"]
