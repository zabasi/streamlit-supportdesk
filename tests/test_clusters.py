"""Cluster routing experiment: k-means clusters with keywords, and a retriever that searches only the
clusters a query is routed to (filter) or ranks them higher (boost)."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.clusters import (
    ClusterRoutedRetriever,
    distinctive_words,
    index_vectors,
    kmeans,
    load_clusters,
)
from supportdesk.retrieval.index import IndexUnusable, VectorIndex
from supportdesk.types import CaseInput

ROOT = Path(__file__).resolve().parents[1]


def build_script():
    spec = importlib.util.spec_from_file_location("build_clusters", ROOT / "scripts" / "build_clusters.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_clusters(path, index, keywords=("zebra", "yak")):
    """Two clusters: the documentation chunks, and everything else."""
    docs = [c.chunk_id for c in index.chunks if c.source_type == "documentation"]
    rest = [c.chunk_id for c in index.chunks if c.source_type != "documentation"]
    rows = [{"cluster": 0, "label": "docs", "keywords": [keywords[0]], "chunk_ids": docs},
            {"cluster": 1, "label": "issues", "keywords": [keywords[1]], "chunk_ids": rest}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return set(docs), set(rest)


def retriever(index, clients, path, **kw):
    options = {"doc_slots": 3, "clusters": path, "top": 1, "route": "keywords", "mode": "filter", "boost": 0.05}
    return ClusterRoutedRetriever(index, clients.embedding, **{**options, **kw})


def test_kmeans_is_deterministic_and_covers_every_row():
    rng = np.random.default_rng(1)
    vectors = np.vstack([rng.normal(0, 0.1, (20, 8)) + np.eye(8)[0], rng.normal(0, 0.1, (20, 8)) + np.eye(8)[3]])
    labels = kmeans(vectors, 2, seed=0)
    assert labels.tolist() == kmeans(vectors, 2, seed=0).tolist()  # the same seed, the same clusters
    assert set(labels[:20]) == {0} and set(labels[20:]) == {1}  # the two groups found, numbered by first row
    assert len(set(kmeans(vectors, 50).tolist())) <= 40  # never more clusters than rows


def test_distinctive_words_are_the_clusters_own():
    words = distinctive_words(["session state widget", "session state callback"], ["deploy server", "deploy cloud"])
    assert words[:2] == ["session", "state"] and "deploy" not in words


def test_vectors_are_read_back_in_chunk_order(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    vectors = index_vectors(loaded)
    assert vectors.shape[0] == len(index.chunks) and np.allclose(vectors, index_vectors(index))
    position, score = loaded.search(list(vectors[0]), limit=1, as_of=None)[0]  # a chunk is its own best match
    assert position == 0 and score == pytest.approx(1.0, abs=1e-5)


def test_build_clusters_covers_every_chunk_once(index, clients, settings, tmp_path):
    s = dataclasses.replace(settings,
                            paths=dataclasses.replace(settings.paths, clusters=tmp_path / "clusters.jsonl"),
                            retrieval=dataclasses.replace(settings.retrieval, cluster_size=max(1, len(index.chunks) // 3)))
    out = build_script().build(s, index, clients)
    rows = load_clusters(out, index)  # checked against the index: every chunk in exactly one cluster
    assert len(rows) >= 2 and all(r["keywords"] and r["label"] for r in rows)
    assert {r["labelled_by"] for r in rows} <= {"model", "words"}  # the fake chat's answer is unusable: words
    assert json.loads(out.with_suffix(".manifest.json").read_text())["clusters"] == len(rows)


def test_filter_searches_only_the_routed_cluster(index, clients, tmp_path):
    docs, rest = write_clusters(tmp_path / "c.jsonl", index)
    r = retriever(index, clients, tmp_path / "c.jsonl")
    hits = r.search("zebra widget value", k=6)
    assert r.last_route == [0] and hits and {h.chunk.chunk_id for h in hits} <= docs
    hits = r.search("yak deploy problem", k=6)
    assert r.last_route == [1] and {h.chunk.chunk_id for h in hits} <= rest


def test_boost_keeps_everything_searchable(index, clients, tmp_path):
    docs, _ = write_clusters(tmp_path / "c.jsonl", index)
    r = retriever(index, clients, tmp_path / "c.jsonl", mode="boost", boost=0.0)
    plain = retriever(index, clients, tmp_path / "c.jsonl", mode="boost", boost=0.0)
    hits = r.search("yak issue", k=6)
    assert {h.chunk.chunk_id for h in hits} & docs  # documentation still has its places: nothing is excluded
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in plain.search("yak issue", k=6)]
    assert {h.source for h in hits} <= {"cluster", "dense"}


def test_embedding_routing_embeds_the_cluster_descriptions_once(index, clients, tmp_path):
    write_clusters(tmp_path / "c.jsonl", index)
    r = retriever(index, clients, tmp_path / "c.jsonl", route="embedding", top=2)
    r.search("first query", k=4)
    calls = clients.embedding.calls
    r.search("second query", k=4)
    assert clients.embedding.calls == calls + 1  # only the new query; the descriptions are kept
    assert sorted(r.last_route) == [0, 1]


def test_missing_or_foreign_clusters_are_reported(index, clients, tmp_path):
    with pytest.raises(IndexUnusable, match="build_clusters"):
        retriever(index, clients, tmp_path / "none.jsonl")
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"cluster": 0, "label": "x", "keywords": ["x"], "chunk_ids": ["nope"]}) + "\n")
    with pytest.raises(IndexUnusable, match="another index"):
        retriever(index, clients, path)


def test_advanced_runs_with_the_cluster_retriever(index, clients, settings, tmp_path):
    write_clusters(tmp_path / "c.jsonl", index)
    s = dataclasses.replace(settings, paths=dataclasses.replace(settings.paths, clusters=tmp_path / "c.jsonl"),
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"))
    pipeline = make_pipeline("advanced", "cluster_doc_slots", index, clients, s)
    assert isinstance(pipeline.retriever, ClusterRoutedRetriever)
    assert pipeline.run(CaseInput(case_id="c", title="zebra", body="st.session_state lost")).retrieved


def test_cluster_settings_are_checked():
    s = Settings.load(env={}, overrides=["pipeline.retriever=cluster_doc_slots"])
    assert (s.retrieval.cluster_route, s.retrieval.cluster_mode, s.retrieval.cluster_top) == ("keywords", "filter", 2)
    for bad in ("retrieval.cluster_route=fuzzy", "retrieval.cluster_mode=maybe", "retrieval.cluster_top=0"):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=[bad])
