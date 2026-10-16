"""class_doc_slots: the hierarchical-class retriever (topic labels per chunk), through the normal settings."""
import dataclasses
import json

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.class_routing import (
    TOPICS,
    ClassRoutedRetriever,
    heuristic_label,
    load_class_labels,
)
from supportdesk.retrieval.index import IndexUnusable
from supportdesk.types import CaseInput


def write_labels(path, index):
    path.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "label": heuristic_label(c.text)}) + "\n"
                            for c in index.chunks), encoding="utf-8")


def with_classes(settings, path, **retrieval):
    return dataclasses.replace(
        settings, paths=dataclasses.replace(settings.paths, class_labels=path),
        retrieval=dataclasses.replace(settings.retrieval, **retrieval),
        pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"))


def build(index, clients, settings, path, **retrieval):
    p = make_pipeline("advanced", "class_doc_slots", index, clients, with_classes(settings, path, **retrieval))
    return p, p.retriever


def test_filter_searches_only_the_routed_classes(index, clients, settings, tmp_path):
    write_labels(tmp_path / "l.jsonl", index)
    _, r = build(index, clients, settings, tmp_path / "l.jsonl", class_route="centroid", class_mode="filter", class_top=1)
    assert isinstance(r, ClassRoutedRetriever)
    hits = r.search("session state lost after rerun", k=6)
    routed = set().union(*(r.members[c] for c in r.last_route))
    position = {c.chunk_id: i for i, c in enumerate(index.chunks)}
    assert hits and all(position[h.chunk.chunk_id] in routed for h in hits)


def test_boost_keeps_every_chunk_findable(index, clients, settings, tmp_path):
    write_labels(tmp_path / "l.jsonl", index)
    _, r = build(index, clients, settings, tmp_path / "l.jsonl", class_route="centroid", class_mode="boost",
                 class_boost=0.0)
    plain = r.search("widget value", k=6)
    assert {h.source for h in plain} <= {"class", "dense"} and len(plain) == min(6, len(index.chunks))


def test_llm_routing_falls_back_to_keywords_on_an_unusable_answer(index, clients, settings, tmp_path):
    write_labels(tmp_path / "l.jsonl", index)
    _, r = build(index, clients, settings, tmp_path / "l.jsonl", class_route="llm")
    r.search("st.cache_data does not refresh", k=4)  # the fake chat's answer has no labels
    assert r.last_route and set(r.last_route) <= set(TOPICS)


def test_labels_for_another_index_or_a_missing_file_are_refused(index, tmp_path):
    with pytest.raises(IndexUnusable, match="build_class_labels"):
        load_class_labels(tmp_path / "none.jsonl", index)
    (tmp_path / "other.jsonl").write_text(json.dumps({"chunk_id": "nope", "label": "caching"}) + "\n")
    with pytest.raises(IndexUnusable, match="another index"):
        load_class_labels(tmp_path / "other.jsonl", index)


def test_advanced_runs_with_the_class_retriever(index, clients, settings, tmp_path):
    write_labels(tmp_path / "l.jsonl", index)
    p, _ = build(index, clients, settings, tmp_path / "l.jsonl", class_route="llm", class_mode="boost")
    assert p.run(CaseInput(case_id="c", title="Session state lost", body="st.session_state lost")).retrieved


def test_class_settings_are_checked():
    s = Settings.load(env={})
    assert (s.retrieval.class_route, s.retrieval.class_mode, s.retrieval.class_top) == ("llm", "filter", 2)
    for bad in ("retrieval.class_route=fuzzy", "retrieval.class_mode=maybe", "retrieval.class_top=0"):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=[bad])
