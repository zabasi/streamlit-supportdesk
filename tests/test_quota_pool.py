"""retrieval.pool_mode = quota: fixed shares of the reranker's candidates per query and search method."""
import dataclasses
from collections import Counter

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.state import case_to_dict, hit_from_dict
from supportdesk.registry import make_pipeline
from supportdesk.types import CaseInput

CASE = CaseInput(case_id="c", title="Widget value lost", body="st.text_input value lost after st.button rerun")


def pipeline(index, clients, settings, **retrieval):
    s = dataclasses.replace(settings,
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                            retrieval=dataclasses.replace(settings.retrieval, pool_mode="quota", **retrieval))
    return make_pipeline("advanced", "dense_doc_slots", index, clients, s)


def candidates(p, queries):
    state = {"case": case_to_dict(CASE), "queries": queries}
    return [hit_from_dict(h) for h in p._retrieve_node(state)["candidates"]]


def test_the_pool_takes_fixed_shares_without_duplicates(index, clients, settings):
    p = pipeline(index, clients, settings)
    hits = candidates(p, ["text_input value lost after button rerun", "widget values across reruns",
                          "session state callbacks"])
    sources = Counter(h.source for h in hits)
    assert sources["report-dense"] <= 5 and sources["report-bm25"] <= 5 and sources["concept-dense"] <= 10
    assert sources["report-dense"] == min(5, len(index.chunks))
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)  # every chunk once
    assert [h.source for h in hits][: sources["report-dense"]] == ["report-dense"] * sources["report-dense"]


def test_without_concept_queries_only_the_report_shares_are_used(index, clients, settings):
    hits = candidates(pipeline(index, clients, settings), ["text_input value lost after button rerun"])
    assert {h.source for h in hits} <= {"report-dense", "report-bm25"} and len(hits) <= 10


def test_the_shares_are_settings(index, clients, settings):
    p = pipeline(index, clients, settings, quota_report_dense=2, quota_report_bm25=0, quota_concepts_dense=3)
    sources = Counter(h.source for h in candidates(p, ["value lost after rerun", "widget state reruns"]))
    assert sources["report-dense"] <= 2 and sources["report-bm25"] == 0 and sources["concept-dense"] <= 3


def test_advanced_runs_end_to_end_with_the_quota_pool(index, clients, settings):
    assert pipeline(index, clients, settings).run(CASE).retrieved


def test_pool_mode_is_a_checked_setting():
    assert Settings.load(env={}).retrieval.pool_mode == "fuse"  # default: advanced unchanged
    with pytest.raises(ConfigError, match="pool_mode"):
        Settings.load(env={}, overrides=["retrieval.pool_mode=mixed"])
