"""retrieval.pool_mode = union: the report's candidates all kept, each concept query adding its own share."""
import dataclasses
from collections import Counter

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.state import case_to_dict, hit_from_dict
from supportdesk.registry import make_pipeline
from supportdesk.types import CaseInput

CASE = CaseInput(case_id="c", title="Widget value lost", body="st.text_input value lost after st.button rerun")
REPORT = "text_input value lost after button rerun"
CONCEPTS = ["widget values across reruns", "session state callbacks", "button behavior"]


def pipeline(index, clients, settings, **retrieval):
    s = dataclasses.replace(settings,
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                            retrieval=dataclasses.replace(settings.retrieval, pool_mode="union", **retrieval))
    return make_pipeline("advanced", "dense_doc_slots", index, clients, s)


def candidates(p, queries):
    return [hit_from_dict(h) for h in p._retrieve_node({"case": case_to_dict(CASE), "queries": queries})["candidates"]]


def test_the_reports_candidates_are_all_kept_in_their_order(index, clients, settings):
    p = pipeline(index, clients, settings, union_report=6, union_per_concept=3, union_max=40)
    hits = candidates(p, [REPORT, *CONCEPTS])
    own = [h.chunk.chunk_id for h in p.retriever.search(REPORT, k=6)][:6]
    assert [h.chunk.chunk_id for h in hits[: len(own)]] == own
    assert {h.source for h in hits[: len(own)]} == {"report"}


def test_each_concept_query_adds_its_own_share_without_duplicates(index, clients, settings):
    p = pipeline(index, clients, settings, union_report=4, union_per_concept=2, union_max=40)
    hits = candidates(p, [REPORT, *CONCEPTS])
    sources = Counter(h.source for h in hits)
    assert sources["concept"] <= 2 * len(CONCEPTS)
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_the_cap_holds_and_without_concepts_only_the_report_remains(index, clients, settings):
    capped = candidates(pipeline(index, clients, settings, union_report=4, union_per_concept=10, union_max=7),
                        [REPORT, *CONCEPTS])
    assert len(capped) <= 7
    alone = candidates(pipeline(index, clients, settings, union_report=5), [REPORT])
    assert {h.source for h in alone} == {"report"} and len(alone) <= 5


def test_advanced_runs_end_to_end_with_the_union_pool(index, clients, settings):
    assert pipeline(index, clients, settings, decompose=True).run(CASE).retrieved


def test_union_settings_are_checked():
    s = Settings.load(env={})
    assert (s.retrieval.union_report, s.retrieval.union_per_concept, s.retrieval.union_max) == (15, 12, 40)
    with pytest.raises(ConfigError, match="union"):
        Settings.load(env={}, overrides=["retrieval.union_max=10"])  # below union_report
    with pytest.raises(ConfigError, match="pool_mode"):
        Settings.load(env={}, overrides=["retrieval.pool_mode=all"])
