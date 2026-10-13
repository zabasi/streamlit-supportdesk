"""Hierarchical retrieval: section summaries first, then the chunks inside the best sections."""
import json

import pytest

from scripts.summarize_sections import main as summarize
from supportdesk.pipelines.hierarchical import build_advanced_hier, build_hier_baseline
from supportdesk.retrieval.hierarchy import SOURCE, HierarchicalRetriever, SectionSummaries
from supportdesk.retrieval.index import IndexUnusable
from supportdesk.retrieval.retriever import DenseRetriever, DocSlotsRetriever
from supportdesk.types import CaseInput
from tests.helpers import SAMPLE, override

CASE = CaseInput(case_id="c", title="Session state resets", body="The value is gone after a rerun.")


@pytest.fixture()
def summaries(tmp_path, settings):
    out = tmp_path / "sections.jsonl"
    assert summarize(["--mode", "offline", "--set", f"paths.data_dir={SAMPLE}", "--set", f"paths.summaries={out}",
                      "--set", f"paths.cache_path={tmp_path / 'summary_cache.sqlite'}",
                      "--set", f"paths.ledger_path={tmp_path / 'summary_ledger.jsonl'}"]) == 0
    return out


def hier(index, clients, summaries, *, sections=2, places=3, base=None):
    return HierarchicalRetriever(base or DenseRetriever(index, clients.embedding), index, clients.embedding,
                                 summaries, sections=sections, places=places)


def test_missing_summaries_are_reported_like_an_unbuilt_index(index, clients, tmp_path):
    with pytest.raises(IndexUnusable, match="summarize_sections.py"):
        hier(index, clients, tmp_path / "none.jsonl")


def test_chunks_inside_the_best_sections_fill_the_places_and_are_marked(index, clients, summaries):
    retriever = hier(index, clients, summaries, sections=2, places=3)
    hits = retriever.search("session state is lost after a rerun", k=6)
    assert len(hits) == 6 and [h.rank for h in hits] == [1, 2, 3, 4, 5, 6]
    assert len({h.chunk.chunk_id for h in hits}) == 6  # nothing twice
    inside = [h for h in hits if h.source == SOURCE]
    assert 1 <= len(inside) <= 3
    rows = [json.loads(line) for line in summaries.read_text().splitlines()]
    vector = clients.embedding.embed(["session state is lost after a rerun"], stage="query")[0][0]
    best = retriever.level.best(vector, 2)
    assert len(best) == 2 and best[0]["score"] >= best[1]["score"] and best[0] in [{**r, "score": best[0]["score"]} for r in rows]
    for h in inside:
        assert h.chunk.source_type == "documentation"
        assert any(h.chunk.source_id == s["source_id"] and h.chunk.start_char < s["end_char"]
                   and h.chunk.end_char > s["start_char"] for s in best)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)  # best first, as usual


def test_without_places_it_is_the_base_search(index, clients, summaries):
    base = DenseRetriever(index, clients.embedding)
    off = hier(index, clients, summaries, places=0)
    query = "widgets reset when the page reruns"
    assert [h.chunk.chunk_id for h in off.search(query, k=6)] == [h.chunk.chunk_id for h in base.search(query, k=6)]


def test_chunks_dated_after_the_case_are_never_taken_from_a_section(index, clients, summaries):
    retriever = hier(index, clients, summaries, sections=5, places=6)
    visible = {h.chunk.chunk_id for h in retriever.search("session state", k=6, as_of="2000-01-01T00:00:00Z")}
    allowed = {c.chunk_id for c in index.chunks if c.as_of is None or c.as_of < "2000-01-01T00:00:00Z"}
    assert visible <= allowed


def test_section_summaries_are_embedded_once_on_first_use(index, clients, summaries):
    level = SectionSummaries(summaries, clients.embedding, {c.source_id for c in index.chunks})
    assert level._vectors is None
    vector = clients.embedding.embed(["callbacks"], stage="query")[0][0]
    level.best(vector, 1)
    first = level._vectors
    level.best(vector, 1)
    assert level._vectors is first and level.best(vector, 0) == []


def test_hier_baseline_runs_end_to_end(index, clients, settings, summaries):
    s = override(settings, paths={"summaries": str(summaries)})
    result = build_hier_baseline(hier(index, clients, summaries), clients, s).run(CASE)
    assert result.error is None and result.outcome != "error" and result.retrieved
    assert clients.chat.calls == 1  # no extra model call: the hierarchy costs embeddings only


def test_advanced_hier_uses_the_hierarchical_retriever_for_every_search(index, clients, settings, summaries):
    s = override(settings, retrieval={"candidate_k": 15, "doc_slots": 10, "max_queries": 3, "top_k": 6})
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=10)
    pipeline = build_advanced_hier(hier(index, clients, summaries, places=4, base=base), clients, s)
    result = pipeline.run(CASE)
    assert result.error is None and result.retrieved


def test_the_shipped_hierarchical_profiles_differ_from_their_bases_only_in_the_retriever():
    from supportdesk.config import Settings

    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    base, hier_baseline = flat("baseline"), flat("hier_baseline")
    assert {k for k in base if base[k] != hier_baseline[k]} == {"pipeline.name", "pipeline.retriever"}
    assert hier_baseline["pipeline.retriever"] == "hierarchical"
    adv, adv_hier = flat("advanced"), flat("advanced_hier")
    assert {k for k in adv if adv[k] != adv_hier[k]} == {"pipeline.name", "pipeline.retriever",
                                                          "retrieval.hier_places"}
    assert adv_hier["pipeline.retriever"] == "hierarchical_doc_slots"


def test_the_pipelines_are_registered_with_descriptions():
    from supportdesk.registry import PIPELINES, RETRIEVERS, pipeline_description
    for name in ("hier_baseline", "advanced_hier", "retrieve_loop_best", "retrieve_loop_mixed"):
        assert name in PIPELINES and pipeline_description(name)
    assert {"hierarchical", "hierarchical_doc_slots"} <= set(RETRIEVERS)
