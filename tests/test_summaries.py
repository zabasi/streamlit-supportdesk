"""Hierarchical retrieval through section summaries."""
import dataclasses
import json

import pytest

from supportdesk.pipelines.stages.evidence import SummaryReranker
from supportdesk.retrieval.index import IndexUnusable
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

PAGE = ("Widgets\n\nIntro.\n\n## Order of operations\n\nCallbacks run before the script reruns.\n\n"
        "## Widget life cycle\n\nWidgets are created, kept and cleaned up.\n\n## Forms\n\nForms batch input.\n")


def chunk(n, a, b, as_of=None):
    return Chunk(chunk_id=f"doc:w:{n}", source_id="doc:w", source_type="documentation", title="Widgets",
                 text=PAGE[a:b], url="u", start_char=a, end_char=b, as_of=as_of)


def summaries(tmp_path):
    rows = []
    for heading, summary in (("Order of operations", "the order in which callbacks and the rerun execute"),
                             ("Widget life cycle", "how widgets are created and cleaned up between reruns"),
                             ("Forms", "forms batch widget input until submission")):
        a = PAGE.index(f"## {heading}")
        b = PAGE.find("\n## ", a + 3) + 1 or len(PAGE)
        rows.append({"source_id": "doc:w", "start_char": a, "end_char": b, "title": "Widgets", "heading": heading,
                     "summary": summary})
    path = tmp_path / "sections.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def stage(clients, settings, tmp_path, n=1, chunks=None):
    s = dataclasses.replace(settings,
                            retrieval=dataclasses.replace(settings.retrieval, summary_sections=n, top_k=6, doc_slots=0),
                            paths=dataclasses.replace(settings.paths, summaries=summaries(tmp_path)))
    third = len(PAGE) // 3
    chunks = chunks or [chunk(0, 0, third), chunk(1, third - 10, 2 * third), chunk(2, 2 * third - 10, len(PAGE))]
    return SummaryReranker(clients, s, chunks), chunks


def test_the_best_matching_sections_chunks_are_added_and_marked(clients, settings, tmp_path):
    reranker, _ = stage(clients, settings, tmp_path, n=1)
    case = CaseInput(case_id="c", title="callbacks run before the rerun", body="order of callbacks and the rerun")
    pool = []
    reranker.inner = lambda case, query, hits: pool.extend(hits) or hits
    reranker(case, "q", [])
    order = (PAGE.index("## Order of operations"), PAGE.index("## Widget life cycle"))
    added = [h for h in pool if h.source == "summary"]
    assert added and all(h.chunk.start_char < order[1] and h.chunk.end_char > order[0] for h in added)
    assert [h.rank for h in pool] == list(range(1, len(pool) + 1))


def test_existing_candidates_dates_and_zero_sections(clients, settings, tmp_path):
    reranker, chunks = stage(clients, settings, tmp_path, n=3)
    seen = []
    reranker.inner = lambda case, query, hits: seen.append(list(hits)) or hits
    found = [Hit(chunks[1], 0.9, 1, "dense")]
    reranker(CaseInput(case_id="c", title="widgets", body="widgets"), "q", found)
    ids = [h.chunk.chunk_id for h in seen[-1]]
    assert ids[0] == "doc:w:1" and len(ids) == len(set(ids))  # the found chunk is not added twice
    later = [chunk(0, 0, 60, as_of="2030-01-01T00:00:00Z"), chunk(1, 50, len(PAGE))]
    reranker, _ = stage(clients, settings, tmp_path, n=3, chunks=later)
    reranker.inner = lambda case, query, hits: seen.append(list(hits)) or hits
    reranker(CaseInput(case_id="c", title="widgets", body="widgets", created_at="2024-01-01T00:00:00Z"), "q", [])
    assert "doc:w:0" not in [h.chunk.chunk_id for h in seen[-1]]  # not visible at the case's date
    off, _ = stage(clients, settings, tmp_path, n=0)
    off.inner = lambda case, query, hits: seen.append(list(hits)) or hits
    off(CaseInput(case_id="c", title="widgets", body="widgets"), "q", found)
    assert seen[-1] == found


def test_missing_summaries_are_reported_like_an_unbuilt_index(clients, settings, tmp_path):
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, summary_sections=2),
                            paths=dataclasses.replace(settings.paths, summaries=tmp_path / "none.jsonl"))
    with pytest.raises(IndexUnusable, match="summarize_sections.py"):
        SummaryReranker(clients, s, [chunk(0, 0, 20)])


def test_the_script_summarizes_every_section_offline(tmp_path):
    from scripts.summarize_sections import main
    from tests.helpers import SAMPLE
    out = tmp_path / "sections.jsonl"
    assert main(["--mode", "offline", "--set", f"paths.data_dir={SAMPLE}", "--set", f"paths.summaries={out}",
                 "--set", f"paths.cache_path={tmp_path / 'cache.sqlite'}",
                 "--set", f"paths.ledger_path={tmp_path / 'ledger.jsonl'}"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all(r["summary"] and r["end_char"] > r["start_char"] for r in rows)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["sections"] == len(rows) and manifest["prompt_sha256"]


def test_the_summaries_experiment_changes_only_the_summary_level():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    a = flat("candidate-1-dense-rerank-section")  # the earlier advanced, which this variant was built on
    b = flat("clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts_summaries")
    assert {k for k in a if a[k] != b[k]} == {"pipeline.name", "pipeline.scope_gate", "retrieval.decompose",
                                                "retrieval.complete", "retrieval.summary_sections",
                                                "retrieval.top_k", "retrieval.rerank_keep_docs",
                                                "context.max_tokens"}
    assert b["retrieval.summary_sections"] == 3
