"""Search with fixed chunks, give the model whole sections."""
import json

from supportdesk.eval.metrics import gold_share
from supportdesk.pipelines.stages.sections import Sections, page_texts, section_spans
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

PAGE = """Widget behavior

Intro.

## Order of operations

Callbacks run first.

### Example

```python
# not a heading
st.button("Go")
```

## Widget life cycle

Widgets are created and removed.
"""


def chunk(a, b, kind="documentation", source="doc:w"):
    return Chunk(chunk_id=f"{source}:{a}", source_id=source, source_type=kind, title="Widget behavior",
                 text=PAGE[a:b], url="u", start_char=a, end_char=b)


def test_pages_rebuild_from_overlapping_chunks_and_a_gap_leaves_the_page_out():
    third = len(PAGE) // 3
    parts = [chunk(0, 2 * third), chunk(third, len(PAGE))]
    assert page_texts(parts) == {"doc:w": PAGE}
    assert page_texts([chunk(0, third), chunk(2 * third, len(PAGE))]) == {}  # a gap: cannot rebuild


def test_sections_keep_subsections_with_their_section_and_ignore_fences():
    spans = section_spans(PAGE)
    starts = [PAGE[a:b].splitlines()[0] for a, b in spans]
    assert starts == ["Widget behavior", "## Order of operations", "## Widget life cycle"]
    order = PAGE[spans[1][0]:spans[1][1]]
    assert "### Example" in order and "# not a heading" in order  # subsection and fenced comment stay inside


def test_a_chunk_becomes_the_whole_sections_it_overlaps_with_exact_offsets():
    sections = Sections([chunk(0, len(PAGE))])
    start, end = PAGE.index("Intro."), PAGE.index("Widgets are created")  # a chunk across two sections
    got = sections.of(Hit(chunk(start, end), 0.9, 1, "dense"))
    order = next(g for g in got if g.chunk.text.startswith("## Order of operations"))
    assert "### Example" in order.chunk.text  # the whole section, subsection included
    assert all(PAGE[g.chunk.start_char:g.chunk.end_char] == g.chunk.text for g in got)
    issue = Hit(chunk(0, 10, kind="issue", source="issue:1"), 0.5, 2, "dense")
    assert sections.of(issue) == [issue]  # other sources stay as they are


def test_the_trimmer_keeps_the_section_that_best_matches_the_report(clients, settings):
    import dataclasses

    from supportdesk.pipelines.stages.sections import SectionTrimmer
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, top_k=1, doc_slots=0))
    trimmer = SectionTrimmer(clients, s, [chunk(0, len(PAGE))])
    whole = Hit(chunk(0, len(PAGE)), 0.9, 1, "dense")
    other = Hit(Chunk(chunk_id="doc:x:0", source_id="doc:x", source_type="documentation", title="t",
                      text="Unrelated text about caching.", url="u", start_char=0, end_char=29), 0.1, 2, "dense")
    case = CaseInput(case_id="c", title="Widgets are removed", body="My widget life cycle: created and removed")
    clients.chat.script[clients.chat.calls + 1] = json.dumps({"ranking": ["C1", "C2"]})
    (kept,) = trimmer(case, "q", [whole, other])
    assert kept.chunk.text.startswith("## Widget life cycle")


def test_gold_share_is_the_gold_part_of_the_evidence():
    gold = {"source_id": "doc:w", "start_char": 10, "end_char": 30}
    evidence = [{"source_id": "doc:w", "start_char": 0, "end_char": 40},
                {"source_id": "doc:w", "start_char": 20, "end_char": 40},   # overlaps: counted once
                {"source_id": "issue:1", "start_char": 0, "end_char": 60}]
    assert gold_share(evidence, [gold]) == 20 / 100
    assert gold_share([{"source_id": "doc:w", "start_char": 10, "end_char": 30}], [gold]) == 1.0


def test_the_section_experiments_change_only_their_stage_and_room():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    base = flat(d)
    assert {k for k in base if base[k] != flat(f"{d}_trimmed")[k]} == {"pipeline.name"}
    sections = flat(f"{d}_sections")
    assert {k for k in base if base[k] != sections[k]} == {"pipeline.name", "retrieval.top_k"}
    assert sections["retrieval.top_k"] == 8 and sections["context.max_tokens"] == base["context.max_tokens"]


def test_a_section_bigger_than_its_chunk_is_cut_to_the_chunk():
    sections = Sections([chunk(0, len(PAGE))])
    start = PAGE.index("Callbacks run first")
    small = Hit(chunk(start, start + 15), 0.9, 1, "dense")  # a chunk smaller than its section
    (got,) = sections.of(small)
    assert (got.chunk.start_char, got.chunk.end_char) == (start, start + 15)
    assert got.chunk.text == PAGE[start:start + 15]


def test_retrieval_runs_report_the_evidence_size_and_gold_share(tmp_path, capsys):
    from supportdesk.cli import main
    from supportdesk.eval.report import render_report
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]
    assert main(["index", *base]) == 0
    assert main(["eval-retrieval", *base, "--run-id", "r"]) == 0
    capsys.readouterr()
    run = tmp_path / "art" / "runs" / "r"
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert all(r["packet_tokens"] > 0 for r in rows)
    metrics = json.loads((run / "summary.json").read_text())["metrics"]
    assert metrics["packet_tokens_median"] > 0 and metrics["packet_chunks_mean"] > 0
    assert "Evidence tokens in the prompt (median)" in render_report([run])
