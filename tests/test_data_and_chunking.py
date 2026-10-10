"""The corpus contract, cleaning, environment extraction, chunking and exclusions."""
import json

import pytest

from supportdesk.data.chunking import Exclusions, FixedTokenChunker, build_chunks, chunk_text
from supportdesk.data.clean import clean_text
from supportdesk.data.corpus import CorpusError, load_corpus, write_corpus
from supportdesk.data.environment import extract_environment, missing_environment
from tests.helpers import ROOT

REAL = ROOT / "data" / "processed"


def row(sid="issue:1", kind="issue", text="Title: T\n\nbody", **meta):
    return {"source_id": sid, "source_type": kind, "text": text, "metadata": meta}


def test_sample_corpus_loads_with_counts(corpus):
    assert corpus.manifest()["by_type"] == {"issue": 6, "comment": 6, "documentation": 3, "release_section": 2}
    assert len(corpus.sha256) == 64


@pytest.mark.skipif(not (REAL / "corpus.jsonl").exists(), reason="real corpus not present")
def test_the_real_corpus_loads_and_every_record_maps():
    real = load_corpus(REAL)
    assert real.manifest()["by_type"] == {"issue": 126, "comment": 214, "documentation": 54,
                                          "release_section": 111, "linked_reference": 35}
    assert real.notes["linked_reference_dated_by_retrieved_at"] == 10
    assert all(r.url and r.title for r in real.records)
    assert all(r.as_of is None for r in real.records if r.source_type == "documentation")
    assert all(r.as_of for r in real.records if r.source_type != "documentation")


def test_missing_corpus_says_where_and_what_to_do(tmp_path):
    with pytest.raises(CorpusError) as info:
        load_corpus(tmp_path / "processed")
    assert "corpus.jsonl" in str(info.value) and "configs/sample.toml" in str(info.value)


@pytest.mark.parametrize("bad, message", [
    ({"source_type": "issue", "text": "x", "metadata": {}}, "'source_id' must be a str"),
    (row(kind="tweet"), "unknown source_type"),
    (row(text="   "), "empty text"),
])
def test_records_that_break_the_contract_are_refused(tmp_path, bad, message):
    write_corpus(tmp_path, [bad])
    with pytest.raises(CorpusError, match=message):
        load_corpus(tmp_path)


def test_edited_text_duplicates_and_bad_json_are_refused(tmp_path):
    path = write_corpus(tmp_path, [row()])
    edited = json.loads(path.read_text())
    edited["text"] += " (edited)"
    path.write_text(json.dumps(edited) + "\n")
    with pytest.raises(CorpusError, match="text_sha256"):
        load_corpus(tmp_path)
    write_corpus(tmp_path, [row(), row()])
    with pytest.raises(CorpusError, match="duplicate source_id"):
        load_corpus(tmp_path)
    path.write_text("{not json\n")
    with pytest.raises(CorpusError, match="not JSON"):
        load_corpus(tmp_path)


def test_each_type_maps_to_one_shape(tmp_path):
    write_corpus(tmp_path, [
        row("issue:7", number=7, html_url="https://x/7", title="Bug", created_at="2024-01-01T00:00:00Z",
            state="closed", streamlit_version={"value": "1.30.0", "evidence_text": "1.30.0"}),
        row("comment:1", "comment", "Issue context: Bug\n\nfix", parent_source_id="issue:7", issue_number=7,
            html_url="https://x/7#c1", created_at="2024-01-02T00:00:00Z"),
        row("release:1.31.0:section:1", "release_section", "Streamlit 1.31.0\n- fix", version="1.31.0",
            release_date="2024-02-01", html_url="https://x/r"),
        row("linked:abc", "linked_reference", "Forum post", title="Post", html_url="https://f/1",
            retrieved_at="2026-10-08T00:00:00Z", linked_from_source_ids=["issue:7"]),
        row("doc:1", "documentation", "Doc\n\ntext", title="Doc", html_url="https://d", commit="abc123",
            embedding_allowed=True),
        row("doc:hidden", "documentation", "x", title="Hidden", html_url="https://d2", embedding_allowed=False),
    ])
    c = load_corpus(tmp_path)
    by = {r.source_id: r for r in c.records}
    assert "doc:hidden" not in by and c.notes["not_embeddable_skipped"] == 1
    assert by["issue:7"].version == "1.30.0" and by["issue:7"].issue_number == 7
    assert by["issue:7"].meta == {"state": "closed"}
    assert by["comment:1"].title == "Comment on issue #7" and by["comment:1"].parent_source_id == "issue:7"
    assert by["release:1.31.0:section:1"].title == "Release 1.31.0" and by["release:1.31.0:section:1"].as_of == "2024-02-01"
    assert by["linked:abc"].as_of == "2026-10-08T00:00:00Z"  # no date of its own: hidden from older cases
    assert by["linked:abc"].linked_issue_numbers == frozenset({7})
    assert by["doc:1"].as_of is None and by["doc:1"].version == "abc123"


def test_chunks_are_anchored_spans_and_respect_limits():
    from supportdesk.tokenizer import count_tokens, tokenize_with_spans
    text = "  " + " ".join(f"word{i}" for i in range(1000)) + "\n"
    pieces = chunk_text(text, max_tokens=100, overlap=20)
    assert all(text[a:b] == piece for piece, a, b in pieces)  # character offsets into the text
    assert all(count_tokens(piece) <= 100 for piece, _, _ in pieces)
    spans = tokenize_with_spans(text)
    assert pieces[1][1] == spans[80][1]  # the second window starts 20 tokens before the first ends
    assert pieces[-1][2] == spans[-1][2]
    with pytest.raises(ValueError):
        chunk_text(text, max_tokens=10, overlap=10)


def test_baseline_chunker_windows_are_512_with_64_overlap(corpus):
    from supportdesk.tokenizer import count_tokens
    long = next(r for r in corpus.records if r.source_type == "issue")
    long = long.__class__(**{**long.__dict__, "text": " ".join(f"w{i}" for i in range(1200))})
    pieces = FixedTokenChunker(max_tokens=512, overlap_tokens=64)(long)
    assert [count_tokens(p.text) for p in pieces] == [512, 512, 304]  # starts at 0, 448, 896
    assert [p.chunk_id for p in pieces] == [f"{long.source_id}:{i:03d}" for i in range(3)]


def test_clean_text_masks_pii_and_secrets_but_keeps_code():
    raw = "mail me at jane.doe@example.com <!-- template -->\nkey sk-abcdefghijklmnop1234\n```python\nst.write(1)\n```"
    out = clean_text(raw)
    assert "jane.doe" not in out and "<email>" in out
    assert "sk-abcdef" not in out and "template" not in out
    assert "st.write(1)" in out


def test_environment_extraction_never_guesses():
    env = extract_environment("Streamlit: 1.32.0, Python: 3.11. Using Chrome on Ubuntu.")
    assert env == {"streamlit_version": "1.32.0", "python_version": "3.11",
                   "operating_system": "Ubuntu", "browser": "Chrome"}
    assert extract_environment("my app is slow") == {}
    assert missing_environment({"streamlit_version": "1.32.0"}) == [
        "python_version", "operating_system", "browser"]


def test_exclusions_drop_an_issue_its_comments_and_pages_linked_from_it(tmp_path):
    write_corpus(tmp_path, [
        row("issue:7", number=7, html_url="u", title="Bug", created_at="2024-01-01T00:00:00Z"),
        row("comment:1", "comment", "c", parent_source_id="issue:7", issue_number=7, html_url="u",
            created_at="2024-01-02T00:00:00Z"),
        row("linked:abc", "linked_reference", "post", title="Post", html_url="u", created_at="2024-01-03T00:00:00Z",
            linked_from_source_ids=["issue:7"]),
        row("issue:8", number=8, html_url="u", title="Other", created_at="2024-01-01T00:00:00Z"),
    ])
    records = load_corpus(tmp_path).records
    chunker = FixedTokenChunker(max_tokens=512, overlap_tokens=64)
    kept = {c.source_id for c in build_chunks(records, Exclusions(issue_numbers=frozenset({7})), chunker=chunker)}
    assert kept == {"issue:8"}


def test_chunk_metadata_carries_provenance(chunks):
    issue = next(c for c in chunks if c.source_id == "fixture-issue-9001")
    assert issue.url.endswith("/issues/9001") and issue.version == "1.32.0"
    assert issue.meta["state"] == "closed" and issue.as_of == "2024-03-10T09:00:00Z"
    doc = next(c for c in chunks if c.source_type == "documentation")
    assert doc.as_of is None and doc.version == "fixture-commit-0001"


@pytest.mark.skipif(not (REAL / "corpus.jsonl").exists(), reason="real corpus not present")
def test_every_chunk_of_the_real_corpus_meets_the_chunk_contract():
    from supportdesk.data.chunking import check_chunks
    real = load_corpus(REAL)
    chunks = build_chunks(real.records, chunker=FixedTokenChunker(max_tokens=512, overlap_tokens=64))
    check_chunks(real.records, chunks)  # raises on any violation
    by_id = {r.source_id: r for r in real.records}
    assert all(by_id[c.source_id].text[c.start_char:c.end_char] == c.text for c in chunks)
    assert {c.source_id for c in chunks} == set(by_id)  # every source is searchable


def test_the_contract_check_rejects_a_chunk_that_is_not_an_exact_span(corpus):
    from dataclasses import replace

    from supportdesk.data.chunking import ChunkContractError, check_chunks
    good = FixedTokenChunker(max_tokens=512, overlap_tokens=64)(corpus.records[0])
    check_chunks(corpus.records, good)
    for bad, problem in ((replace(good[0], text=good[0].text.upper()), "not source.text"),
                         (replace(good[0], start_char=None), "missing offsets"),
                         (replace(good[0], source_text_sha256="0" * 64), "hash differs")):
        with pytest.raises(ChunkContractError, match=problem):
            check_chunks(corpus.records, [bad])


def test_a_corpus_with_personal_data_is_refused_not_edited(tmp_path, clients, settings):
    from dataclasses import replace as dc_replace

    from supportdesk.indexing import run_indexing
    write_corpus(tmp_path, [row(number=1, html_url="u", title="T", created_at="2024-01-01T00:00:00Z",
                                text="Title: T\n\nmail me at jane@example.com")])
    s = dc_replace(settings, paths=dc_replace(settings.paths, data_dir=tmp_path))
    state = run_indexing(s, clients, cases=[], exclusions=Exclusions())
    assert state["sensitive"] == [("issue:1", ["email"])]
    assert "chunks" not in state and "index" not in state  # stopped before chunking and embedding
    assert clients.ledger.total_cost_usd == 0
