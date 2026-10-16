"""The LangChain rule chunker keeps every character of a source and exact offsets."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langchain_text_splitters")

from supportdesk.config import Settings
from supportdesk.data.chunking import build_chunks, check_chunks, make_chunker
from supportdesk.data.corpus import load_corpus
from supportdesk.data.langchain_chunkers import rule_spans

ROOT = Path(__file__).resolve().parents[1]
PAGE = ("Session State\n\n# Session State\n\nSession State shares variables between reruns.\n\n"
        "### Initialize values in Session State\n\nThe API follows a field-based API:\n\n"
        "```python\n# a comment, not a heading\nif 'key' not in st.session_state:\n    st.session_state.key = 'value'\n```\n\n"
        "### Delete items\n\n" + "Delete with del. " * 150 + "\n")


def covered(text, spans):
    have = set()
    for _, a, b, _ in spans:
        have.update(range(a, b))
    return all(i in have for i, ch in enumerate(text) if not ch.isspace())


def test_every_character_is_kept_with_exact_offsets():
    spans = rule_spans(PAGE, "documentation", chunk_size=400, overlap=50)
    assert covered(PAGE, spans)  # LangChain rewrites whitespace; nothing may be lost because of it
    assert all(PAGE[a:b] == piece for piece, a, b, _ in spans)
    starts = {a for _, a, _, _ in spans}
    assert PAGE.index("### Initialize values") in starts and PAGE.index("### Delete items") in starts


def test_the_real_documentation_is_fully_covered():
    records = [r for r in load_corpus(ROOT / "data/processed").records if r.source_type == "documentation"]
    assert records and all(covered(r.text, rule_spans(r.text, r.source_type)) for r in records)


def test_langchain_rule_is_a_chunking_method():
    s = Settings.load(env={}, overrides=["chunking.method=langchain_rule"])
    records = load_corpus(ROOT / "data/processed").records[:40]
    chunks = build_chunks(records, chunker=make_chunker(s))
    check_chunks(records, chunks)  # every chunk is exactly its source slice
    assert chunks and s.chunking.extras() == {"langchain_chunk_chars": 1600, "langchain_overlap_chars": 200}


def test_the_experiment_reads_the_repository_gold_by_default():
    spec = importlib.util.spec_from_file_location("hier_class_rag", ROOT / "scripts" / "hier_class_rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module._cases(None, ROOT / "data/eval")
    assert len(cases) == 15 and all(c.evidence_spans for c in cases)


def test_chunks_tile_every_page_so_section_completion_works():
    """LangChain drops the whitespace it cuts at; the gaps are closed so each page can be rebuilt from
    its chunks (section completion needs that)."""
    from supportdesk.pipelines.stages.sections import page_texts

    s = Settings.load(env={}, overrides=["chunking.method=langchain_rule"])
    records = [r for r in load_corpus(ROOT / "data/processed").records if r.source_type == "documentation"]
    chunks = build_chunks(records, chunker=make_chunker(s))
    pages = page_texts(chunks)
    assert set(pages) == {r.source_id for r in records}
    assert all(pages[r.source_id] == r.text[: len(pages[r.source_id])] for r in records)
