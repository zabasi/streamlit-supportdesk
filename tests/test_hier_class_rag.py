"""Hierarchical-class RAG experiment: gold overlay, LangChain spans, class routing."""
from __future__ import annotations

import json

from supportdesk.data.chunking import check_chunks
from supportdesk.data.corpus import load_corpus
from supportdesk.data.langchain_chunkers import LangChainRuleChunker, rule_spans
from supportdesk.eval.annotated import gold_from_row, load_annotated_cases
from supportdesk.eval.cases import case_from_dict
from supportdesk.retrieval.class_routing import (
    ClassRoutedRetriever,
    centroid_routes,
    class_centroids,
    heuristic_label,
    kmeans_labels,
)
from tests.helpers import SAMPLE


def _annotated_row():
    return {
        "case_id": "DEV-1",
        "input": {"case_id": "DEV-1", "title": "widget key", "body": "st.session_state lost after expander"},
        "sources": [
            {"source": {"source_id": "doc:pinned:session_state.md", "source_type": "documentation",
                        "text": "hello", "metadata": {}, "text_sha256": "x"},
             "selected_spans": [
                 {"span_id": "s1", "start_char": 0, "end_char": 5, "text": "hello"},
             ]},
            {"source": {"source_id": "issue:99", "source_type": "issue",
                        "text": "other", "metadata": {}, "text_sha256": "y"},
             "selected_spans": [
                 {"span_id": "s2", "start_char": 0, "end_char": 5, "text": "other"},
             ]},
        ],
    }


def test_gold_from_annotated_uses_source_ids_and_spans():
    gold, spans = gold_from_row(_annotated_row())
    assert gold == ["doc:pinned:session_state.md", "issue:99"]
    assert spans[0]["source_id"] == "doc:pinned:session_state.md"
    assert spans[0]["start_char"] == 0 and spans[0]["text"] == "hello"


def test_annotated_case_overlays_official_gold(tmp_path):
    official = case_from_dict({
        "case_id": "DEV-1", "family_id": "family:1", "split": "dev", "category": "ambiguous",
        "input": {"title": "old", "body": "old body"}, "acceptable_outcomes": ["request_info"],
        "gold_evidence": ["doc:old"],
    })
    path = tmp_path / "ann.jsonl"
    path.write_text(json.dumps(_annotated_row()) + "\n", encoding="utf-8")
    cases = load_annotated_cases(path, official=[official])
    assert cases[0].category == "ambiguous"
    assert cases[0].gold_evidence == ["doc:pinned:session_state.md", "issue:99"]
    assert cases[0].evidence_spans[1]["source_id"] == "issue:99"


def test_case_from_dict_keeps_evidence_spans():
    case = case_from_dict({
        "case_id": "DEV-1", "family_id": "family:1", "split": "dev", "category": "answerable",
        "input": {"title": "t", "body": "b"}, "acceptable_outcomes": ["answer"],
        "gold_evidence": ["doc:x"],
        "evidence_spans": [{"source_id": "doc:x", "start_char": 0, "end_char": 2, "text": "ab"}],
    })
    assert case.evidence_spans[0]["end_char"] == 2


def test_rule_chunker_spans_are_exact_source_slices():
    text = "# Title\n\nSession State keeps widget values.\n\n## Caveats\n\nDo not write after instantiate.\n"
    spans = rule_spans(text, "documentation", chunk_size=80, overlap=10)
    assert spans and all(text[a:b] == piece for piece, a, b, _ in spans)
    corpus = load_corpus(SAMPLE)
    chunker = LangChainRuleChunker(chunk_size=240, overlap=40)
    chunks = [c for record in corpus.records for c in chunker(record)]
    check_chunks(corpus.records, chunks)
    assert chunks


def test_heuristic_and_kmeans_labels(index):
    assert heuristic_label("Please use st.session_state with a key") == "session_state"
    assert heuristic_label("st.data_editor key is read-only") == "data_editor"
    import numpy as np
    rng = np.random.default_rng(0)
    vectors = np.vstack([rng.normal(0, 0.05, (8, 6)) + [1, 0, 0, 0, 0, 0],
                         rng.normal(0, 0.05, (8, 6)) + [0, 1, 0, 0, 0, 0]])
    types = ["documentation"] * 8 + ["issue"] * 8
    labels, space = kmeans_labels(vectors, types, k=2, method="kmeans")
    assert set(labels) <= {"0", "1"} and len(labels) == 16
    cents = class_centroids(space, labels)
    routed = centroid_routes(space[0], cents, top=1)
    assert routed[0] == labels[0]


def test_class_router_filters_to_routed_class(index, clients):
    labels = ["keep" if c.source_type == "documentation" else "drop" for c in index.chunks]
    routed = ClassRoutedRetriever(index, clients.embedding, labels=labels, doc_slots=2, top=1,
                                  route="centroid")
    query = index.chunks[0].text[:80]
    # force the keep class by using a documentation chunk as the query
    docs = [c for c in index.chunks if c.source_type == "documentation"]
    hits = routed.search(docs[0].text[:120] if docs else query, k=4)
    assert hits
    assert all(labels[index.chunks.index(h.chunk)] == routed.last_route[0] for h in hits)
