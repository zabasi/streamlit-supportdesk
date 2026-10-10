"""The single-change experiments: clean_query (the query) and doc_slots (the retriever)."""
import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.clean_query import CleanQuery, clean_report
from supportdesk.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    DocSlotsRetriever,
    SplitHybridRetriever,
)
from supportdesk.types import CaseInput

REPORT = """### Checklist

- [X] I have searched the existing issues for similar issues.

### Summary

The text field is empty after pressing the button. See https://example.com/x

### Reproducible Code Example

```Python
import streamlit as st
from time import sleep
with st.expander("Inside"):
    st.text_area("Text", key="text")
raise StreamlitAPIException("boom")
```

### Debug info

- Streamlit version: 1.13.0
- Python version: 3.10.5

### Additional Information

<img width="799" src="https://user-images.githubusercontent.com/x.png">
Happens only with the expander.

---

Community voting on feature requests enables the Streamlit team to understand which features matter.
"""


def test_the_template_goes_and_the_meaning_stays():
    out = clean_report(REPORT)
    for kept in ("The text field is empty after pressing the button.", 'with st.expander("Inside"):',
                 'st.text_area("Text", key="text")', 'StreamlitAPIException("boom")',
                 "Happens only with the expander."):
        assert kept in out
    for dropped in ("Checklist", "searched the existing issues", "Streamlit version", "Python version",
                    "https://", "<img", "Community voting", "from time import sleep", "###", "---"):
        assert dropped not in out
    assert clean_report("Plain text with no template.") == "Plain text with no template."


def test_the_query_has_the_title_and_the_reporters_follow_ups_only():
    case = CaseInput(case_id="c", title="Expander loses state", body=REPORT,
                     history=[{"role": "assistant", "body": "Which version?"},
                              {"role": "user", "body": "Also on 1.32.0, see https://example.com"}])
    (query,) = CleanQuery(4000)(case, {})
    assert query.startswith("Expander loses state\n") and "Also on 1.32.0, see" in query
    assert "Which version?" not in query and "https://" not in query
    assert len(CleanQuery(50)(case, {})[0]) == 50


def test_the_index_can_search_one_source_type(index):
    probe = [1.0] + [0.0] * (index.dimension - 1)
    found = index.search(probe, limit=50, as_of=None, source_types=("documentation",))
    assert found and {index.chunks[i].source_type for i, _ in found} == {"documentation"}
    assert len(index.search(probe, limit=50, as_of=None)) > len(found)


def test_doc_slots_keep_places_for_documentation_in_relevance_order(index, clients):
    query = "st.session_state is empty after refresh"
    plain = DenseRetriever(index, clients.embedding).search(query, k=6)
    for slots in (1, 3):
        hits = DocSlotsRetriever(index, clients.embedding, doc_slots=slots).search(query, k=6)
        assert len(hits) == 6 and [h.rank for h in hits] == list(range(1, 7))
        assert sum(h.chunk.source_type == "documentation" for h in hits) >= slots
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    none = DocSlotsRetriever(index, clients.embedding, doc_slots=0).search(query, k=6)
    assert [h.chunk.chunk_id for h in none] == [h.chunk.chunk_id for h in plain]  # 0 slots = plain dense


def test_each_experiment_differs_from_the_baseline_in_exactly_one_way():
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    base = flat("baseline")
    for name, expected in (("clean_query", {"pipeline.name"}),
                           ("doc_slots", {"pipeline.name", "pipeline.retriever", "retrieval.doc_slots"}),
                           ("hybrid_rrf", {"pipeline.name", "pipeline.retriever"}),
                           ("hybrid_split", {"pipeline.name", "pipeline.retriever", "retrieval.bm25_slots"}),
                           ("rerank_llm", {"pipeline.name", "retrieval.candidate_k"}),
                           ("rewrite_llm", {"pipeline.name"}),
                           ("split_llm", {"pipeline.name", "retrieval.max_queries"}),
                           ("split_sentences", {"pipeline.name", "retrieval.max_queries"})):
        other = flat(name)
        assert {k for k in base if base[k] != other[k]} == expected, name


def test_doc_slots_cannot_exceed_top_k():
    with pytest.raises(ConfigError, match="retrieval.doc_slots"):
        Settings.load(env={}, overrides=["retrieval.doc_slots=7"])


def split(index, clients, slots):
    return SplitHybridRetriever(index, clients.embedding, bm25_slots=slots, bm25_k1=1.5, bm25_b=0.75)


def test_split_hybrid_takes_a_fixed_share_from_each_without_merging_scores(index, clients):
    query = "session state reset rerun page"  # BM25 and dense disagree on this one
    dense = DenseRetriever(index, clients.embedding).search(query, k=6)
    bm25 = BM25Retriever(index.chunks, k1=1.5, b=0.75).search(query, k=6)
    hits = split(index, clients, 3).search(query, k=6)
    ids = [h.chunk.chunk_id for h in hits]
    assert len(hits) == 6 and len(set(ids)) == 6 and [h.rank for h in hits] == list(range(1, 7))
    assert [h.source for h in hits] == ["dense", "bm25"] * 3  # the two lists take turns
    assert [h.chunk.chunk_id for h in hits if h.source == "dense"] == [h.chunk.chunk_id for h in dense[:3]]
    top_dense = {h.chunk.chunk_id for h in dense[:3]}
    expected_bm25 = [h.chunk.chunk_id for h in bm25 if h.chunk.chunk_id not in top_dense][:3]
    assert [h.chunk.chunk_id for h in hits if h.source == "bm25"] == expected_bm25  # skips dense's picks


def test_split_hybrid_never_repeats_a_chunk_dense_already_took(index, clients):
    # On the sample, BM25's only matches for this query are dense's own top 3: BM25 has nothing new
    # to add, so dense fills the remaining places instead of a chunk being shown twice.
    hits = split(index, clients, 3).search("st.session_state is empty after refresh", k=6)
    assert len({h.chunk.chunk_id for h in hits}) == 6 and {h.source for h in hits} == {"dense"}


def test_split_hybrid_fills_from_dense_when_bm25_finds_nothing(index, clients):
    hits = split(index, clients, 3).search("zzqx wvvq", k=6)  # no indexed word matches
    assert len(hits) == 6 and {h.source for h in hits} == {"dense"}


def test_split_hybrid_with_no_bm25_slots_is_plain_dense(index, clients):
    query = "st.session_state is empty after refresh"
    plain = [h.chunk.chunk_id for h in DenseRetriever(index, clients.embedding).search(query, k=6)]
    assert [h.chunk.chunk_id for h in split(index, clients, 0).search(query, k=6)] == plain


def test_bm25_slots_cannot_exceed_top_k():
    with pytest.raises(ConfigError, match="retrieval.bm25_slots"):
        Settings.load(env={}, overrides=["retrieval.bm25_slots=7"])


def scripted(clients, text):
    clients.chat.script[clients.chat.calls + 1] = text


def test_the_rewriter_returns_the_models_queries_or_falls_back_to_the_clean_report(clients, settings):
    import dataclasses
    import json

    from supportdesk.pipelines.stages.rewrite_query import QueryRewriter
    three = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, max_queries=3))
    rewriter = QueryRewriter(clients, three)
    case = CaseInput(case_id="c", title="Expander loses state", body=REPORT)
    scripted(clients, json.dumps({"queries": ["st.text_area value lost in st.expander", "session state widget key",
                                              "session state widget key", "fourth one", "fifth"]}))
    assert rewriter(case, {}) == ["st.text_area value lost in st.expander", "session state widget key", "fourth one"]
    system = clients.chat.seen[-1][0]["content"]
    assert "at most 3 queries" in system and "{max_queries}" not in system
    scripted(clients, "not json at all")
    assert rewriter(case, {}) == CleanQuery(three.context.max_query_chars)(case, {})  # unusable: the clean report


def test_the_sentence_splitter_gives_title_code_then_sentences():
    from supportdesk.pipelines.stages.split_query import SentenceSplitQuery
    case = CaseInput(case_id="c", title="Expander loses state", body=REPORT)
    queries = SentenceSplitQuery(4000, 4)(case, {})
    assert queries[0] == "Expander loses state"
    assert "st.expander" in queries[1] and "st.text_area" in queries[1]
    assert queries[2] == "The text field is empty after pressing the button."
    assert len(queries) <= 4 and len(set(queries)) == len(queries)
    assert len(SentenceSplitQuery(4000, 1)(case, {})) == 1


def test_the_reranker_applies_the_models_order_and_keeps_the_original_when_unusable(index, clients, settings):
    import json

    from supportdesk.pipelines.stages.llm_rerank import LLMReranker
    case = CaseInput(case_id="c", title="Session state lost", body="st.session_state empty after F5")
    hits = DenseRetriever(index, clients.embedding).search("session state lost", k=6)
    reranker = LLMReranker(clients, settings)
    scripted(clients, json.dumps({"ranking": ["C3", "C1", "C9", "C3", "x"]}))  # invalid and repeated labels ignored
    ranked = reranker(case, "q", hits)
    assert [h.chunk.chunk_id for h in ranked] == [hits[2].chunk.chunk_id, hits[0].chunk.chunk_id] + [
        h.chunk.chunk_id for i, h in enumerate(hits) if i not in (0, 2)]
    assert [h.rank for h in ranked] == list(range(1, 7))
    assert "C6:" in clients.chat.seen[-1][1]["content"]  # every candidate is shown to the model
    scripted(clients, '{"ranking": []}')
    assert reranker(case, "q", hits) == hits


def test_rerank_and_split_pipelines_run_end_to_end(index, clients, settings):
    import dataclasses
    import json

    from supportdesk.pipelines.experiments import build_rerank_llm, build_split_llm
    answer = json.dumps({"outcome": "request_info", "user_reply": "Which version?", "citations": []})
    wide = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, candidate_k=12))
    rerank = build_rerank_llm(DenseRetriever(index, clients.embedding), clients, wide)
    case = CaseInput(case_id="c", title="Session state lost", body="st.session_state empty after F5")
    scripted(clients, json.dumps({"ranking": ["C12", "C11"]}))
    clients.chat.script[clients.chat.calls + 2] = answer
    result = rerank.run(case)
    assert result.outcome == "request_info" and len(result.retrieved) == 6
    assert result.trace["stages"]["reranker"] == "LLMReranker"
    three = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, max_queries=3))
    split = build_split_llm(DenseRetriever(index, clients.embedding), clients, three)
    scripted(clients, json.dumps({"queries": ["session state reset", "st.session_state refresh", "F5 loses values"]}))
    clients.chat.script[clients.chat.calls + 2] = answer
    result = split.run(case)
    assert result.trace["n_queries"] == 3 and result.outcome == "request_info"
