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
                           ("rewrite_llm", {"pipeline.name"}),
                           ("split_llm", {"pipeline.name", "retrieval.max_queries"}),
                           ("split_sentences", {"pipeline.name", "retrieval.max_queries"}),
                           ("clean_doc_slots", {"pipeline.name", "pipeline.retriever", "retrieval.doc_slots"}),
                           ("clean_hybrid_split", {"pipeline.name", "pipeline.retriever", "retrieval.bm25_slots"}),
                           ("clean_hybrid_doc_slots", {"pipeline.name", "pipeline.retriever",
                                                       "retrieval.doc_slots"}),
                           ("clean_split_doc_slots", {"pipeline.name", "pipeline.retriever",
                                                      "retrieval.doc_slots", "retrieval.bm25_slots"})):
        other = flat(name)
        assert {k for k in base if base[k] != other[k]} == expected, name


def test_doc_slots_can_fill_the_candidates_but_not_more():
    with pytest.raises(ConfigError, match="retrieval.doc_slots"):
        Settings.load(env={}, overrides=["retrieval.doc_slots=7"])  # 6 candidates without candidate_k
    assert Settings.load(env={}, overrides=["retrieval.candidate_k=30", "retrieval.doc_slots=10"]).retrieval.doc_slots == 10
    with pytest.raises(ConfigError, match="retrieval.doc_slots"):
        Settings.load(env={}, overrides=["retrieval.candidate_k=30", "retrieval.doc_slots=31"])


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


def test_the_model_split_pipeline_runs_end_to_end(index, clients, settings):
    import dataclasses
    import json

    from supportdesk.pipelines.experiments import build_split_llm
    answer = json.dumps({"outcome": "request_info", "user_reply": "Which version?", "citations": []})
    case = CaseInput(case_id="c", title="Session state lost", body="st.session_state empty after F5")
    three = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, max_queries=3))
    split = build_split_llm(DenseRetriever(index, clients.embedding), clients, three)
    scripted(clients, json.dumps({"queries": ["session state reset", "st.session_state refresh", "F5 loses values"]}))
    clients.chat.script[clients.chat.calls + 2] = answer
    result = split.run(case)
    assert result.trace["n_queries"] == 3 and result.outcome == "request_info"


def test_the_combinations_use_the_clean_query_and_their_own_retrieval(index, clients, settings):
    import dataclasses

    from supportdesk.pipelines.experiments import build_clean_doc_slots, build_clean_hybrid_split
    case = CaseInput(case_id="c", title="Session state lost", body=REPORT)
    slots = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, doc_slots=3))
    run = build_clean_doc_slots(DocSlotsRetriever(index, clients.embedding, doc_slots=3), clients, slots)
    assert run.describe()["query"] == "CleanQuery"
    retrieval = run.retrieve(case)
    assert retrieval.queries[0].startswith("Session state lost\n") and "Checklist" not in retrieval.queries[0]
    assert sum(h.chunk.source_type == "documentation" for h in retrieval.hits) >= 3

    split = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, bm25_slots=3))
    hybrid = build_clean_hybrid_split(split_retriever := split_hybrid(index, clients), clients, split)
    assert hybrid.describe()["retriever"] == type(split_retriever).__name__ and len(hybrid.retrieve(case).hits) == 6


def split_hybrid(index, clients):
    return SplitHybridRetriever(index, clients.embedding, bm25_slots=3, bm25_k1=1.5, bm25_b=0.75)


def hybrid_slots(index, clients, slots):
    from supportdesk.retrieval.retriever import HybridDocSlotsRetriever
    return HybridDocSlotsRetriever(index, clients.embedding, doc_slots=slots, pool=30, rrf_k=60,
                                   bm25_k1=1.5, bm25_b=0.75)


def test_hybrid_doc_slots_keeps_places_for_documentation(index, clients):
    hits = hybrid_slots(index, clients, 3).search("session state reset rerun page", k=6)
    assert len({h.chunk.chunk_id for h in hits}) == 6 and [h.rank for h in hits] == list(range(1, 7))
    assert sum(h.chunk.source_type == "documentation" for h in hits) >= 3
    assert {h.source for h in hits} == {"hybrid"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_doc_slots_without_reserved_places_is_plain_hybrid(index, clients):
    from supportdesk.retrieval.retriever import HybridRetriever
    plain = HybridRetriever(index, clients.embedding, pool=30, rrf_k=60, bm25_k1=1.5, bm25_b=0.75)
    for query in ("session state reset rerun page", "st.session_state is empty after refresh"):
        assert [h.chunk.chunk_id for h in hybrid_slots(index, clients, 0).search(query, k=6)] == [
            h.chunk.chunk_id for h in plain.search(query, k=6)]


def test_clean_hybrid_doc_slots_differs_from_clean_doc_slots_only_in_its_retriever():
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    a, b = flat("clean_doc_slots"), flat("clean_hybrid_doc_slots")
    assert {k for k in a if a[k] != b[k]} == {"pipeline.name", "pipeline.retriever"}
    from supportdesk.registry import PIPELINES
    assert PIPELINES["clean_hybrid_doc_slots"].__doc__.startswith("clean_doc_slots with hybrid search")


def split_slots(index, clients, docs, bm25):
    from supportdesk.retrieval.retriever import SplitDocSlotsRetriever
    return SplitDocSlotsRetriever(index, clients.embedding, doc_slots=docs, bm25_slots=bm25,
                                  bm25_k1=1.5, bm25_b=0.75)


def test_split_doc_slots_takes_half_from_each_method_and_keeps_documentation_places(index, clients):
    hits = split_slots(index, clients, 3, 3).search("session state reset rerun page", k=6)
    assert len({h.chunk.chunk_id for h in hits}) == 6 and [h.rank for h in hits] == list(range(1, 7))
    assert [h.source for h in hits] == ["dense", "bm25"] * 3  # half each, taking turns
    assert sum(h.chunk.source_type == "documentation" for h in hits) >= 3
    for method in ("dense", "bm25"):  # each method's picks keep their own score order
        own = [h.score for h in hits if h.source == method]
        assert own == sorted(own, reverse=True)


def test_split_doc_slots_lets_dense_fill_when_bm25_finds_nothing(index, clients):
    hits = split_slots(index, clients, 3, 3).search("zzqx wvvq", k=6)
    assert len(hits) == 6 and {h.source for h in hits} == {"dense"}
    assert sum(h.chunk.source_type == "documentation" for h in hits) >= 3


def test_split_doc_slots_without_bm25_places_matches_dense_doc_slots(index, clients):
    query = "session state reset rerun page"
    plain = DocSlotsRetriever(index, clients.embedding, doc_slots=3).search(query, k=6)
    mine = split_slots(index, clients, 3, 0).search(query, k=6)
    assert sorted(h.chunk.chunk_id for h in mine) == sorted(h.chunk.chunk_id for h in plain)


def test_clean_split_doc_slots_differs_from_clean_hybrid_doc_slots_only_in_how_it_combines():
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    a, b = flat("clean_hybrid_doc_slots"), flat("clean_split_doc_slots")
    assert {k for k in a if a[k] != b[k]} == {"pipeline.name", "pipeline.retriever", "retrieval.bm25_slots"}


def test_retrieval_runs_record_which_method_found_each_chunk(tmp_path, capsys):
    import json

    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]
    assert main(["index", *base]) == 0
    assert main(["eval-retrieval", *base, "--pipeline", "clean_split_doc_slots", "--run-id", "r"]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in (tmp_path / "art" / "runs" / "r" / "results.jsonl").read_text().splitlines()]
    found = {h["found_by"] for row in rows for h in row["retrieved"]}
    assert found and found <= {"dense", "bm25"}
