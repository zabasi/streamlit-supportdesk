"""Single-change experiments: each is the baseline with exactly one difference, so its effect can be
measured on its own (`supportdesk eval-retrieval --pipeline NAME --split dev`). A change that helps
moves into `advanced`."""
from __future__ import annotations

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import StagedPipeline
from .stages import default_stages
from .stages.clean_query import CleanQuery
from .stages.llm_rerank import DocPlacesReranker, LLMReranker
from .stages.rewrite_query import QueryRewriter
from .stages.split_query import SentenceSplitQuery


def build_clean_query(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with a cleaner query: the report without its template, links and debug info."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_query", clients=clients, settings=settings, **stages)


def build_doc_slots(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with places in the search results kept for documentation (retriever dense_doc_slots)."""
    return StagedPipeline(name="doc_slots", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))


def build_hybrid_rrf(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with hybrid search: dense and BM25 results merged by Reciprocal Rank Fusion."""
    return StagedPipeline(name="hybrid_rrf", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))


def build_hybrid_split(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with split hybrid search: half the results from dense, half from BM25, not merged."""
    return StagedPipeline(name="hybrid_split", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))



def build_rewrite_llm(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with a rewritten query: the model turns the report into one documentation-style query."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    return StagedPipeline(name="rewrite_llm", clients=clients, settings=settings, **stages)


def build_split_llm(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with split queries: the model writes up to 3 focused queries, results merged by RRF."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    return StagedPipeline(name="split_llm", clients=clients, settings=settings, **stages)


def build_split_sentences(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Clean query split into up to 4 queries (title, code, sentences), results merged by RRF."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = SentenceSplitQuery(settings.context.max_query_chars, settings.retrieval.max_queries)
    return StagedPipeline(name="split_sentences", clients=clients, settings=settings, **stages)


# ---- combinations: built on the cleaned query, the strongest single change ----

def build_clean_doc_slots(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Cleaned query with 3 of the 6 results kept for documentation (clean_query + doc_slots)."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_doc_slots", clients=clients, settings=settings, **stages)



def build_clean_hybrid_split(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Cleaned query with split hybrid search: 3 results from dense, 3 from BM25 (clean_query + hybrid_split)."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_hybrid_split", clients=clients, settings=settings, **stages)


def build_clean_hybrid_doc_slots(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_doc_slots with hybrid search: both its lists merge dense and BM25 by rank fusion."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_hybrid_doc_slots", clients=clients, settings=settings, **stages)


def build_clean_split_doc_slots(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_doc_slots with split hybrid search: half the results from dense, half from BM25, not merged."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_split_doc_slots", clients=clients, settings=settings, **stages)


# ---- reranking on the best retrieval ----

def build_clean_doc_slots_rerank(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_doc_slots with reranking: 15 candidates ordered by one model call; at least 3 docs kept."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank", clients=clients, settings=settings, **stages)


def build_clean_hybrid_doc_slots_rerank(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_hybrid_doc_slots with reranking: 15 candidates ordered by one model call; at least 3 docs kept."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_hybrid_doc_slots_rerank", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_doc_slots where a model call chooses the 3 documentation chunks from 10 candidates."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = DocPlacesReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank_docs", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_prompt(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """clean_doc_slots_rerank with a reranking prompt that puts explanations first."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank_prompt", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Both reranking options: 10 doc candidates, the fixed prompt ranks all 15, best 6 kept (3+ docs)."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt", clients=clients, settings=settings, **stages)


# ---- variants of clean_doc_slots_rerank_docs_prompt (the design advanced uses) ----

def build_clean_hybrid_doc_slots_rerank_docs_prompt(retriever: Retriever, clients: Clients,
                                                    settings: Settings) -> StagedPipeline:
    """clean_doc_slots_rerank_docs_prompt with hybrid candidates (dense and BM25 merged by RRF)."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_hybrid_doc_slots_rerank_docs_prompt", clients=clients, settings=settings,
                          **stages)


def build_clean_doc_slots_rerank_docs_prompt_wide(retriever: Retriever, clients: Clients,
                                                  settings: Settings) -> StagedPipeline:
    """clean_doc_slots_rerank_docs_prompt with a wider pool: 20 candidates, 15 of them documentation."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_wide", clients=clients, settings=settings,
                          **stages)


# ---- what the reranker sees: query-focused or longer snippets ----

def _clean_rerank(name: str, retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name=name, clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_focused(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt where the reranker sees each candidate's best-matching window."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_focused", retriever, clients, settings)


def build_clean_hybrid_doc_slots_rerank_docs_prompt_focused(retriever, clients, settings):
    """clean_hybrid_doc_slots_rerank_docs_prompt where the reranker sees each candidate's best-matching window."""
    return _clean_rerank("clean_hybrid_doc_slots_rerank_docs_prompt_focused", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_long(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt where the reranker sees the first 1,200 characters of each candidate."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_long", retriever, clients, settings)


def build_clean_hybrid_doc_slots_rerank_docs_prompt_long(retriever, clients, settings):
    """clean_hybrid_doc_slots_rerank_docs_prompt where the reranker sees the first 1,200 characters of each candidate."""
    return _clean_rerank("clean_hybrid_doc_slots_rerank_docs_prompt_long", retriever, clients, settings)


# ---- how much documentation, and how many chunks, reach the model ----

def build_clean_doc_slots_rerank_docs_prompt_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt keeping at least 4 documentation chunks after reranking."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_keep4", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_keep5(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt keeping at least 5 documentation chunks after reranking."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_keep5", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_top8(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt showing the model 8 chunks instead of 6."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_top8", retriever, clients, settings)


# ---- better queries: the cleaned report plus written or split queries, on the advanced design ----

def _with_queries(name: str, query, retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    stages = default_stages(retriever, clients, settings)
    stages["query"] = query
    stages["reranker"] = LLMReranker(clients, settings)
    return StagedPipeline(name=name, clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_rewrite(retriever, clients, settings):
    """The advanced design searching with the cleaned report and one query naming the concept behind it."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_rewrite", QueryRewriter(clients, settings),
                         retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose(retriever, clients, settings):
    """The advanced design searching with the cleaned report and up to 3 queries, one per concept involved."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose", QueryRewriter(clients, settings),
                         retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_sentences(retriever, clients, settings):
    """The advanced design searching with the cleaned report and its pieces (title, code, sentences)."""
    query = SentenceSplitQuery(settings.context.max_query_chars, settings.retrieval.max_queries,
                               settings.retrieval.keep_report_query)
    return _with_queries("clean_doc_slots_rerank_docs_prompt_sentences", query, retriever, clients, settings)
