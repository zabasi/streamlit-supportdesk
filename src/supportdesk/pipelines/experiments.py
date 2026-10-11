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
