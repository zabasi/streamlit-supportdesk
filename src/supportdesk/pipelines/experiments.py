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
from .stages.evidence import ExpandingReranker, ExtractingReranker, GradedReranker, MergedReranker
from .stages.llm_rerank import DocPlacesReranker, LLMReranker
from .stages.rewrite_query import QueryRewriter
from .stages.sections import SectionReranker, SectionTrimmer
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


# ---- variants of clean_doc_slots_rerank_docs_prompt (the earlier advanced design) ----

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


# ---- variants of decompose: more room and focused snippets ----

def build_clean_doc_slots_rerank_docs_prompt_decompose_top8(retriever, clients, settings):
    """decompose showing the model 8 chunks instead of 6."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_top8", QueryRewriter(clients, settings),
                         retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_focused(retriever, clients, settings):
    """decompose where the reranker sees each candidate's best-matching window."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_focused", QueryRewriter(clients, settings),
                         retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_top8_focused(retriever, clients, settings):
    """decompose with 8 chunks and focused snippets for the reranker."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_top8_focused",
                         QueryRewriter(clients, settings), retriever, clients, settings)


# ---- the teammate's structure-aware chunkers, on the two best designs ----

def build_clean_doc_slots_rerank_docs_prompt_structure(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt on structure chunks (headings, code fences), with its own index."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_structure", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_token_structure(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt on token-budgeted structure chunks, with its own index."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_token_structure", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure(retriever, clients, settings):
    """decompose on structure chunks (headings, code fences), with its own index."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure",
                         QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure(retriever, clients, settings):
    """decompose on token-budgeted structure chunks, with its own index."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure",
                         QueryRewriter(clients, settings), retriever, clients, settings)


# ---- the chunking experiments with 8 chunks, at least 4 of them documentation ----


def build_clean_doc_slots_rerank_docs_prompt_structure_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_structure showing the model 8 chunks, at least 4 of them documentation."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_structure_top8_keep4", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_token_structure showing the model 8 chunks, at least 4 of them documentation."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure showing the model 8 chunks, at least 4 of them documentation."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_token_structure showing the model 8 chunks, at least 4 of them documentation."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4", QueryRewriter(clients, settings), retriever, clients, settings)


# ---- contextual chunk headers: chunks embedded with "page title › headings" in front ----


def build_clean_doc_slots_rerank_docs_prompt_decompose_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose with contextual chunk headers in the embeddings (its own index)."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_ctx", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure with contextual chunk headers in the embeddings (its own index)."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_token_structure with contextual chunk headers in the embeddings (its own index)."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4 with contextual chunk headers in the embeddings (its own index)."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top8_keep4", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4 with contextual chunk headers in the embeddings (its own index)."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx_top8_keep4", QueryRewriter(clients, settings), retriever, clients, settings)


# ---- contextual chunk headers on clean_doc_slots_rerank_docs_prompt (shares the header indexes) ----


def build_clean_doc_slots_rerank_docs_prompt_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt with contextual chunk headers in the embeddings (its own index)."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_ctx", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_structure_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_structure with contextual chunk headers in the embeddings (its own index)."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_structure_ctx", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_token_structure with contextual chunk headers in the embeddings (its own index)."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_token_structure_ctx", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_structure_ctx_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_structure_top8_keep4 with contextual chunk headers in the embeddings (its own index)."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_structure_ctx_top8_keep4", retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx_top8_keep4(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4 with contextual chunk headers in the embeddings (its own index)."""
    return _clean_rerank("clean_doc_slots_rerank_docs_prompt_token_structure_ctx_top8_keep4", retriever, clients, settings)


# ---- large structured chunks, without and with context headers, on decompose ----


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large(retriever, clients, settings):
    """decompose on large structure chunks."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_large", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx(retriever, clients, settings):
    """decompose on large structure chunks, with context headers."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx_all(retriever, clients, settings):
    """decompose on large structure chunks, with context headers naming every section."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx_all", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large(retriever, clients, settings):
    """decompose on large token_structure chunks."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx(retriever, clients, settings):
    """decompose on large token_structure chunks, with context headers."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx_all(retriever, clients, settings):
    """decompose on large token_structure chunks, with context headers naming every section."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx_all", QueryRewriter(clients, settings), retriever, clients, settings)


# ---- more small structure chunks within the same evidence budget ----


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top12(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure showing the model 12 small sections within the same 3,200-token budget."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_top12", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top15(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure showing the model 15 small sections within the same 3,200-token budget."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_top15", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top12(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx showing the model 12 small sections within the same 3,200-token budget."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top12", QueryRewriter(clients, settings), retriever, clients, settings)


def build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top15(retriever, clients, settings):
    """clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx showing the model 15 small sections within the same 3,200-token budget."""
    return _with_queries("clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top15", QueryRewriter(clients, settings), retriever, clients, settings)


# ---- search with fixed chunks, give the model whole sections ----

def _index_chunks(retriever) -> list:
    index = getattr(retriever, "index", None)
    return list(index.chunks) if index is not None else []


def build_clean_doc_slots_rerank_docs_prompt_decompose_sections(retriever, clients, settings):
    """decompose where the reranker ranks whole sections of the candidates instead of fixed chunks."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = SectionReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_sections", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_decompose_trimmed(retriever, clients, settings):
    """decompose where each kept chunk is cut down to its one section that best matches the report."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = SectionTrimmer(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_trimmed", clients=clients, settings=settings, **stages)


# ---- less noise without cutting chunks: merge overlaps, or keep only chunks graded relevant ----

def build_clean_doc_slots_rerank_docs_prompt_decompose_merged(retriever, clients, settings):
    """decompose where kept chunks of one page that overlap or touch are merged, so shared text appears once."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_decompose_graded(retriever, clients, settings):
    """decompose where the reranker also grades candidates and those graded not relevant are dropped."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = GradedReranker(clients, settings)
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_graded", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract(retriever, clients, settings):
    """decompose_merged, then one model call keeps only the parts of the chosen chunks that help."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = ExtractingReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_extract", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract_drop(retriever, clients, settings):
    """decompose_merged_extract asking what to drop, with the concept queries shown and the top chunk kept."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = ExtractingReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_extract_drop", clients=clients, settings=settings, **stages)


# ---- smaller fixed chunks on decompose_merged ----


def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320(retriever, clients, settings):
    """decompose_merged on 320-token fixed chunks, 6 chunks to the model."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320_top9(retriever, clients, settings):
    """decompose_merged on 320-token fixed chunks, 9 chunks to the model."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320_top9", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384(retriever, clients, settings):
    """decompose_merged on 384-token fixed chunks, 6 chunks to the model."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384", clients=clients, settings=settings, **stages)


def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384_top8(retriever, clients, settings):
    """decompose_merged on 384-token fixed chunks, 8 chunks to the model."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384_top8", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_page(retriever, clients, settings):
    """decompose_merged where the found documentation pages add their other chunks (page) before reranking."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = ExpandingReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_page", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_section(retriever, clients, settings):
    """decompose_merged where the found documentation pages add their other chunks (section) before reranking."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = ExpandingReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_section", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused(retriever, clients, settings):
    """decompose_merged where the reranker sees focused snippets."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_focused", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_concepts(retriever, clients, settings):
    """decompose_merged where the reranker sees the concept queries."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_concepts", clients=clients, settings=settings, **stages)



def build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts(retriever, clients, settings):
    """decompose_merged where the reranker sees focused snippets and the concept queries."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    stages["reranker"] = MergedReranker(clients, settings, _index_chunks(retriever))
    return StagedPipeline(name="clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts", clients=clients, settings=settings, **stages)
