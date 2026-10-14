"""Where pipelines and retrievers are registered. The CLI and the web UI both build through here,
so a new pipeline or retriever becomes available everywhere by adding one line."""
from __future__ import annotations

from .config import ConfigError, Settings
from .llm.factory import Clients
from .pipelines.baseline import build_baseline
from .pipelines.experiments import (
    build_clean_doc_slots,
    build_clean_doc_slots_rerank,
    build_clean_doc_slots_rerank_docs,
    build_clean_doc_slots_rerank_docs_prompt,
    build_clean_doc_slots_rerank_docs_prompt_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose,
    build_clean_doc_slots_rerank_docs_prompt_decompose_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose_focused,
    build_clean_doc_slots_rerank_docs_prompt_decompose_graded,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_concepts,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_page,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_section,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract_drop,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320_top9,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384_top8,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts,
    build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts_summaries,
    build_clean_doc_slots_rerank_docs_prompt_decompose_sections,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced_branch,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top12,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top15,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx_all,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top12,
    build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top15,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx_all,
    build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_decompose_top8,
    build_clean_doc_slots_rerank_docs_prompt_decompose_top8_focused,
    build_clean_doc_slots_rerank_docs_prompt_decompose_trimmed,
    build_clean_doc_slots_rerank_docs_prompt_focused,
    build_clean_doc_slots_rerank_docs_prompt_keep4,
    build_clean_doc_slots_rerank_docs_prompt_keep5,
    build_clean_doc_slots_rerank_docs_prompt_long,
    build_clean_doc_slots_rerank_docs_prompt_rewrite,
    build_clean_doc_slots_rerank_docs_prompt_sentences,
    build_clean_doc_slots_rerank_docs_prompt_structure,
    build_clean_doc_slots_rerank_docs_prompt_structure_ctx,
    build_clean_doc_slots_rerank_docs_prompt_structure_ctx_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_structure_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_token_structure,
    build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx,
    build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4,
    build_clean_doc_slots_rerank_docs_prompt_top8,
    build_clean_doc_slots_rerank_docs_prompt_wide,
    build_clean_doc_slots_rerank_prompt,
    build_clean_hybrid_doc_slots,
    build_clean_hybrid_doc_slots_rerank,
    build_clean_hybrid_doc_slots_rerank_docs_prompt,
    build_clean_hybrid_doc_slots_rerank_docs_prompt_focused,
    build_clean_hybrid_doc_slots_rerank_docs_prompt_long,
    build_clean_hybrid_split,
    build_clean_query,
    build_clean_split_doc_slots,
    build_doc_slots,
    build_hybrid_rrf,
    build_hybrid_split,
    build_rewrite_llm,
    build_split_llm,
    build_split_sentences,
)
from .pipelines.retrieve_loop import build_retrieve_loop
from .pipelines.scope_gate import build_advanced_scoped, build_scope_gate, build_scope_gate_rules
from .retrieval.index import VectorIndex
from .retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    DocSlotsRetriever,
    HybridDocSlotsRetriever,
    HybridRetriever,
    SplitDocSlotsRetriever,
    SplitHybridRetriever,
)

#: Pipelines by name: each builds the pipeline from (retriever, clients, settings); its docstring is its one-line description.
PIPELINES = {"baseline": build_baseline, "advanced": build_advanced_scoped,
             "scope_gate": build_scope_gate, "scope_gate_rules": build_scope_gate_rules,
             "retrieve_loop": build_retrieve_loop,
             "clean_query": build_clean_query, "doc_slots": build_doc_slots,
             "hybrid_rrf": build_hybrid_rrf, "hybrid_split": build_hybrid_split,
             "rewrite_llm": build_rewrite_llm,
             "split_llm": build_split_llm, "split_sentences": build_split_sentences,
             "clean_doc_slots": build_clean_doc_slots,
             "clean_hybrid_split": build_clean_hybrid_split,
             "clean_hybrid_doc_slots": build_clean_hybrid_doc_slots,
             "clean_split_doc_slots": build_clean_split_doc_slots,
             "clean_doc_slots_rerank": build_clean_doc_slots_rerank,
             "clean_hybrid_doc_slots_rerank": build_clean_hybrid_doc_slots_rerank,
             "clean_doc_slots_rerank_docs": build_clean_doc_slots_rerank_docs,
             "clean_doc_slots_rerank_prompt": build_clean_doc_slots_rerank_prompt,
             "clean_doc_slots_rerank_docs_prompt": build_clean_doc_slots_rerank_docs_prompt,
             "clean_hybrid_doc_slots_rerank_docs_prompt": build_clean_hybrid_doc_slots_rerank_docs_prompt,
             "clean_doc_slots_rerank_docs_prompt_wide": build_clean_doc_slots_rerank_docs_prompt_wide,
             "clean_doc_slots_rerank_docs_prompt_focused": build_clean_doc_slots_rerank_docs_prompt_focused,
             "clean_hybrid_doc_slots_rerank_docs_prompt_focused": build_clean_hybrid_doc_slots_rerank_docs_prompt_focused,
             "clean_doc_slots_rerank_docs_prompt_long": build_clean_doc_slots_rerank_docs_prompt_long,
             "clean_hybrid_doc_slots_rerank_docs_prompt_long": build_clean_hybrid_doc_slots_rerank_docs_prompt_long,
             "clean_doc_slots_rerank_docs_prompt_keep4": build_clean_doc_slots_rerank_docs_prompt_keep4,
             "clean_doc_slots_rerank_docs_prompt_keep5": build_clean_doc_slots_rerank_docs_prompt_keep5,
             "clean_doc_slots_rerank_docs_prompt_top8": build_clean_doc_slots_rerank_docs_prompt_top8,
             "clean_doc_slots_rerank_docs_prompt_rewrite": build_clean_doc_slots_rerank_docs_prompt_rewrite,
             "clean_doc_slots_rerank_docs_prompt_decompose": build_clean_doc_slots_rerank_docs_prompt_decompose,
             "clean_doc_slots_rerank_docs_prompt_sentences": build_clean_doc_slots_rerank_docs_prompt_sentences,
             "clean_doc_slots_rerank_docs_prompt_decompose_top8": build_clean_doc_slots_rerank_docs_prompt_decompose_top8,
             "clean_doc_slots_rerank_docs_prompt_decompose_focused": build_clean_doc_slots_rerank_docs_prompt_decompose_focused,
             "clean_doc_slots_rerank_docs_prompt_decompose_top8_focused": build_clean_doc_slots_rerank_docs_prompt_decompose_top8_focused,
             "clean_doc_slots_rerank_docs_prompt_structure": build_clean_doc_slots_rerank_docs_prompt_structure,
             "clean_doc_slots_rerank_docs_prompt_token_structure": build_clean_doc_slots_rerank_docs_prompt_token_structure,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure": build_clean_doc_slots_rerank_docs_prompt_decompose_structure,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure,
             "clean_doc_slots_rerank_docs_prompt_structure_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_structure_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_token_structure_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_decompose_ctx": build_clean_doc_slots_rerank_docs_prompt_decompose_ctx,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_ctx_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_ctx": build_clean_doc_slots_rerank_docs_prompt_ctx,
             "clean_doc_slots_rerank_docs_prompt_structure_ctx": build_clean_doc_slots_rerank_docs_prompt_structure_ctx,
             "clean_doc_slots_rerank_docs_prompt_token_structure_ctx": build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx,
             "clean_doc_slots_rerank_docs_prompt_structure_ctx_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_structure_ctx_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_token_structure_ctx_top8_keep4": build_clean_doc_slots_rerank_docs_prompt_token_structure_ctx_top8_keep4,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_large": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx_all": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_large_ctx_all,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx,
             "clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx_all": build_clean_doc_slots_rerank_docs_prompt_decompose_token_structure_large_ctx_all,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_top12": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top12,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_top15": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_top15,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top12": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top12,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top15": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_ctx_top15,
             "clean_doc_slots_rerank_docs_prompt_decompose_sections": build_clean_doc_slots_rerank_docs_prompt_decompose_sections,
             "clean_doc_slots_rerank_docs_prompt_decompose_trimmed": build_clean_doc_slots_rerank_docs_prompt_decompose_trimmed,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged": build_clean_doc_slots_rerank_docs_prompt_decompose_merged,
             "clean_doc_slots_rerank_docs_prompt_decompose_graded": build_clean_doc_slots_rerank_docs_prompt_decompose_graded,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_extract": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_extract_drop": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_extract_drop,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320_top9": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed320_top9,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384_top8": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_fixed384_top8,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_page": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_page,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_section": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_expand_section,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_focused": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_concepts": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_concepts,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced,
             "clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced_branch": build_clean_doc_slots_rerank_docs_prompt_decompose_structure_advanced_branch,
             "clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts_summaries": build_clean_doc_slots_rerank_docs_prompt_decompose_merged_focused_concepts_summaries}

RETRIEVERS = {
    "dense": lambda index, clients, s: DenseRetriever(index, clients.embedding),
    "hybrid_split": lambda index, clients, s: SplitHybridRetriever(
        index, clients.embedding, bm25_slots=s.retrieval.bm25_slots,
        bm25_k1=s.retrieval.bm25_k1, bm25_b=s.retrieval.bm25_b),
    "hybrid_doc_slots": lambda index, clients, s: HybridDocSlotsRetriever(
        index, clients.embedding, doc_slots=s.retrieval.doc_slots, pool=s.retrieval.hybrid_pool,
        rrf_k=s.retrieval.rrf_k, bm25_k1=s.retrieval.bm25_k1, bm25_b=s.retrieval.bm25_b),
    "split_doc_slots": lambda index, clients, s: SplitDocSlotsRetriever(
        index, clients.embedding, doc_slots=s.retrieval.doc_slots, bm25_slots=s.retrieval.bm25_slots,
        bm25_k1=s.retrieval.bm25_k1, bm25_b=s.retrieval.bm25_b),
    "dense_doc_slots": lambda index, clients, s: DocSlotsRetriever(
        index, clients.embedding, doc_slots=s.retrieval.doc_slots),
    "bm25": lambda index, clients, s: BM25Retriever(
        index.chunks, k1=s.retrieval.bm25_k1, b=s.retrieval.bm25_b),
    "hybrid": lambda index, clients, s: HybridRetriever(
        index, clients.embedding, pool=s.retrieval.hybrid_pool, rrf_k=s.retrieval.rrf_k,
        bm25_k1=s.retrieval.bm25_k1, bm25_b=s.retrieval.bm25_b),
}


def pipeline_description(name: str) -> str:
    return (PIPELINES[name].__doc__ or "").strip().splitlines()[0]


def _require(kind: str, name: str, options) -> None:
    if name not in options:
        hint = " (run `supportdesk pipelines` to see them)" if kind == "pipeline" else ""
        raise ConfigError(f"unknown {kind} {name!r}; choose from {sorted(options)}{hint}")


def require(pipeline: str, retriever: str) -> None:
    """Fail fast, before anything is loaded, when a name is not registered."""
    _require("pipeline", pipeline, PIPELINES)
    _require("retriever", retriever, RETRIEVERS)


def make_retriever(name: str, index: VectorIndex, clients: Clients, settings: Settings):
    _require("retriever", name, RETRIEVERS)
    return RETRIEVERS[name](index, clients, settings)


def make_pipeline(name: str, retriever: str, index: VectorIndex, clients: Clients,
                  settings: Settings):
    require(name, retriever)
    return PIPELINES[name](make_retriever(retriever, index, clients, settings), clients, settings)
