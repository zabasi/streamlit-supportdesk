"""The improved pipeline: the baseline's stages, its own profile (configs/pipelines/advanced.toml),
and the changes being tested. Two ways to change it, both leaving the baseline untouched:

1. Replace a stage (query, retriever, reranker, packet, generator, validator, summarizer) with a
   class from pipelines/stages/:
       stages["reranker"] = LLMReranker(clients, settings)

2. Add a node to this pipeline's graph, after an existing one. The function reads the plain-data
   state and returns the keys it updates (see docs/schemas.md, SupportState):
       extra_nodes = [("grade_evidence", grade_evidence, "rerank")]

See CLAUDE.md for the rules every stage and node follows.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import StagedPipeline
from .scope_gate import ScopeGatedPipeline
from .stages import default_stages
from .stages.clean_query import CleanQuery
from .stages.llm_rerank import LLMReranker
from .stages.scope_classifier import LLMScopeClassifier


def build_advanced(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Improved: cleaned query, 10 doc candidates of 15, reranked by one model call; best 6 kept (3+ docs)."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    stages["reranker"] = LLMReranker(clients, settings)
    extra_nodes: list = []
    return StagedPipeline(name="advanced", clients=clients, settings=settings, extra_nodes=extra_nodes, **stages)


def build_advanced_scoped(retriever: Retriever, clients: Clients, settings: Settings) -> ScopeGatedPipeline:
    """Improved advanced retrieval with an LLM pre-retrieval Streamlit scope gate."""
    base = build_advanced(retriever, clients, settings)
    classifier = LLMScopeClassifier(clients, Path("configs/prompts/scope_system.txt"))
    return ScopeGatedPipeline(
        name="advanced",
        clients=clients,
        settings=settings,
        classifier=classifier,
        extra_nodes=(),
        **{stage: getattr(base, stage) for stage in (
            "query", "retriever", "reranker", "packet", "generator", "validator", "summarizer"
        )},
    )
