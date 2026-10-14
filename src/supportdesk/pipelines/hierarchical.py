"""Hierarchical retrieval, alone and on top of the advanced design.

Both use a hierarchical retriever (retrieval/hierarchy.py): the query is compared with the section
summaries first, and some of the returned chunks come from inside the best-matching sections.
"""
from __future__ import annotations

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import StagedPipeline
from .stages import default_stages
from .stages.evidence import CorrectiveReranker
from .stages.rewrite_query import QueryRewriter


def build_hier_baseline(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with hierarchical retrieval: section summaries first, then chunks inside the best sections."""
    return StagedPipeline(name="hier_baseline", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))


def build_advanced_hier(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """advanced with hierarchical retrieval: each of its searches also looks inside the best sections."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = QueryRewriter(clients, settings)
    index = getattr(retriever, "index", None)
    stages["reranker"] = CorrectiveReranker(clients, settings, list(index.chunks) if index is not None else [],
                                            retriever)
    return StagedPipeline(name="advanced_hier", clients=clients, settings=settings, **stages)
