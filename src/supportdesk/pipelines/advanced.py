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

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import StagedPipeline
from .stages import default_stages


def build_advanced(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Improved: the baseline's stages with its own profile (more chunks, hybrid search)."""
    stages = default_stages(retriever, clients, settings)
    extra_nodes: list = []
    return StagedPipeline(name="advanced", clients=clients, settings=settings, extra_nodes=extra_nodes, **stages)
