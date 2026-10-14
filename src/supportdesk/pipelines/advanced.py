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
from .stages.evidence import CorrectiveReranker
from .stages.rewrite_query import make_query_rewriter


def build_advanced(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Improved: report + concept queries; reranked on focused snippets with the concepts; best 6 merged."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = make_query_rewriter(clients, settings)  # with scope_gate = combined: the shared call
    index = getattr(retriever, "index", None)  # pages are rebuilt from the index's chunks for merging
    # CorrectiveReranker with corrective_turns = 0 is ExpandingReranker, and that with expand = none is
    # MergedReranker; --set retrieval.corrective_turns=1 or retrieval.expand=... try those on this design
    stages["reranker"] = CorrectiveReranker(clients, settings, list(index.chunks) if index is not None else [],
                                            retriever)
    extra_nodes: list = []
    return StagedPipeline(name="advanced", clients=clients, settings=settings, extra_nodes=extra_nodes, **stages)
