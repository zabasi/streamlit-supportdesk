"""The reference pipeline."""
from __future__ import annotations

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import Retriever
from .staged import StagedPipeline
from .stages import default_stages


def build_baseline(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Reference: retrieve the top-k chunks, ask the model once; no reranking, no validation."""
    return StagedPipeline(name="baseline", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))
