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


def build_clean_query(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with a cleaner query: the report without its template, links and debug info."""
    stages = default_stages(retriever, clients, settings)
    stages["query"] = CleanQuery(settings.context.max_query_chars)
    return StagedPipeline(name="clean_query", clients=clients, settings=settings, **stages)


def build_doc_slots(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Baseline with places in the search results kept for documentation (retriever dense_doc_slots)."""
    return StagedPipeline(name="doc_slots", clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))
