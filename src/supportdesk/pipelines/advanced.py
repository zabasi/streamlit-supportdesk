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

import dataclasses
from pathlib import Path

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import BM25Retriever, Hit, Retriever, fuse_hits
from .staged import StagedPipeline
from .stages import default_stages
from .stages.clean_query import CleanQuery
from .stages.evidence import CorrectiveReranker
from .stages.report_views import report_views
from .stages.rewrite_query import make_query_rewriter
from .state import case_from_dict, hit_to_dict

CONCEPTS_RERANK, PLAIN_RERANK = "rerank_docs_concepts_system.txt", "rerank_docs_system.txt"

class WeightedFusionPipeline(StagedPipeline):
    """advanced's pipeline. With retrieval.concept_weight below 1, the concept queries' rankings count less
    than the cleaned report's when they are fused, so the report's matches keep the lead and a page found
    only by a concept query can still reach the candidates. At 1 (the default) it is StagedPipeline as is.
    (Kept here, not in StagedPipeline, so the code that defines the baseline is unchanged.)"""

    def _retrieve_node(self, state) -> dict:
        r, queries = self.settings.retrieval, state["queries"]
        if r.pool_mode == "quota":
            return {"candidates": [hit_to_dict(h) for h in self._quota_pool(state)]}
        if r.pool_mode == "union":
            return {"candidates": [hit_to_dict(h) for h in self._union_pool(state)]}
        case = case_from_dict(state["case"])
        views = report_views(case, r.report_views)  # the title / code views, searched besides the queries
        if not views and (r.concept_weight == 1.0 or not r.keep_report_query or len(queries) < 2):
            return super()._retrieve_node(state)
        as_of = case.created_at if r.historical_filter else None
        pool = self.settings.pool_size
        rankings = [self.retriever.search(q, k=pool, as_of=as_of) for q in [*queries, *views]]
        concept = r.concept_weight if r.keep_report_query else 1.0
        weights = ([1.0] + [concept] * (len(queries) - 1)  # the report first, then the concepts
                   + [r.view_weight] * len(views))  # then the views
        return {"candidates": [hit_to_dict(h) for h in fuse_hits(rankings, pool, rrf_k=r.rrf_k, weights=weights)]}

    def _union_pool(self, state) -> list[Hit]:
        """retrieval.pool_mode = union: no fusion and nothing displaced. The cleaned report (the first query)
        keeps its own best union_report candidates; each concept query then adds its best union_per_concept
        candidates that are not already in the pool, taken in turns (one from each query), until
        union_max candidates. The reranker ranks them all."""
        r, queries = self.settings.retrieval, state["queries"]
        case = case_from_dict(state["case"])
        as_of = case.created_at if r.historical_filter else None
        report, concepts = queries[0], queries[1:]
        picked: dict[str, Hit] = {}
        found = self.retriever.search(report, k=r.union_report, as_of=as_of)
        extras = {"question", "bm25"}  # candidates a question_doc_slots retriever adds: kept, not cut
        own = [h for h in found if h.source not in extras][: r.union_report] + [h for h in found if h.source in extras]
        for hit in own:
            if len(picked) < r.union_max and hit.chunk.chunk_id not in picked:
                source = hit.source if hit.source in extras else "report"
                picked[hit.chunk.chunk_id] = Hit(hit.chunk, hit.score, len(picked) + 1, source)
        depth = r.union_per_concept + r.union_report  # deep enough to find that many new ones
        rankings = [self.retriever.search(q, k=depth, as_of=as_of) for q in concepts]
        positions, added = [0] * len(rankings), [0] * len(rankings)
        while len(picked) < r.union_max:
            progressed = False
            for n, ranking in enumerate(rankings):  # one new candidate from each concept query in turn
                if added[n] >= r.union_per_concept or len(picked) >= r.union_max:
                    continue
                while positions[n] < len(ranking) and ranking[positions[n]].chunk.chunk_id in picked:
                    positions[n] += 1
                if positions[n] < len(ranking):
                    hit = ranking[positions[n]]
                    picked[hit.chunk.chunk_id] = Hit(hit.chunk, hit.score, len(picked) + 1, "concept")
                    positions[n] += 1
                    added[n] += 1
                    progressed = True
            if not progressed:
                break
        return list(picked.values())

    def _quota_pool(self, state) -> list[Hit]:
        """retrieval.pool_mode = quota: fixed shares instead of one fused list. The cleaned report (the first
        query) gives its best quota_report_dense dense chunks and its best quota_report_bm25 keyword (BM25)
        chunks; the concept queries give quota_concepts_dense dense chunks between them, taken in turns from
        each one's ranking. A chunk already picked is skipped. All of them go to the reranker."""
        r, queries = self.settings.retrieval, state["queries"]
        case = case_from_dict(state["case"])
        as_of = case.created_at if r.historical_filter else None
        depth = max(self.settings.pool_size, r.quota_report_dense, r.quota_report_bm25, r.quota_concepts_dense)
        picked: dict[str, Hit] = {}

        def take(ranking: list[Hit], n: int, source: str) -> None:
            added = 0
            for hit in ranking:
                if added == n:
                    return
                if hit.chunk.chunk_id not in picked:
                    picked[hit.chunk.chunk_id] = Hit(hit.chunk, hit.score, len(picked) + 1, source)
                    added += 1

        report, concepts = queries[0], queries[1:]
        take(self.retriever.search(report, k=depth, as_of=as_of), r.quota_report_dense, "report-dense")
        take(self._bm25().search(report, k=depth, as_of=as_of), r.quota_report_bm25, "report-bm25")
        rankings = [self.retriever.search(q, k=depth, as_of=as_of) for q in concepts]
        positions, added = [0] * len(rankings), 0
        while rankings and added < r.quota_concepts_dense and any(p < len(x) for p, x in zip(positions, rankings)):
            for n, ranking in enumerate(rankings):  # one from each concept query in turn
                while positions[n] < len(ranking) and ranking[positions[n]].chunk.chunk_id in picked:
                    positions[n] += 1
                if positions[n] < len(ranking) and added < r.quota_concepts_dense:
                    hit = ranking[positions[n]]
                    picked[hit.chunk.chunk_id] = Hit(hit.chunk, hit.score, len(picked) + 1, "concept-dense")
                    positions[n] += 1
                    added += 1
        return list(picked.values())

    def _bm25(self) -> BM25Retriever:
        if getattr(self, "_bm25_retriever", None) is None:
            index = getattr(self.retriever, "index", None)
            if index is None:
                raise ValueError("retrieval.pool_mode = quota needs a retriever over an index (dense_doc_slots)")
            r = self.settings.retrieval
            self._bm25_retriever = BM25Retriever(list(index.chunks), k1=r.bm25_k1, b=r.bm25_b)
        return self._bm25_retriever

def build_advanced(retriever: Retriever, clients: Clients, settings: Settings) -> StagedPipeline:
    """Improved: report + concept queries; reranked on focused snippets with the concepts; best 6 merged.
    With retrieval.decompose = false, the cleaned report is the only query: no concept queries, no
    "Mechanisms involved" for the reranker, and the reranking prompt without them; the rest is unchanged."""
    if not settings.retrieval.decompose:
        prompts = settings.prompts
        if Path(prompts.rerank).name == CONCEPTS_RERANK:  # the concepts prompt refers to a list that is now absent
            prompts = dataclasses.replace(prompts, rerank=Path(prompts.rerank).with_name(PLAIN_RERANK))
        settings = dataclasses.replace(settings, prompts=prompts, retrieval=dataclasses.replace(
            settings.retrieval, rerank_show_concepts=False))
    stages = default_stages(retriever, clients, settings)
    stages["query"] = (make_query_rewriter(clients, settings) if settings.retrieval.decompose  # combined gate: shared
                       else CleanQuery(settings.context.max_query_chars))
    index = getattr(retriever, "index", None)  # pages are rebuilt from the index's chunks for merging
    # CorrectiveReranker with corrective_turns = 0 is ExpandingReranker, and that with expand = none is
    # MergedReranker; --set retrieval.corrective_turns=1 or retrieval.expand=... try those on this design
    stages["reranker"] = CorrectiveReranker(clients, settings, list(index.chunks) if index is not None else [],
                                            retriever)
    if settings.pipeline.generator == "controlled_v2":  # one structured assessment; Python routes the action
        from .stages.controlled import ControlledGenerator

        stages["generator"] = ControlledGenerator(clients, settings)
    extra_nodes: list = []
    return WeightedFusionPipeline(name="advanced", clients=clients, settings=settings, extra_nodes=extra_nodes,
                                  **stages)
