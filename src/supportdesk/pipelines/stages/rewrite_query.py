"""Search queries written by the model: one documentation-style query, or several focused ones."""
from __future__ import annotations

from pathlib import Path

from ...types import CaseInput
from ..parsing import NotJson, extract_json
from .clean_query import CleanQuery


def model_queries(text: str, limit: int) -> list[str]:
    """The queries in the model's JSON, or [] if the output is unusable."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return []
    found = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(found, list):
        return []
    queries = [" ".join(q.split())[:300] for q in found if isinstance(q, str) and q.strip()]
    return list(dict.fromkeys(queries))[:limit]


class QueryRewriter:
    """Search queries written by the model from the cleaned report (up to retrieval.max_queries)."""

    def __init__(self, clients, settings):
        self.clients = clients
        self.max_queries = settings.retrieval.max_queries
        self.keep_report = settings.retrieval.keep_report_query
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.rewrite).read_text(encoding="utf-8").replace(
            "{max_queries}", str(self.max_queries))

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        report = self.clean(case, env)[0]
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": report}]
        text, _ = self.clients.chat.complete(messages, stage="rewrite", json_mode=True)
        return with_report(report, model_queries(text, self.max_queries), self.keep_report)


def with_report(report: str, written: list[str], keep_report: bool) -> list[str]:
    """The queries a query stage returns: the cleaned report first (retrieval.keep_report_query), then
    the model's queries; with no usable model queries, the report alone."""
    if keep_report:
        return [report] + [q for q in written if q != report]
    return written or [report]


def make_query_rewriter(clients, settings):
    """The writer of the concept queries for every stage that uses them (the query stage, the reranker's
    "Mechanisms involved", extraction, the coverage judge, section summaries). With
    pipeline.scope_gate = combined it is the combined scope-and-queries call, shared with the scope gate,
    so the report is read by the model once; otherwise its own rewrite call (replayed from the cache
    after the first stage that makes it)."""
    if settings.pipeline.scope_gate == "combined":
        from .scope_rewrite import CombinedQueryRewriter, shared_scope_rewrite
        return CombinedQueryRewriter(shared_scope_rewrite(clients, settings), settings)
    return QueryRewriter(clients, settings)
