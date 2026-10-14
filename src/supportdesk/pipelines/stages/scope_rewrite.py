"""One model call that checks a report's scope and writes its search queries (pipeline.scope_gate = combined).

The separate scope gate and the query rewrite both read the same cleaned report, one call after the other.
This stage asks the model both questions at once: is the report about Streamlit, and which documentation
queries would find the mechanisms behind it. The scope gate and every stage that uses the concept queries
(the query stage, the reranker's "Mechanisms involved", ...) read the one answer, so advanced makes one
model call before search instead of two.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from ...types import CaseInput
from .clean_query import CleanQuery
from .rewrite_query import model_queries, with_report
from .scope_classifier import ScopeDecision, scope_from_text

_REMEMBERED = 256  # answers kept per shared call; a run reads each report a few times in a row


class ScopeRewrite:
    """The combined call, made at most once per report: (scope decision, the model's concept queries)."""

    def __init__(self, clients, settings):
        self.clients = clients
        self.max_queries = settings.retrieval.max_queries
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.scope_rewrite).read_text(encoding="utf-8").replace(
            "{max_queries}", str(self.max_queries))
        self.answers: OrderedDict[str, tuple[ScopeDecision, list[str]]] = OrderedDict()

    def __call__(self, case: CaseInput) -> tuple[ScopeDecision, list[str]]:
        report = self.clean(case, {})[0]
        if report in self.answers:  # the same report again: the gate, then the query stage, the reranker
            self.answers.move_to_end(report)
            return self.answers[report]
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": report}]
        text, _ = self.clients.chat.complete(messages, stage="scope_rewrite", json_mode=True)
        answer = scope_from_text(text, "combined"), model_queries(text, self.max_queries)
        self.answers[report] = answer
        while len(self.answers) > _REMEMBERED:
            self.answers.popitem(last=False)
        return answer


_SHARED: dict[tuple, ScopeRewrite] = {}


def shared_scope_rewrite(clients, settings) -> ScopeRewrite:
    """The one ScopeRewrite for these clients and settings, so the gate and every stage built from them
    share its answers: building advanced creates several stages that each need the concept queries."""
    key = (id(clients.chat), settings.prompts.scope_rewrite, settings.retrieval.max_queries,
           settings.context.max_query_chars)
    shared = _SHARED.get(key)
    if shared is None or shared.clients.chat is not clients.chat:  # a new client that reuses an old id
        shared = _SHARED[key] = ScopeRewrite(clients, settings)
    return shared


class CombinedScopeClassifier:
    """The scope gate's classifier, reading the combined call's decision."""

    def __init__(self, shared: ScopeRewrite):
        self.shared = shared

    def __call__(self, case: CaseInput) -> ScopeDecision:
        return self.shared(case)[0]


class CombinedQueryRewriter:
    """The query stage (and the concept queries for other stages), reading the combined call's queries;
    returns what QueryRewriter returns: the cleaned report first, then the model's queries."""

    def __init__(self, shared: ScopeRewrite, settings):
        self.shared = shared
        self.keep_report = settings.retrieval.keep_report_query

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        report = self.shared.clean(case, env)[0]
        return with_report(report, self.shared(case)[1], self.keep_report)


__all__ = ["CombinedQueryRewriter", "CombinedScopeClassifier", "ScopeRewrite", "shared_scope_rewrite"]
