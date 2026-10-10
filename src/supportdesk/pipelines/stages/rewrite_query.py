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
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.rewrite).read_text(encoding="utf-8").replace(
            "{max_queries}", str(self.max_queries))

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        report = self.clean(case, env)[0]
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": report}]
        text, _ = self.clients.chat.complete(messages, stage="rewrite", json_mode=True)
        return model_queries(text, self.max_queries) or [report]  # unusable output: the cleaned report
