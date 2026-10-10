"""Reranking with one model call: the candidates most useful for the report come first."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from ...retrieval.retriever import Hit
from ...types import CaseInput
from ..parsing import NotJson, extract_json
from .clean_query import CleanQuery

_LABEL = re.compile(r"^C(\d+)$")


def model_ranking(text: str, count: int) -> list[int]:
    """Positions (0-based) in the model's order, valid and without repeats; [] if unusable."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return []
    labels = payload.get("ranking") if isinstance(payload, dict) else None
    order = []
    for label in labels if isinstance(labels, list) else []:
        match = _LABEL.match(str(label).strip())
        if match and 1 <= int(match.group(1)) <= count and int(match.group(1)) - 1 not in order:
            order.append(int(match.group(1)) - 1)
    return order


class LLMReranker:
    """Reorders the candidates with one model call: the most useful for the report first."""

    def __init__(self, clients, settings):
        self.clients, self.snippet = clients, settings.retrieval.rerank_snippet_chars
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.rerank).read_text(encoding="utf-8")

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        if len(hits) < 2:
            return hits
        listing = "\n\n".join(
            f"C{n}: [{h.chunk.source_type}] {h.chunk.title}\n{' '.join(h.chunk.text.split())[: self.snippet]}"
            for n, h in enumerate(hits, start=1))
        content = f"## Report\n{self.clean(case, {})[0]}\n\n## Candidates\n{listing}"
        text, _ = self.clients.chat.complete([{"role": "system", "content": self.prompt},
                                              {"role": "user", "content": content}],
                                             stage="rerank", json_mode=True)
        order = model_ranking(text, len(hits))
        if not order:  # unusable output: keep the retriever's order
            return hits
        ranked = [hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in set(order)]
        return [replace(h, rank=n) for n, h in enumerate(ranked, start=1)]
