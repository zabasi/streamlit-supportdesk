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


def keep_doc_places(ranked: list[Hit], top_k: int, min_docs: int) -> list[Hit]:
    """Make sure the first ``top_k`` hold at least ``min_docs`` documentation chunks, promoting the
    best-ranked documentation from below the cut in place of the lowest-ranked other chunks."""
    head, tail = list(ranked[:top_k]), list(ranked[top_k:])
    is_doc = lambda h: h.chunk.source_type == "documentation"
    for doc in [h for h in tail if is_doc(h)]:
        if sum(map(is_doc, head)) >= min_docs:
            break
        out = next((j for j in range(len(head) - 1, -1, -1) if not is_doc(head[j])), None)
        if out is None:
            break
        tail.insert(0, head.pop(out))
        head.append(doc)
        tail.remove(doc)
    return head + tail


class LLMReranker:
    """Reorders the candidates with one model call: the most useful for the report first."""

    def __init__(self, clients, settings):
        self.clients, self.snippet = clients, settings.retrieval.rerank_snippet_chars
        # with documentation places (retrieval.doc_slots), they survive reranking
        self.top_k, self.min_docs = settings.retrieval.top_k, settings.retrieval.doc_slots
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
        ranked = ([hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in set(order)]
                  if order else list(hits))  # unusable output: the retriever's order
        if self.min_docs:
            ranked = keep_doc_places(ranked, self.top_k, self.min_docs)
        if ranked == list(hits):
            return hits
        return [replace(h, rank=n) for n, h in enumerate(ranked, start=1)]
