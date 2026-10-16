"""Reranking with one model call: the candidates most useful for the report come first."""
from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

from ...retrieval.retriever import Hit
from ...tokenizer import lexical_tokens
from ...types import CaseInput
from ..parsing import NotJson, extract_json
from .clean_query import CleanQuery

_LABEL = re.compile(r"^C(\d+)$")
_HEADING = re.compile(r"^#{1,6} .+$", re.MULTILINE)


def term_weights(report: str, texts: list[str]) -> dict[str, float]:
    """Each report word, weighted by how rare it is among the candidates: a word every candidate
    contains cannot tell them apart and weighs 0 (the IDF idea of BM25, over these candidates)."""
    sets = [set(lexical_tokens(t)) for t in texts]
    weights = {}
    for term in set(lexical_tokens(report)):
        if term.replace(".", "").isdigit():  # a bare number ("1", "3") says nothing about the topic
            continue
        df = sum(term in words for words in sets)
        if df:
            weights[term] = math.log(len(sets) / df)
    return weights


def focused_snippet(text: str, weights: dict[str, float], size: int) -> str:
    """The ``size``-character window of ``text`` (starting at a line) whose report words weigh most,
    with the nearest section heading above it. Ties go to the earliest window."""
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n" and i + 1 < len(text)]
    best, best_score = 0, -1.0
    for start in starts:
        window = text[start:start + size]
        score = sum(weights.get(t, 0.0) for t in set(lexical_tokens(window)))
        if score > best_score:
            best, best_score = start, score
    window = text[best:best + size]
    if best + size < len(text) and " " in window:
        window = window.rsplit(" ", 1)[0]  # do not end in the middle of a word
    headings = [m.group(0) for m in _HEADING.finditer(text, 0, best)]
    heading = headings[-1][:80] + " … " if headings and not window.lstrip().startswith("#") else ""
    return heading + " ".join(window.split())


def model_ranking(text: str, count: int) -> list[int]:
    """Positions (0-based) in the model's order, valid and without repeats; [] if unusable."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return []
    labels = payload.get("ranking") if isinstance(payload, dict) else None
    order = []
    for label in labels if isinstance(labels, list) else []:
        if isinstance(label, dict):
            label = label.get("label", "")
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


GRADES = ("relevant", "partly", "not")


def model_grades(text: str, count: int) -> dict[int, str]:
    """Positions (0-based) the model graded, with their grade; {} if unusable or ungraded."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return {}
    entries = payload.get("ranking") if isinstance(payload, dict) else None
    grades = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        match = _LABEL.match(str(entry.get("label", "")).strip())
        grade = str(entry.get("grade", "")).strip().lower()
        if match and 1 <= int(match.group(1)) <= count and grade in GRADES:
            grades.setdefault(int(match.group(1)) - 1, grade)
    return grades


_HEADING_LINE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_headings(text: str, *, limit: int = 6) -> list[str]:
    """The Markdown headings inside a chunk, in order (not lines inside code fences): the sections it
    contains, which its focused window may not show."""
    out, fenced = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING_LINE.match(line)):
            out.append(m.group(1).strip())
    return out[:limit]


class LLMReranker:
    """Reorders the candidates with one model call: the most useful for the report first."""

    def __init__(self, clients, settings):
        self.clients, self.snippet = clients, settings.retrieval.rerank_snippet_chars
        self.snippet_mode = settings.retrieval.rerank_snippet  # start | focused
        self.show_sections = settings.retrieval.rerank_show_sections
        self.show_question = settings.retrieval.rerank_show_question
        from ...llm.factory import chat_client_for
        self.chat = chat_client_for(clients, settings, settings.models.rerank)  # models.rerank, or the chat model
        self.concepts = None
        if settings.retrieval.rerank_show_concepts:  # the query stage's concept queries (replayed from the cache)
            from .rewrite_query import make_query_rewriter
            self.concepts = make_query_rewriter(clients, settings)
            self.skip = 1 if settings.retrieval.keep_report_query else 0
        # documentation kept in the final top_k: retrieval.rerank_keep_docs, else retrieval.doc_slots
        keep = settings.retrieval.rerank_keep_docs
        self.top_k = settings.retrieval.top_k
        self.min_docs = settings.retrieval.doc_slots if keep is None else keep
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.rerank).read_text(encoding="utf-8")

    def model_order(self, case: CaseInput, hits: list[Hit]) -> list[int]:
        """One model call: positions in ``hits``, most useful first; [] if the output is unusable."""
        return model_ranking(self.model_text(case, hits), len(hits))

    def model_text(self, case: CaseInput, hits: list[Hit]) -> str:
        """The model's raw answer to the reranking request for ``hits``."""
        report = self.clean(case, {})[0]
        if self.snippet_mode == "focused":
            weights = term_weights(report, [h.chunk.text for h in hits])
            snippets = [focused_snippet(h.chunk.text, weights, self.snippet) for h in hits]
        else:
            snippets = [" ".join(h.chunk.text.split())[: self.snippet] for h in hits]
        def header(h: Hit) -> str:
            # rerank_show_sections: the sections the whole chunk contains; rerank_show_question: for a
            # candidate found through a section's symptom question, the question that matched the report
            sections = chunk_headings(h.chunk.text) if self.show_sections else []
            question = (h.chunk.meta or {}).get("matched_question") if self.show_question else None
            return (("Sections: " + " · ".join(sections) + "\n" if sections else "")
                    + (f'Found for: "{question}"\n' if question else ""))

        listing = "\n\n".join(f"C{n}: [{h.chunk.source_type}] {h.chunk.title}\n{header(h)}{snippet}"
                                for n, (h, snippet) in enumerate(zip(hits, snippets), start=1))
        content = f"## Report\n{report}\n\n"
        if self.concepts is not None:
            mechanisms = self.concepts(case, {})[self.skip:]
            if mechanisms:
                content += "## Mechanisms involved\n" + "\n".join(f"- {m}" for m in mechanisms) + "\n\n"
        content += f"## Candidates\n{listing}"
        text, _ = self.chat.complete([{"role": "system", "content": self.prompt},
                                              {"role": "user", "content": content}],
                                             stage="rerank", json_mode=True)
        return text

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        if len(hits) < 2:
            return hits
        order = self.model_order(case, hits)
        ranked = ([hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in set(order)]
                  if order else list(hits))  # unusable output: the retriever's order
        if self.min_docs:
            ranked = keep_doc_places(ranked, self.top_k, self.min_docs)
        if ranked == list(hits):
            return hits
        return [replace(h, rank=n) for n, h in enumerate(ranked, start=1)]


class DocPlacesReranker:
    """The model chooses which documentation fills the documentation places; the rest as retrieved."""

    def __init__(self, clients, settings):
        self.ranker = LLMReranker(clients, settings)
        self.top_k, self.places = settings.retrieval.top_k, settings.retrieval.rerank_doc_places

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        docs = [h for h in hits if h.chunk.source_type == "documentation"]
        order = self.ranker.model_order(case, docs) if len(docs) > 1 else []
        by_model = [docs[i] for i in order] + [d for i, d in enumerate(docs) if i not in set(order)]
        chosen = by_model[: self.places]  # the documentation places, as the model judged them
        ids = {h.chunk.chunk_id for h in chosen}
        rest = [h for h in hits if h.chunk.chunk_id not in ids][: self.top_k - len(chosen)]
        shown = sorted(chosen + rest, key=lambda h: h.rank)  # the retriever's order (similarity)
        kept = {h.chunk.chunk_id for h in shown}
        ranked = shown + [h for h in hits if h.chunk.chunk_id not in kept]
        return [replace(h, rank=n) for n, h in enumerate(ranked, start=1)]
