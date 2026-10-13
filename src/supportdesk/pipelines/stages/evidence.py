"""Less noise in the prompt without cutting chunks apart.

MergedReranker: the reranker keeps chunks as before; kept chunks of one page that overlap or touch are
then merged into one continuous span, so shared text appears once. Nothing is cut, so no gold is lost.
GradedReranker: the reranker also grades each candidate (relevant, partly, not); chunks graded "not"
are dropped, so the model gets as many chunks as are relevant, up to top_k (at least
retrieval.min_chunks). The documentation minimum applies only to chunks not graded "not".
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from ...retrieval.retriever import Hit
from ...types import CaseInput, Chunk
from ..parsing import NotJson, extract_json
from .clean_query import CleanQuery
from .llm_rerank import LLMReranker, keep_doc_places, model_grades, model_ranking
from .sections import Sections, page_texts


def merge_overlaps(hits: list[Hit], pages: dict[str, str]) -> list[Hit]:
    """Hits of one page whose spans overlap or touch become one span (best rank kept); others unchanged."""
    groups: list[list[Hit]] = []
    for h in hits:
        c = h.chunk
        mergeable = c.source_id in pages and c.start_char is not None and c.end_char is not None
        for group in groups:
            g = group[0].chunk
            if mergeable and g.source_id == c.source_id and any(
                    o.chunk.start_char <= c.end_char and c.start_char <= o.chunk.end_char for o in group):
                group.append(h)
                break
        else:
            groups.append([h])
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        first, page = group[0], pages[group[0].chunk.source_id]
        a, b = min(h.chunk.start_char for h in group), max(h.chunk.end_char for h in group)
        chunk = replace(first.chunk, chunk_id=f"{first.chunk.source_id}#{a}-{b}", text=page[a:b],
                        start_char=a, end_char=b)
        merged.append(replace(first, chunk=chunk))
    return [replace(h, rank=n) for n, h in enumerate(merged, start=1)]


class MergedReranker:
    """The reranker as before, then overlapping or touching kept chunks of one page are merged."""

    def __init__(self, clients, settings, chunks: list[Chunk]):
        self.ranker, self.pages = LLMReranker(clients, settings), page_texts(chunks)
        self.top_k = settings.retrieval.top_k

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        kept = self.ranker(case, query, hits)[: self.top_k]
        return merge_overlaps(kept, self.pages)


class GradedReranker:
    """The reranker ranks and grades the candidates; chunks graded "not" relevant are dropped."""

    def __init__(self, clients, settings):
        self.ranker = LLMReranker(clients, settings)
        r = settings.retrieval
        self.top_k, self.min_chunks = r.top_k, min(r.min_chunks, r.top_k)
        self.min_docs = r.doc_slots if r.rerank_keep_docs is None else r.rerank_keep_docs

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        if len(hits) < 2:
            return hits
        text = self.ranker.model_text(case, hits)
        order, grades = model_ranking(text, len(hits)), model_grades(text, len(hits))
        if not order or not grades:  # unusable or ungraded output: behave like the plain reranker
            ranked = ([hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in set(order)]
                      if order else list(hits))
            ranked = keep_doc_places(ranked, self.top_k, self.min_docs) if self.min_docs else ranked
            return [replace(h, rank=n) for n, h in enumerate(ranked[: self.top_k], start=1)]
        ranked = [i for i in order] + [i for i in range(len(hits)) if i not in set(order)]
        useful = [i for i in ranked if grades.get(i) != "not"]
        kept = useful[: self.top_k]
        if len(kept) < self.min_chunks:  # too few judged relevant: fill from the ranking
            kept += [i for i in ranked if i not in kept][: self.min_chunks - len(kept)]
        chosen = [hits[i] for i in kept]
        others = [hits[i] for i in useful if i not in kept]  # documentation the minimum may promote
        docs_available = sum(h.chunk.source_type == "documentation" for h in chosen + others)
        if self.min_docs and docs_available:
            chosen = keep_doc_places(chosen + others, len(chosen), min(self.min_docs, docs_available))[: len(chosen)]
        return [replace(h, rank=n) for n, h in enumerate(chosen, start=1)]


_PART = re.compile(r"^P(\d+)$")


def listed_parts(text: str, key: str, count: int) -> list[int] | None:
    """Positions (0-based) of the parts the model listed under ``key``, in order; None if unusable."""
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return None
    labels = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        return None
    found = []
    for label in labels if isinstance(labels, list) else []:
        match = _PART.match(str(label).strip())
        if match and 1 <= int(match.group(1)) <= count and int(match.group(1)) - 1 not in found:
            found.append(int(match.group(1)) - 1)
    return sorted(found)


def kept_parts(text: str, count: int) -> list[int]:
    """Positions (0-based) of the parts the model kept, in order; [] if the output is unusable."""
    return listed_parts(text, "keep", count) or []


class ExtractingReranker:
    """MergedReranker's chunks, then one model call keeps only the parts that help with the report.
    Documentation chunks are split into their sections (clipped to the chunk, so nothing is added);
    issues and comments stay whole. The model sees each part's full text. Kept neighbouring parts of a
    page are joined again. Unusable output, or nothing kept, keeps everything."""

    def __init__(self, clients, settings, chunks: list[Chunk]):
        self.merged, self.sections = MergedReranker(clients, settings, chunks), Sections(chunks)
        self.clients, self.clean = clients, CleanQuery(settings.context.max_query_chars)
        self.prompt = Path(settings.prompts.extract).read_text(encoding="utf-8")
        r = settings.retrieval
        self.mode, self.protect_top = r.extract_mode, r.extract_protect_top
        self.concepts = None
        if r.extract_show_concepts:  # the query stage's concept queries (its call replays from the cache)
            from .rewrite_query import QueryRewriter
            self.concepts, self.skip = QueryRewriter(clients, settings), 1 if r.keep_report_query else 0

    def parts(self, hit: Hit) -> list[Hit]:
        c = hit.chunk
        if c.source_id not in self.sections.pages:
            return [hit]
        page, found = self.sections.pages[c.source_id], []
        for a, b in self.sections.spans[c.source_id]:
            lo, hi = max(a, c.start_char), min(b, c.end_char)
            if hi > lo and page[lo:hi].strip():
                found.append(replace(hit, chunk=replace(c, chunk_id=f"{c.source_id}#{lo}-{hi}", text=page[lo:hi],
                                                        start_char=lo, end_char=hi)))
        return found or [hit]

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        kept = self.merged(case, query, hits)
        owned = [(i, p) for i, h in enumerate(kept) for p in self.parts(h)]
        parts = [p for _, p in owned]
        if len(parts) < 2:
            return kept
        protected = {n for n, (i, _) in enumerate(owned) if i < self.protect_top}
        listing = "\n\n".join(f"P{n}: [{p.chunk.source_type}] {p.chunk.title}\n{p.chunk.text.strip()}"
                                for n, p in enumerate(parts, start=1))
        content = f"## Report\n{self.clean(case, {})[0]}\n\n"
        if self.concepts is not None:
            mechanisms = self.concepts(case, {})[self.skip:]
            if mechanisms:
                content += "## Mechanisms involved\n" + "\n".join(f"- {m}" for m in mechanisms) + "\n\n"
        content += f"## Parts\n{listing}"
        text, _ = self.clients.chat.complete([{"role": "system", "content": self.prompt},
                                              {"role": "user", "content": content}],
                                             stage="extract", json_mode=True)
        listed = listed_parts(text, self.mode, len(parts))
        if listed is None:  # unusable output: keep everything
            return kept
        if self.mode == "drop":
            chosen = [n for n in range(len(parts)) if n not in set(listed) or n in protected]
        else:
            chosen = sorted(set(listed) | protected)
        if not chosen:  # nothing kept: keep everything
            return kept
        return merge_overlaps([parts[i] for i in chosen], self.sections.pages)



class ExpandingReranker:
    """Before reranking, the documentation pages among the candidates bring more of their chunks: all of
    them (expand = page) or only those sharing a section with a found chunk (expand = section), nearest
    first, inserted after the chunk that brought them, up to retrieval.expand_max_candidates in total.
    Then MergedReranker ranks, keeps the best and merges overlaps. Added chunks are marked "expand"."""

    def __init__(self, clients, settings, chunks: list[Chunk]):
        self.inner, self.sections = MergedReranker(clients, settings, chunks), Sections(chunks)
        self.mode, self.cap = settings.retrieval.expand, settings.retrieval.expand_max_candidates
        self.by_page: dict[str, list[Chunk]] = {}
        for c in chunks:
            if c.source_type == "documentation" and c.start_char is not None:
                self.by_page.setdefault(c.source_id, []).append(c)

    def siblings(self, chunk: Chunk) -> list[Chunk]:
        page = self.by_page.get(chunk.source_id, [])
        if self.mode == "section" and chunk.source_id in self.sections.spans:
            mine = [(a, b) for a, b in self.sections.spans[chunk.source_id]
                    if a < chunk.end_char and b > chunk.start_char]
            page = [c for c in page if any(a < c.end_char and b > c.start_char for a, b in mine)]
        return sorted((c for c in page if c.chunk_id != chunk.chunk_id),
                      key=lambda c: (abs(c.start_char - chunk.start_char), c.start_char))

    def expand(self, hits: list[Hit]) -> list[Hit]:
        if self.mode == "none":
            return hits
        seen = {h.chunk.chunk_id for h in hits}
        room = max(0, self.cap - len(hits))
        out = []
        for h in hits:
            out.append(h)
            if h.chunk.source_type != "documentation":
                continue
            for c in self.siblings(h.chunk):
                if room and c.chunk_id not in seen:
                    out.append(Hit(c, h.score, 0, "expand"))
                    seen.add(c.chunk_id)
                    room -= 1
        return [replace(h, rank=n) for n, h in enumerate(out, start=1)]

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        return self.inner(case, query, self.expand(hits))
