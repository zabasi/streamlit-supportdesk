"""Search with fixed chunks, give the model whole sections.

Fixed chunks are easy to find but carry neighbouring sections as noise; single sections are clean to
read but hard to find. These stages keep the search as it is and, before the prompt, replace each
retrieved documentation chunk by the page sections it overlaps. A section is a top-level section of
its page (cut at the shallowest heading level from ## down, outside code fences), so a "##" section
keeps its "###" subsections. A section bigger than the chunk is cut to the part inside the chunk, so
the evidence can only shrink. Each section becomes a chunk with exact offsets into its page, so
citations and the passage metrics work unchanged. Pages are rebuilt from the index's own chunks.
"""
from __future__ import annotations

import re
from dataclasses import replace
from itertools import pairwise

from ...retrieval.retriever import Hit
from ...tokenizer import lexical_tokens
from ...types import CaseInput, Chunk
from .clean_query import CleanQuery
from .llm_rerank import LLMReranker, term_weights

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}(#{2,6})[ \t]+\S")


def page_texts(chunks: list[Chunk]) -> dict[str, str]:
    """Each documentation page's full text, rebuilt from its chunks (they tile the page with
    overlaps). A page whose chunks leave a gap is left out, and its chunks are kept as they are."""
    by_page: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.source_type == "documentation" and c.start_char is not None and c.end_char is not None:
            by_page.setdefault(c.source_id, []).append(c)
    pages = {}
    for source_id, parts in by_page.items():
        parts.sort(key=lambda c: c.start_char)
        if parts[0].start_char != 0:
            continue
        text, reached = "", 0
        for c in parts:
            if c.start_char > reached:  # a gap: the page cannot be rebuilt exactly
                break
            text += c.text[reached - c.start_char:] if c.end_char > reached else ""
            reached = max(reached, c.end_char)
        else:
            pages[source_id] = text
    return pages


def _headings(text: str) -> list[tuple[int, int]]:
    """(position, level) of every heading from ## down, outside code fences."""
    headings, fence, pos = [], None, 0
    for line in text.splitlines(keepends=True):
        opener = _FENCE.match(line)
        if fence:
            if opener and opener.group(1)[0] == fence:
                fence = None
        elif opener:
            fence = opener.group(1)[0]
        else:
            heading = _HEADING.match(line)
            if heading:
                headings.append((pos, len(heading.group(1))))
        pos += len(line)
    return headings


def section_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of the page's top-level sections: cut before each heading of the shallowest level
    from ## down (outside code fences). Text before the first such heading is a section too."""
    headings = _headings(text)
    if not headings:
        return [(0, len(text))]
    top = min(level for _, level in headings)
    cuts = sorted({0, *(p for p, level in headings if level == top), len(text)})
    return [(a, b) for a, b in pairwise(cuts) if text[a:b].strip()]


def nested_sections(text: str) -> list[tuple[int, int, int]]:
    """(start, end, level) of every section at every level from ## down: a section runs from its heading
    to the next heading of the same or a higher level (outside code fences)."""
    headings = _headings(text)
    out = []
    for n, (start, level) in enumerate(headings):
        end = next((p for p, lv in headings[n + 1:] if lv <= level), len(text))
        out.append((start, end, level))
    return out


def innermost_section(sections: list[tuple[int, int, int]], pos: int) -> tuple[int, int, int] | None:
    """The deepest section containing ``pos`` (the subsection it is in), or None before the first heading."""
    around = [s for s in sections if s[0] <= pos < s[1]]
    return max(around, key=lambda s: (s[2], s[0])) if around else None


class Sections:
    """Whole sections for the documentation chunks of a search; other chunks stay as they are."""

    def __init__(self, chunks: list[Chunk]):
        self.pages = page_texts(chunks)
        self.spans = {source_id: section_spans(text) for source_id, text in self.pages.items()}

    def of(self, hit: Hit) -> list[Hit]:
        c = hit.chunk
        if c.source_id not in self.pages:
            return [hit]
        page = self.pages[c.source_id]
        found = []
        for a, b in self.spans[c.source_id]:
            if a < c.end_char and b > c.start_char:  # the section overlaps the chunk: take it whole,
                if b - a > c.end_char - c.start_char:  # unless it is bigger than the chunk: then only
                    a, b = max(a, c.start_char), min(b, c.end_char)  # the part inside the chunk
                found.append(replace(hit, chunk=replace(c, chunk_id=f"{c.source_id}#{a}-{b}", text=page[a:b],
                                                        start_char=a, end_char=b)))
        return found or [hit]


def unique(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in hits:
        if h.chunk.chunk_id not in seen:
            seen.add(h.chunk.chunk_id)
            out.append(h)
    return out


class SectionReranker:
    """The candidates' documentation chunks become whole sections, and one model call ranks the
    sections and other candidates together (at most retrieval.section_candidates of them)."""

    def __init__(self, clients, settings, chunks: list[Chunk]):
        self.sections, self.ranker = Sections(chunks), LLMReranker(clients, settings)
        self.limit = settings.retrieval.section_candidates

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        units = unique([u for h in hits for u in self.sections.of(h)])[: self.limit]
        units = [replace(u, rank=n) for n, u in enumerate(units, start=1)]
        return self.ranker(case, query, units)


class SectionTrimmer:
    """The reranker keeps chunks as before; each kept documentation chunk is then replaced by its one
    section that best matches the report (report words weighted by rarity, as for focused snippets)."""

    def __init__(self, clients, settings, chunks: list[Chunk]):
        self.sections, self.ranker = Sections(chunks), LLMReranker(clients, settings)
        self.top_k, self.clean = settings.retrieval.top_k, CleanQuery(settings.context.max_query_chars)

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        ranked = self.ranker(case, query, hits)
        kept = ranked[: self.top_k]
        options = [self.sections.of(h) for h in kept]
        weights = term_weights(self.clean(case, {})[0], [u.chunk.text for group in options for u in group])
        def score(u):
            words = set(lexical_tokens(u.chunk.text))
            return sum(w for term, w in weights.items() if term in words)
        trimmed = unique([max(group, key=score) if len(group) > 1 else group[0] for group in options])
        return [replace(u, rank=n) for n, u in enumerate(trimmed, start=1)]  # only the trimmed sections
