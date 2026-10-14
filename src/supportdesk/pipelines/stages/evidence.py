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
            from .rewrite_query import make_query_rewriter
            self.concepts, self.skip = make_query_rewriter(clients, settings), 1 if r.keep_report_query else 0

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
        if self.mode == "branch":  # the found chunk's ## parent and all its children, by heading path
            path = tuple(chunk.meta.get("heading_path") or ())
            if len(path) < 2:  # the page's introduction (or no path): no branch to add
                return []
            page = [c for c in page if tuple(c.meta.get("heading_path") or ())[:2] == path[:2]]
        elif self.mode == "section" and chunk.source_id in self.sections.spans:
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



class CorrectiveReranker:
    """Corrective retrieval on top of another reranker (by default advanced's ExpandingReranker).
    After reranking, a judge (prompts.judge) reads the kept chunks and answers yes or no. On no, a model
    writes one new query (prompts.loop_rewrite), the retriever searches with it, and the old and new
    candidates are reranked together, so the best chunks of both rounds can be kept. At most
    retrieval.corrective_turns times; 0 does nothing extra. Chunks from a corrective search are marked
    "corrective"; each case's decisions are left in ``last_trace`` for the run's results.
    With retrieval.corrective_judge = coverage, the judge (prompts.judge_coverage) instead lists the concept
    queries ("mechanisms") no kept passage explains, and one new query is written for each (prompts.
    rewrite_missing, at most corrective_max_missing); without concept queries it falls back to relevance."""

    def __init__(self, clients, settings, chunks: list[Chunk], retriever, inner=None):
        self.inner = inner or ExpandingReranker(clients, settings, chunks)
        self.clients, self.retriever, self.settings = clients, retriever, settings
        self.turns = settings.retrieval.corrective_turns
        self.clean = CleanQuery(settings.context.max_query_chars)
        self.judge_prompt = Path(settings.prompts.judge).read_text(encoding="utf-8")
        self.rewrite_prompt = Path(settings.prompts.loop_rewrite).read_text(encoding="utf-8")
        self.last_trace: dict | None = None
        r = settings.retrieval
        self.mode, self.max_missing = r.corrective_judge, r.corrective_max_missing
        if self.mode == "coverage":  # the concept queries name the mechanisms (that call replays from the cache)
            from .rewrite_query import make_query_rewriter
            self.concepts, self.skip = make_query_rewriter(clients, settings), 1 if r.keep_report_query else 0
            self.coverage_prompt = Path(settings.prompts.judge_coverage).read_text(encoding="utf-8")
            self.missing_prompt = Path(settings.prompts.rewrite_missing).read_text(encoding="utf-8")

    @staticmethod
    def passages(hits: list[Hit], limit: int) -> str:
        return "\n\n".join(f"[{h.chunk.chunk_id}] {h.chunk.title}\n{h.chunk.text[:limit]}" for h in hits) \
            or "(no passages)"

    def judge(self, question: str, kept: list[Hit]) -> str:
        messages = [{"role": "system", "content": self.judge_prompt},
                    {"role": "user", "content": f"Question:\n{question}\n\nPassages:\n{self.passages(kept, 1200)}"}]
        text, _ = self.clients.chat.complete(messages, stage="judge", json_mode=False)
        return "yes" if text.strip().lower().startswith("yes") else "no"

    def rewrite(self, question: str, previous: list[str], kept: list[Hit]) -> str:
        messages = [{"role": "system", "content": self.rewrite_prompt},
                    {"role": "user", "content": f"Question:\n{question}\n\nPrevious queries:\n" + "\n".join(previous)
                     + f"\n\nPassages already retrieved:\n{self.passages(kept, 400)}"}]
        text, _ = self.clients.chat.complete(messages, stage="loop_rewrite", json_mode=False)
        return " ".join(text.split()) or " ".join(question.split())

    def missing(self, question: str, mechanisms: list[str], kept: list[Hit]) -> list[str] | None:
        """The mechanisms no kept passage explains; None if the judge's answer is unusable."""
        listing = "\n".join(f"M{n}: {m}" for n, m in enumerate(mechanisms, start=1))
        messages = [{"role": "system", "content": self.coverage_prompt},
                    {"role": "user", "content": f"## Report\n{question}\n\n## Mechanisms involved\n{listing}\n\n"
                                                f"## Passages\n{self.passages(kept, 1200)}"}]
        text, _ = self.clients.chat.complete(messages, stage="judge", json_mode=True)
        try:
            payload = extract_json(text)
        except (NotJson, ValueError):
            return None
        labels = payload.get("missing") if isinstance(payload, dict) else None
        if not isinstance(labels, list):
            return None
        found = []
        for label in labels:
            match = re.match(r"^M(\d+)$", str(label).strip())
            if match and 1 <= int(match.group(1)) <= len(mechanisms) and mechanisms[int(match.group(1)) - 1] not in found:
                found.append(mechanisms[int(match.group(1)) - 1])
        return found

    def query_for(self, question: str, mechanism: str, previous: list[str]) -> str:
        messages = [{"role": "system", "content": self.missing_prompt},
                    {"role": "user", "content": f"## Report\n{question}\n\n## Missing mechanism\n{mechanism}\n\n"
                                                "## Previous queries\n" + "\n".join(previous)}]
        text, _ = self.clients.chat.complete(messages, stage="loop_rewrite", json_mode=False)
        return " ".join(text.split()) or mechanism

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        kept = self.inner(case, query, hits)
        if not self.turns:
            self.last_trace = None
            return kept
        question = self.clean(case, {})[0]
        trace: dict = {"judgements": [], "queries": [], "turns": 0}
        pool, previous = list(hits), [query]
        mechanisms = []
        if self.mode == "coverage":
            queries = self.concepts(case, {})
            mechanisms, previous = queries[self.skip:], [query, *queries]
        for _ in range(self.turns):
            if self.mode == "coverage" and mechanisms:
                gaps = self.missing(question, mechanisms, kept)
                trace["judgements"].append({"missing": gaps} if gaps is not None else "unusable")
                if not gaps:  # nothing missing, or an unusable answer: stop
                    break
                new_queries = [self.query_for(question, m, previous) for m in gaps[: self.max_missing]]
            else:
                verdict = self.judge(question, kept)
                trace["judgements"].append(verdict)
                if verdict == "yes":
                    break
                new_queries = [self.rewrite(question, previous, kept)]
            as_of = case.created_at if self.settings.retrieval.historical_filter else None
            for new_query in new_queries:
                previous.append(new_query)
                trace["queries"].append(new_query)
                found = self.retriever.search(new_query, k=max(len(hits), self.settings.retrieval.top_k), as_of=as_of)
                seen = {h.chunk.chunk_id for h in pool}
                pool += [replace(h, source="corrective") for h in found if h.chunk.chunk_id not in seen]
            pool = [replace(h, rank=n) for n, h in enumerate(pool, start=1)]
            kept = self.inner(case, query, pool)
            trace["turns"] += 1
        self.last_trace = trace
        return kept



class SummaryReranker:
    """Hierarchical retrieval through section summaries. Before reranking, the queries (the report and
    the concept queries; those calls replay from the cache) are compared with a summary of every
    documentation section (scripts/summarize_sections.py); the fixed chunks of the
    retrieval.summary_sections best-matching sections are added to the candidates, marked "summary",
    if they are not there yet and are visible at the case's date. Then ``inner`` reranks as usual."""

    def __init__(self, clients, settings, chunks: list[Chunk], inner=None):
        import json

        from ...retrieval.filters import parse_ts
        from .rewrite_query import make_query_rewriter
        self.inner = inner or ExpandingReranker(clients, settings, chunks)
        self.n, self.clients, self.settings, self.parse_ts = settings.retrieval.summary_sections, clients, settings, parse_ts
        self.by_page: dict[str, list[Chunk]] = {}
        for c in chunks:
            if c.source_type == "documentation" and c.start_char is not None:
                self.by_page.setdefault(c.source_id, []).append(c)
        self.sections, self.vectors = [], []
        if self.n:
            path = Path(settings.paths.summaries)
            if not path.exists():  # like an index that is not built: commands report it, snapshot --all skips it
                from ...retrieval.index import IndexUnusable
                raise IndexUnusable(f"section summaries {path} not found: run scripts/summarize_sections.py first")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.sections = [r for r in rows if r["source_id"] in self.by_page]
            texts = [f"{r['title']} > {r['heading']}: {r['summary']}" for r in self.sections]
            self.vectors = self.clients.embedding.embed(texts, stage="summary")[0] if texts else []
            self.queries = make_query_rewriter(clients, settings)

    def chunks_of(self, section: dict, as_of: str | None) -> list[Chunk]:
        cutoff = self.parse_ts(as_of)
        found = []
        for c in self.by_page.get(section["source_id"], []):
            if c.start_char < section["end_char"] and c.end_char > section["start_char"]:
                stamp = self.parse_ts(c.as_of)
                if cutoff is None or stamp is None or stamp < cutoff:
                    found.append(c)
        return found

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        if not self.n or not self.sections:
            return self.inner(case, query, hits)
        qvecs = [self.clients.embedding.embed([q], stage="query")[0][0] for q in self.queries(case, {})]
        def score(i):
            return max(sum(a * b for a, b in zip(q, self.vectors[i])) for q in qvecs)
        order = sorted(range(len(self.sections)), key=lambda i: (-score(i), i))
        as_of = case.created_at if self.settings.retrieval.historical_filter else None
        seen, added, used = {h.chunk.chunk_id for h in hits}, [], 0
        for i in order:
            if used >= self.n:
                break
            new = [c for c in self.chunks_of(self.sections[i], as_of) if c.chunk_id not in seen]
            if not new:
                continue
            used += 1
            for c in new:
                seen.add(c.chunk_id)
                added.append(Hit(c, score(i), 0, "summary"))
        pool = [replace(h, rank=n) for n, h in enumerate(hits + added, start=1)]
        return self.inner(case, query, pool)
