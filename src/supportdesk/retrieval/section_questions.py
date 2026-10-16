"""Document-side vocabulary: symptom questions written for each documentation section (doc2query).

A report describes a symptom in its own words; the documentation that explains it often uses words the
report never contains (forms, widget identity, widget life cycle). For each documentation section, an LLM
writes a few problem descriptions a user might write when that section is what they need; these are
embedded and searched as extra entry points that lead back to their section. Built once, offline, from the
documentation only (no evaluation case is involved).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..pipelines.parsing import NotJson, extract_json
from .index import IndexUnusable, VectorIndex
from .retriever import Hit

FILE, VECTORS, MANIFEST = "questions.jsonl", "vectors.npy", "manifest.json"


def generate_questions(sections: list[dict], texts: dict[str, str], chat, *, prompt: str, n: int = 5,
                       chars: int = 2500) -> list[list[str]]:
    """Up to ``n`` questions per section (one call each); a section whose answer is unusable gets none."""
    system = prompt.replace("{n}", str(n))
    out = []
    for s in sections:
        body = texts.get(s["source_id"], "")[s["start_char"]:s["end_char"]][:chars]
        user = f"Page: {s.get('title', '')}\nSection: {s.get('heading', '')}\n\n{body}"
        try:
            raw, _ = chat.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   stage="section_questions", json_mode=True)
            payload = extract_json(raw)
            found = payload.get("questions") if isinstance(payload, dict) else None
            out.append([" ".join(str(q).split()) for q in found if str(q).strip()][:n] if isinstance(found, list) else [])
        except (NotJson, ValueError, TypeError):
            out.append([])
    return out


class QuestionBank:
    """The sections, their questions and the questions' unit vectors."""

    def __init__(self, sections: list[dict], questions: list[list[str]], vectors: np.ndarray, model: str):
        self.sections, self.questions, self.model = sections, questions, model
        self.owner = [n for n, qs in enumerate(questions) for _ in qs]  # question row -> section
        vectors = np.asarray(vectors, dtype=np.float64).reshape(len(self.owner), -1) if self.owner else np.zeros((0, 1))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(vectors) else np.ones((0, 1))
        self.vectors = vectors / np.where(norms == 0, 1, norms)

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        d.joinpath(FILE).write_text("".join(json.dumps({**s, "questions": q}, ensure_ascii=False) + "\n"
                                            for s, q in zip(self.sections, self.questions)), encoding="utf-8")
        np.save(d / VECTORS, self.vectors)
        d.joinpath(MANIFEST).write_text(json.dumps({"embedding_model": self.model, "sections": len(self.sections),
                                                    "questions": len(self.owner)}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory, *, expect_model: str | None = None) -> QuestionBank:
        d = Path(directory)
        if not d.joinpath(FILE).exists():
            raise IndexUnusable(f"section questions {d} not found: run scripts/build_section_questions.py first")
        model = json.loads(d.joinpath(MANIFEST).read_text(encoding="utf-8"))["embedding_model"]
        if expect_model and model != expect_model:
            raise IndexUnusable(f"section questions {d} were embedded with {model}, not {expect_model}: rebuild them")
        rows = [json.loads(line) for line in d.joinpath(FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
        questions = [r.pop("questions") for r in rows]
        return cls(rows, questions, np.load(d / VECTORS), model)

    def ranking(self, query_vector) -> list[tuple[int, float]]:
        """(section, score) by each section's best-matching question, best first."""
        return [(section, score) for section, score, _ in self.matches(query_vector)]

    def matches(self, query_vector) -> list[tuple[int, float, str]]:
        """(section, score, the question that matched best), best first."""
        if not len(self.vectors):
            return []
        q = np.asarray(query_vector, dtype=np.float64)
        scores = self.vectors @ (q / (np.linalg.norm(q) or 1.0))
        flat = [question for qs in self.questions for question in qs]
        best: dict[int, tuple[float, str]] = {}
        for row, score in enumerate(scores):
            section = self.owner[row]
            if section not in best or float(score) > best[section][0]:
                best[section] = (float(score), flat[row])
        ranked = sorted(best.items(), key=lambda kv: (-round(kv[1][0], 6), kv[0]))
        return [(section, score, question) for section, (score, question) in ranked]


def section_chunk(index: VectorIndex, by_source: dict[str, list[int]], section: dict) -> int | None:
    """The index position of the chunk holding most of a section."""
    a, b = section["start_char"], section["end_char"]
    overlap = {i: max(0, min(b, index.chunks[i].end_char) - max(a, index.chunks[i].start_char))
               for i in by_source.get(section["source_id"], [])}
    best = max(overlap, key=overlap.get, default=None)
    return best if best is not None and overlap[best] > 0 else None


class QuestionExpandedRetriever:
    """The base retriever's candidates, all kept, plus the chunks of up to ``extra`` more sections whose
    symptom questions best match the query (sections whose chunk is already a candidate are skipped)."""

    name = "question_doc_slots"

    def __init__(self, base, bank: QuestionBank, *, extra: int, order: str = "append"):
        self.base, self.bank, self.extra, self.order = base, bank, extra, order
        self.index, self.embedder = base.index, base.embedder
        self.by_source: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.index.chunks):
            if c.start_char is not None and c.end_char is not None:
                self.by_source[c.source_id].append(i)

    def search(self, query, *, k, as_of=None):
        hits = list(self.base.search(query, k=k, as_of=as_of))
        have = {h.chunk.chunk_id for h in hits}
        vec, _ = self.embedder.embed([query], stage="query")  # cached: the base search embedded it already
        added = 0
        for section, score, question in self.bank.matches(vec[0]):
            if added >= self.extra:
                break
            pos = section_chunk(self.index, self.by_source, self.bank.sections[section])
            if pos is None or self.index.chunks[pos].chunk_id in have:
                continue
            have.add(self.index.chunks[pos].chunk_id)
            chunk = self.index.chunks[pos]  # carries the question that found it (the reranker may show it)
            chunk = dataclasses.replace(chunk, meta={**(chunk.meta or {}), "matched_question": question})
            hits.append(Hit(chunk, score, len(hits) + 1, "question"))
            added += 1
        return self._arrange(hits, query)

    def _arrange(self, hits: list[Hit], query: str) -> list[Hit]:
        """The candidates' order for the reranker: append (the base's first, then the question-found),
        interleave (one of each in turn) or shuffle (an order that says nothing about where a candidate came
        from; reproducible, seeded by the query)."""
        if self.order == "interleave":
            base = [h for h in hits if h.source != "question"]
            found = [h for h in hits if h.source == "question"]
            mixed = []
            for n in range(max(len(base), len(found))):
                mixed += base[n:n + 1] + found[n:n + 1]
            hits = mixed
        elif self.order == "shuffle":
            hits = list(hits)
            random.Random(int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:12], 16)).shuffle(hits)
        return [dataclasses.replace(h, rank=n) for n, h in enumerate(hits, start=1)]


def build_question_retriever(index: VectorIndex, clients, settings) -> QuestionExpandedRetriever:
    from .retriever import DocSlotsRetriever

    bank = QuestionBank.load(settings.paths.section_questions, expect_model=clients.embedding_model)
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=settings.retrieval.doc_slots)
    return QuestionExpandedRetriever(base, bank, extra=settings.retrieval.question_sections,
                                     order=settings.retrieval.question_order)


__all__ = ["QuestionBank", "QuestionExpandedRetriever", "build_question_retriever", "generate_questions",
           "section_chunk"]
