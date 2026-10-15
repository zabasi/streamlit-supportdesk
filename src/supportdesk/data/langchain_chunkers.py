"""LangChain chunkers for the hierarchical-class RAG experiment.

Rule: ``MarkdownHeaderTextSplitter`` (documentation / release notes) then
``RecursiveCharacterTextSplitter`` on long leftover sections. LLM: the chat model
proposes split offsets for documentation only; issues and comments stay on the rule
chunker. Offline or a bad model answer falls back to an embedding breakpoint pack,
then to the rule chunker. Every chunk is still an exact source span.
"""
from __future__ import annotations

import re

from ..pipelines.parsing import NotJson, extract_json
from ..types import Chunk
from .chunking import _span_chunks
from .corpus import SourceRecord

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError as exc:  # the experiment extra; the rest of the project does not need it
    MarkdownHeaderTextSplitter = RecursiveCharacterTextSplitter = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
MARKDOWN_TYPES = {"documentation", "release_section", "linked_reference"}
LLM_TYPES = {"documentation"}


def _require_langchain() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError("pip install 'langchain-text-splitters>=0.3,<1'") from _IMPORT_ERROR


def _splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    _require_langchain()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def _locate(text: str, piece: str, start_at: int = 0) -> int:
    if not piece:
        return -1
    found = text.find(piece, start_at)
    return found if found >= 0 else text.find(piece)


def _recursive_spans(text: str, *, start: int, chunk_size: int, overlap: int,
                     path: tuple[str, ...] = ()) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """Split ``text`` (a source slice that starts at ``start``) and return source offsets."""
    if len(text) <= chunk_size:
        return [(text, start, start + len(text), path)] if text else []
    out, cursor = [], 0
    for piece in _splitter(chunk_size, overlap).split_text(text):
        at = _locate(text, piece, cursor)
        if at < 0:
            continue
        out.append((piece, start + at, start + at + len(piece), path))
        cursor = at + max(1, len(piece) - overlap)
    return out


def rule_spans(text: str, source_type: str, *, chunk_size: int = 1600,
               overlap: int = 200) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """Exact (piece, start, end, heading_path) spans from the LangChain rule chunker."""
    _require_langchain()
    if not text.strip():
        return []
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    if source_type in MARKDOWN_TYPES and ("\n#" in text or text.startswith("#")):
        header = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
        cursor = 0
        for doc in header.split_text(text):
            piece = doc.page_content
            at = _locate(text, piece, cursor)
            if at < 0:
                continue
            path = tuple(str(v) for v in doc.metadata.values()) if doc.metadata else ()
            sections.append((at, at + len(piece), path))
            cursor = at + 1
    if not sections:
        sections = [(0, len(text), ())]
    out: list[tuple[str, int, int, tuple[str, ...]]] = []
    for start, end, path in sections:
        out.extend(_recursive_spans(text[start:end], start=start, chunk_size=chunk_size,
                                    overlap=overlap, path=path))
    seen, unique = set(), []
    for piece, start, end, path in out:
        if start < end and (start, end) not in seen:
            seen.add((start, end))
            unique.append((piece, start, end, path))
    return unique


def _snap(text: str, pos: int) -> int:
    pos = max(0, min(len(text), int(pos)))
    if pos in (0, len(text)):
        return pos
    window = text[max(0, pos - 80): min(len(text), pos + 80)]
    origin = max(0, pos - 80)
    heading = re.search(r"\n#{1,3}[ \t]", window)
    if heading:
        return origin + heading.start() + 1
    blank = window.find("\n\n")
    if blank >= 0:
        return origin + blank + 2
    nl = window.rfind("\n")
    return origin + nl + 1 if nl >= 0 else pos


def _paragraphs(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for part in re.split(r"(?<=\n\n)", text):
        if part:
            out.append((part, pos, pos + len(part)))
        pos += len(part)
    return out or [(text, 0, len(text))]


def semantic_spans(text: str, embedder, *, chunk_size: int = 1600,
                   threshold: float = 0.35) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """Pack consecutive paragraphs until size or an embedding-similarity drop."""
    paras = _paragraphs(text)
    if len(paras) == 1:
        return _recursive_spans(text, start=0, chunk_size=chunk_size, overlap=0)
    vectors, _ = embedder.embed([p for p, _, _ in paras], stage="semantic_chunk")
    out, buf, start = [], paras[0][0], paras[0][1]
    for i in range(1, len(paras)):
        piece, lo, hi = paras[i]
        prev, cur = vectors[i - 1], vectors[i]
        dot = sum(a * b for a, b in zip(prev, cur))
        if len(buf) + len(piece) > chunk_size or dot < threshold:
            out.append((buf, start, start + len(buf), ()))
            buf, start = piece, lo
        else:
            buf += piece
    if buf:
        out.append((buf, start, start + len(buf), ()))
    return [(p, a, b, path) for p, a, b, path in out if b > a]


def llm_spans(text: str, chat, *, prompt: str, fallback) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """LLM-proposed cuts; ``fallback`` is called with the source text on failure."""
    try:
        raw, _ = chat.complete(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": text[:24000]}],
            stage="llm_chunk", json_mode=True)
        payload = extract_json(raw)
        cuts = payload.get("cuts") if isinstance(payload, dict) else None
        if not isinstance(cuts, list):
            raise ValueError("no cuts")
        points = sorted({0, len(text), *(_snap(text, c) for c in cuts if isinstance(c, (int, float)))})
        points = [p for p in points if 0 <= p <= len(text)]
        if len(points) < 2:
            raise ValueError("too few cuts")
        return [(text[a:b], a, b, ()) for a, b in zip(points, points[1:]) if b > a]
    except (NotJson, ValueError, TypeError, OSError):
        return fallback(text)


class LangChainRuleChunker:
    """LangChain heading splitter + recursive character splitter."""

    def __init__(self, *, chunk_size: int = 1600, overlap: int = 200):
        self.chunk_size, self.overlap = chunk_size, overlap

    def spans(self, record: SourceRecord):
        return rule_spans(record.text, record.source_type, chunk_size=self.chunk_size,
                          overlap=self.overlap)

    def __call__(self, record: SourceRecord) -> list[Chunk]:
        rows = self.spans(record)
        return _span_chunks(record, [(p, a, b) for p, a, b, _ in rows],
                            [path for *_, path in rows])


class LangChainLLMChunker:
    """LLM split points on documentation; the rule chunker on everything else."""

    def __init__(self, chat, *, prompt: str, embedder=None, chunk_size: int = 1600,
                 overlap: int = 200):
        self.chat, self.prompt, self.embedder = chat, prompt, embedder
        self.rule = LangChainRuleChunker(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size

    def spans(self, record: SourceRecord):
        if record.source_type not in LLM_TYPES:
            return self.rule.spans(record)

        def fallback(text: str):
            if self.embedder is not None:
                return semantic_spans(text, self.embedder, chunk_size=self.chunk_size)
            return rule_spans(text, record.source_type, chunk_size=self.chunk_size,
                              overlap=self.rule.overlap)

        return llm_spans(record.text, self.chat, prompt=self.prompt, fallback=fallback)

    def __call__(self, record: SourceRecord) -> list[Chunk]:
        rows = self.spans(record)
        return _span_chunks(record, [(p, a, b) for p, a, b, _ in rows],
                            [path for *_, path in rows])


__all__ = ["LangChainLLMChunker", "LangChainRuleChunker", "llm_spans", "rule_spans", "semantic_spans"]
