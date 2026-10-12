"""Splitting sources into chunks. Evaluation issues, and everything attached to them, are left out
here, before anything is indexed."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..tokenizer import tokenize_with_spans
from ..types import Chunk
from .corpus import SourceRecord
from .structure_chunker import structure_spans
from .token_structure_chunker import CANDIDATE_A, token_structure_spans


@dataclass(frozen=True)
class Exclusions:
    """Issues (and everything attached to them) that must never reach the search index."""

    issue_numbers: frozenset[int] = field(default_factory=frozenset)
    source_ids: frozenset[str] = field(default_factory=frozenset)

    def drops(self, record: SourceRecord) -> bool:
        """An excluded issue takes everything attached to it: its comments, and linked pages that
        were linked from it (they may contain the answer the evaluation is testing for)."""
        numbers = self.issue_numbers
        return (record.source_id in self.source_ids
                or (record.parent_source_id or "") in self.source_ids
                or record.issue_number in numbers
                or bool(record.linked_issue_numbers & numbers))

    def digest(self) -> dict:
        return {"issue_numbers": len(self.issue_numbers), "source_ids": len(self.source_ids)}


def chunk_text(text: str, *, max_tokens: int, overlap: int) -> list[tuple[str, int, int]]:
    """Sliding token window. Returns (piece, start_char, end_char) with ``text[start:end] == piece``."""
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    spans = tokenize_with_spans(text)
    if not spans:
        return []
    out, start, step = [], 0, max_tokens - overlap
    while start < len(spans):
        end = min(start + max_tokens, len(spans))
        first, last = spans[start][1], spans[end - 1][2]
        out.append((text[first:last], first, last))
        if end == len(spans):
            break
        start += step
    return out


class FixedTokenChunker:
    """Fixed-size windows of ``max_tokens`` tokens overlapping by ``overlap_tokens``; each chunk is
    an exact substring of its source. The baseline chunker."""

    def __init__(self, *, max_tokens: int, overlap_tokens: int):
        self.max_tokens, self.overlap_tokens = max_tokens, overlap_tokens

    def __call__(self, record: SourceRecord) -> list[Chunk]:
        pieces = chunk_text(record.text, max_tokens=self.max_tokens, overlap=self.overlap_tokens)
        return [Chunk(chunk_id=f"{record.source_id}:{i:03d}", source_id=record.source_id,
                      source_type=record.source_type, title=record.title, text=piece, url=record.url,
                      parent_source_id=record.parent_source_id, issue_number=record.issue_number,
                      version=record.version, as_of=record.as_of, meta=dict(record.meta),
                      start_char=first, end_char=last, source_text_sha256=record.text_sha256)
                for i, (piece, first, last) in enumerate(pieces) if piece]


def _span_chunks(record: SourceRecord, pieces: list[tuple[str, int, int]],
                 paths: list[tuple[str, ...]] | None = None) -> list[Chunk]:
    """Same chunk records as the fixed chunker, built from exact source slices."""
    chunks = []
    for i, (piece, start, end) in enumerate(pieces):
        meta = dict(record.meta)
        if paths and paths[i]:
            meta["heading_path"] = list(paths[i])
        chunks.append(Chunk(chunk_id=f"{record.source_id}:{i:03d}", source_id=record.source_id,
                            source_type=record.source_type, title=record.title, text=piece,
                            url=record.url, parent_source_id=record.parent_source_id,
                            issue_number=record.issue_number, version=record.version,
                            as_of=record.as_of, meta=meta, start_char=start, end_char=end,
                            source_text_sha256=record.text_sha256))
    return chunks


class StructureChunker:
    """Heading- and fence-aware chunks: h1-h3; short sections merged forward, long ones split."""

    def __init__(self, min_chars: int = 150, max_chars: int = 2000):
        self.min_chars, self.max_chars = min_chars, max_chars

    def __call__(self, record: SourceRecord) -> list[Chunk]:
        spans = [(record.text[start:end], start, end, path)
                 for start, end, path in structure_spans(record.text, max_chars=self.max_chars,
                                                         min_chars=self.min_chars) if record.text[start:end]]
        return _span_chunks(record, [(piece, start, end) for piece, start, end, _ in spans],
                            [path for _, _, _, path in spans])


class TokenStructureChunker:
    """Candidate A: structural blocks packed to a token target (by default 500, soft max 700)."""

    def __init__(self, target: int = 500, soft_max: int = 700, min_section: int = 150):
        self.config = {**CANDIDATE_A, "target_tokens": target, "soft_max_tokens": soft_max,
                       "minimum_section_tokens": min_section}

    def __call__(self, record: SourceRecord) -> list[Chunk]:
        pieces = [(record.text[start:end], start, end)
                  for start, end in token_structure_spans(record.text, record.source_type, self.config)
                  if record.text[start:end]]
        return _span_chunks(record, pieces)


class ChunkContractError(ValueError):
    """A chunker produced a chunk that is not an exact, anchored span of its source."""


def check_chunks(records: list[SourceRecord], chunks: list[Chunk]) -> None:
    """The chunk contract, for any chunker: each chunk names its source, carries the source's hash,
    and ``source.text[start_char:end_char] == chunk.text``."""
    by_id = {r.source_id: r for r in records}
    for c in chunks:
        source = by_id.get(c.source_id)
        problem = ("unknown source" if source is None
                   else "missing offsets" if c.start_char is None or c.end_char is None
                   else "source hash differs" if c.source_text_sha256 != source.text_sha256
                   else "text is not source.text[start_char:end_char]"
                   if source.text[c.start_char:c.end_char] != c.text else None)
        if problem:
            raise ChunkContractError(f"chunk {c.chunk_id}: {problem}")


def build_chunks(records: list[SourceRecord], exclusions: Exclusions | None = None, *,
                 chunker) -> list[Chunk]:
    """Chunk every non-excluded record, in corpus order. ``chunker`` has no default on purpose."""
    exclusions = exclusions or Exclusions()
    return [chunk for record in records if not exclusions.drops(record) for chunk in chunker(record)]


def sources_without_chunks(records: list[SourceRecord], chunks: list[Chunk],
                           exclusions: Exclusions | None = None) -> list[str]:
    """Sources that produced no chunk (for example text with no word characters). Reported, not silent."""
    exclusions = exclusions or Exclusions()
    kept = {c.source_id for c in chunks}
    return sorted(r.source_id for r in records if not exclusions.drops(r) and r.source_id not in kept)


#: Chunkers by name, as selected by the chunking.method setting. To add one (for example an LLM
#: chunker), register it here and in config.CHUNKING_METHODS; the chunk node finds it there.
CHUNKERS = {
    "fixed": lambda settings: FixedTokenChunker(max_tokens=settings.chunking.max_tokens,
                                                overlap_tokens=settings.chunking.overlap_tokens),
    "structure": lambda settings: StructureChunker(settings.chunking.structure_min_chars,
                                                   settings.chunking.structure_max_chars),
    "token_structure": lambda settings: TokenStructureChunker(settings.chunking.token_target,
                                                              settings.chunking.token_soft_max,
                                                              settings.chunking.token_min_section),
}


def make_chunker(settings):
    return CHUNKERS[settings.chunking.method](settings)
