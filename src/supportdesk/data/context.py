"""Contextual chunk headers: what a documentation chunk is about, from its page and headings.

A section such as "Caveats and limitations" rarely repeats the topic of its page; the page title and
the headings above it carry that. With ``chunking.context_header`` the indexer embeds each
documentation chunk as "<page title> › <heading> › <subheading>" followed by the chunk's text. Only
the vector changes: the stored chunk text, its offsets and what the model reads stay the same.
"""
from __future__ import annotations

import re

from ..types import Chunk

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.+?)[ \t#]*$")
SEPARATOR = " › "


def heading_path(text: str, before: int) -> list[str]:
    """The level 1-3 headings that ``text[before]`` sits under, outermost first (code fences skipped)."""
    path: list[tuple[int, str]] = []
    fence = None
    pos = 0
    for line in text[:before].splitlines(keepends=True):
        pos += len(line)
        opener = _FENCE.match(line)
        if fence:
            if opener and opener.group(1)[0] == fence:
                fence = None
            continue
        if opener:
            fence = opener.group(1)[0]
            continue
        heading = _HEADING.match(line.rstrip("\n"))
        if heading:
            level = len(heading.group(1))
            path = [p for p in path if p[0] < level] + [(level, heading.group(2).strip())]
    return [title for _, title in path]


def _after_leading_headings(chunk: Chunk) -> int:
    """Where the chunk's own leading heading lines end, so a chunk that opens with "### X" is under X."""
    start = chunk.start_char or 0
    offset = 0
    for line in chunk.text.splitlines(keepends=True):
        if line.strip() and not _HEADING.match(line.rstrip("\n")):
            break
        offset += len(line)
    return start + offset


def context_header(chunk: Chunk, source_text: str) -> str:
    """"<page title> › <headings above the chunk>" for a documentation chunk; "" for other chunks."""
    if chunk.source_type != "documentation":
        return ""
    parts = [chunk.title.strip()] if chunk.title and chunk.title.strip() else []
    for heading in heading_path(source_text, _after_leading_headings(chunk)):
        if not parts or heading != parts[-1]:
            parts.append(heading)
    return SEPARATOR.join(parts)


def embedding_texts(chunks: list[Chunk], sources: dict[str, str], enabled: bool) -> list[str]:
    """What the indexer embeds: the chunk text, with its context header in front when ``enabled``."""
    if not enabled:
        return [c.text for c in chunks]
    texts = []
    for c in chunks:
        header = context_header(c, sources.get(c.source_id, ""))
        texts.append(f"{header}\n\n{c.text}" if header else c.text)
    return texts
