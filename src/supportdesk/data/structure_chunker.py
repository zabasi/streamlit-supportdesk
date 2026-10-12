"""Heading- and code-fence-aware chunking, the from-scratch winner of the chunking study.

Cuts are the study's ``scratch`` (chunking_study_v06/chunkers/structure.py): headings of level
h1-h3 outside code fences, sections shorter than 150 characters merged forward, sections longer
than 2,000 characters packed from whole blocks. Offsets are Python code points, end exclusive.
Returned spans tile the text. The heading path is metadata and is never inserted into the text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")


@dataclass
class _Span:
    start: int
    end: int
    heading_path: tuple[str, ...] = ()


def _iter_lines(text: str):
    pos, n = 0, len(text)
    while pos < n:
        nl = text.find("\n", pos)
        end = n if nl == -1 else nl + 1
        yield pos, end, text[pos:end].rstrip("\n").rstrip("\r")
        pos = end


def _scan_markdown(text: str):
    """Headings outside fences, a fence flag per line, and lines as (start, end, content)."""
    headings, lines, infence = [], [], []
    open_char = open_len = None
    for start, end, line in _iter_lines(text):
        lines.append((start, end, line))
        if open_char is None:
            m = _FENCE_OPEN.match(line)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                open_char, open_len = m.group(1)[0], len(m.group(1))
                infence.append(True)
                continue
            infence.append(False)
            h = _ATX.match(line)
            if h:
                headings.append((start, len(h.group(1)), (h.group(2) or "").strip()))
        else:
            infence.append(True)
            m = _FENCE_CLOSE.match(line)
            if m and m.group(1)[0] == open_char and len(m.group(1)) >= open_len:
                open_char = open_len = None
    return headings, infence, lines


def _is_atx_heading(line: str) -> bool:
    return _ATX.match(line) is not None


def _sections(text: str, level: int, headings):
    cuts = [(s, lv, title) for s, lv, title in headings if lv <= level]
    bounds = [0] + [s for s, _, _ in cuts if s > 0] + [len(text)]
    stack: list[tuple[int, str]] = []
    path_at: dict[int, tuple[str, ...]] = {}
    for s, lv, title in cuts:
        while stack and stack[-1][0] >= lv:
            stack.pop()
        stack.append((lv, title))
        path_at[s] = tuple(t for _, t in stack)
    return [[a, b, path_at.get(a, ())] for a, b in pairwise(bounds) if b > a]


def _merge_small(secs, text: str, min_chars: int):
    if min_chars <= 0 or len(secs) <= 1:
        return secs
    merged: list[list] = []
    carry = None
    for a, b, path in secs:
        if carry is not None:
            a, path = carry[0], (carry[2] or path)
            carry = None
        if len(text[a:b].strip()) < min_chars:
            carry = [a, b, path]
            continue
        merged.append([a, b, path])
    if carry is not None:
        if merged:
            merged[-1][1] = carry[1]
        else:
            merged.append(carry)
    return merged


def _blocks(a: int, b: int, lines, infence):
    blocks: list[tuple[int, int]] = []
    cur = None
    after_heading = False
    for i, (s, e, content) in enumerate(lines):
        if s < a or e > b:
            continue
        in_fence = infence[i]
        if not in_fence and _is_atx_heading(content):
            if cur is not None:
                blocks.append((cur, s))
            cur, after_heading = s, True
            continue
        if not in_fence and not content.strip():
            if after_heading:
                continue
            if cur is not None:
                blocks.append((cur, e))
                cur = None
            continue
        if cur is None:
            cur = s
        after_heading = False
    if cur is not None:
        blocks.append((cur, b))
    tiled, pos = [], a
    for _, e in blocks:
        if e > pos:
            tiled.append((pos, e))
            pos = e
    if pos < b:
        if tiled:
            tiled[-1] = (tiled[-1][0], b)
        else:
            tiled.append((a, b))
    return tiled


def _split_big_block(text: str, a: int, b: int, max_chars: int):
    pieces, pos = [], a
    while b - pos > max_chars:
        limit = pos + max_chars
        nl = text.rfind("\n", pos + 1, limit)
        if nl != -1:
            cut = nl + 1
        else:
            sp = text.rfind(" ", pos + 1, limit)
            cut = sp + 1 if sp != -1 else limit
        pieces.append((pos, cut))
        pos = cut
    if pos < b:
        pieces.append((pos, b))
    return pieces


def _absorb_blank(spans: list[_Span], text: str) -> list[_Span]:
    out: list[_Span] = []
    for sp in spans:
        if out and not text[sp.start:sp.end].strip():
            out[-1].end = sp.end
        else:
            out.append(sp)
    return out


def structure_spans(text: str, level: int = 3, max_chars: int = 2000, min_chars: int = 150
                    ) -> list[tuple[int, int, tuple[str, ...]]]:
    """Return (start, end, heading_path) spans. ``text[start:end]`` is the chunk text."""
    if not text:
        return []
    headings, infence, lines = _scan_markdown(text)
    secs = _merge_small(_sections(text, level, headings), text, min_chars)
    spans: list[_Span] = []
    for a, b, path in secs:
        if b - a <= max_chars:
            spans.append(_Span(a, b, path))
            continue
        cur_a = cur_b = None
        for ba, bb in _blocks(a, b, lines, infence):
            parts = [(ba, bb)] if bb - ba <= max_chars else _split_big_block(text, ba, bb, max_chars)
            for pa, pb in parts:
                if cur_a is None:
                    cur_a, cur_b = pa, pb
                elif pb - cur_a <= max_chars:
                    cur_b = pb
                else:
                    spans.append(_Span(cur_a, cur_b, path))
                    cur_a, cur_b = pa, pb
        if cur_a is not None:
            spans.append(_Span(cur_a, cur_b, path))
    return [(sp.start, sp.end, sp.heading_path) for sp in _absorb_blank(spans, text)]
