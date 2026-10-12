"""Token-budgeted structural chunking, candidate A from the chunking folder.

Copied from scripts/markdown_structure.py (markdown_blocks) and scripts/token_chunking.py
(parse_blocks, split_prose, overlap_atom, ranges). Token counts use the app's tokenizer
instead of tiktoken. Sizes are candidate A's: target 500, soft max 700, overlap 15-60.
"""
from __future__ import annotations

import re

from ..tokenizer import count_tokens

# Candidate A (config/A.json).
CANDIDATE_A = {
    "target_tokens": 500,
    "soft_max_tokens": 700,
    "minimum_section_tokens": 150,
    "overlap_max_tokens": 60,
    "minimum_overlap_tokens": 15,
    "short_context_before_atomic_tokens": 150,
}


def markdown_blocks(text):
    """Return a complete contiguous partition, plus unmodified fenced spans."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks, fences, headings = [], [], []
    i = 0
    while i < len(lines):
        a = i
        line = lines[i]
        opener = re.match(r"^\s*(`{3,}|~{3,})([^\r\n]*)", line)
        if opener:
            marker = opener[1]
            close = re.compile(r"^\s*" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}[ \t]*(?:\r?\n)?$")
            i += 1
            while i < len(lines):
                if close.match(lines[i]):
                    i += 1
                    break
                i += 1
            fences.append((offsets[a], offsets[i]))
            kind = "fenced_code"
        elif not line.strip():
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            kind = "whitespace"
        elif re.match(r"^ {0,3}#{1,6}[ \t]+", line):
            m = re.match(r"^ {0,3}(#{1,6})[ \t]+(.*)", line.rstrip("\r\n"))
            headings.append({"start_char": offsets[i], "end_char": offsets[i + 1], "level": len(m[1]),
                             "text": m[2].rstrip("#").strip()})
            i += 1
            kind = "heading"
        elif i + 1 < len(lines) and "|" in line and re.match(r"^\s*\|?\s*:?-{3,}:?\s*\|", lines[i + 1]):
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                i += 1
            kind = "table"
        elif re.match(r"^( {4,}|\t)\S", line):
            i += 1
            while i < len(lines) and (not lines[i].strip() or re.match(r"^( {4,}|\t)", lines[i])):
                i += 1
            kind = "indented_block"
        else:
            i += 1
            while i < len(lines) and lines[i].strip():
                if (re.match(r"^\s*(`{3,}|~{3,})", lines[i]) or re.match(r"^ {0,3}#{1,6}[ \t]+", lines[i])
                        or re.match(r"^( {4,}|\t)\S", lines[i])):
                    break
                if i + 1 < len(lines) and "|" in lines[i] and re.match(r"^\s*\|?\s*:?-{3,}:?\s*\|", lines[i + 1]):
                    break
                i += 1
            kind = "paragraph"
        blocks.append({"start": offsets[a], "end": offsets[i], "kind": kind})
    merged = []
    for block in blocks:
        if block["kind"] == "whitespace" and merged:
            merged[-1]["end"] = block["end"]
        else:
            merged.append(block)
    return merged, fences, headings


def parse_blocks(text):
    blocks, fences, headings = markdown_blocks(text)
    for block in blocks:
        raw = text[block["start"]:block["end"]].strip()
        label = re.fullmatch(r"\*\*([^\n*]{1,150})\*\*", raw)
        if block["kind"] == "paragraph" and label:
            block["kind"] = "attached_label"
            headings.append({"start_char": block["start"], "end_char": block["end"], "level": 4,
                             "text": label[1], "origin": "standalone_bold_label"})
    headings.sort(key=lambda h: h["start_char"])
    return blocks, fences, headings


def split_prose(text, block, config):
    if block["kind"] != "paragraph":
        return [block]
    a, end = block["start"], block["end"]
    result = []
    target, maximum = config["target_tokens"], config["soft_max_tokens"]
    while count_tokens(text[a:end]) > maximum:
        options = [a + m.end() for m in re.finditer(r"[.!?][ \t\r\n]+", text[a:end])]
        scored = [(abs(n - target), point) for point in options
                  if target // 2 <= (n := count_tokens(text[a:point])) <= maximum]
        if not scored:
            options = [a + m.end() for m in re.finditer(r"\s+", text[a:end])]
            scored = [(abs(n - target), point) for point in options
                      if target // 2 <= (n := count_tokens(text[a:point])) <= maximum]
        if scored:
            split = min(scored)[1]
        else:
            low, high = a + 1, end
            while low < high:
                mid = (low + high + 1) // 2
                if count_tokens(text[a:mid]) <= target:
                    low = mid
                else:
                    high = mid - 1
            split = low
            while split > a + 1 and count_tokens(text[a:split]) > target:
                split -= 1
        assert a < split < end and count_tokens(text[a:split]) <= maximum
        result.append({"start": a, "end": split, "kind": "paragraph"})
        a = split
    result.append({"start": a, "end": end, "kind": "paragraph"})
    return result


def overlap_atom(text, last, config):
    if last["kind"] != "paragraph":
        return None
    a, b = last["start"], last["end"]
    boundaries = [a] + [a + m.end() for m in re.finditer(r"[.!?]\s+", text[a:b])]
    for start in boundaries:
        n = count_tokens(text[start:b])
        if config["minimum_overlap_tokens"] <= n <= config["overlap_max_tokens"] and text[start:b].strip():
            return {"start": start, "end": b, "kind": "paragraph"}
    return None


def ranges(text, config, source_type="documentation"):
    blocks, fences, headings = parse_blocks(text)
    if count_tokens(text) <= config["soft_max_tokens"]:
        return [(0, len(text), "whole_short_source")], blocks, fences, headings
    atoms = [p for block in blocks for p in split_prose(text, block, config)]
    result, current, novel = [], [], False
    heading_levels = {h["start_char"]: h["level"] for h in headings}
    previous_heading_level = None

    def size():
        return count_tokens(text[current[0]["start"]:current[-1]["end"]]) if current else 0

    def flush(reason, allow_overlap):
        nonlocal current, novel
        if not current or not novel:
            current = []
            novel = False
            return
        a, b = current[0]["start"], current[-1]["end"]
        last = current[-1]
        result.append((a, b, reason))
        current = []
        novel = False
        if allow_overlap:
            tail = overlap_atom(text, last, config)
            if tail and tail["start"] > a:
                current = [tail]

    for i, block in enumerate(atoms):
        if block["kind"] == "heading":
            level = heading_levels[block["start"]]
            hard_boundary = (source_type == "documentation" and previous_heading_level is not None
                             and level <= 3 and level <= previous_heading_level)
            if current and not novel:
                current = []
            elif current and (hard_boundary or size() >= config["minimum_section_tokens"]):
                flush("main_topic_boundary" if hard_boundary else "section_boundary", False)
            previous_heading_level = level
        short_atomic_context = (current and novel and size() < config["short_context_before_atomic_tokens"]
                                and block["kind"] in ["fenced_code", "indented_block", "table"])
        if current and count_tokens(text[current[0]["start"]:block["end"]]) > config["soft_max_tokens"] and not short_atomic_context:
            pending = []
            while current and current[-1]["kind"] in ["heading", "attached_label"]:
                pending.insert(0, current.pop())
            flush("before_atomic_block_or_size_limit", block["kind"] == "paragraph" and not pending)
            if pending:
                current = pending
                novel = True
            if current and count_tokens(text[current[0]["start"]:block["end"]]) > config["soft_max_tokens"]:
                if pending and block["kind"] == "paragraph":
                    combined = {"start": pending[0]["start"], "end": block["end"], "kind": "paragraph"}
                    parts = split_prose(text, combined, config)
                    block = parts[0]
                    atoms[i + 1:i + 1] = parts[1:]
                    current = []
                elif not pending:
                    current = []
        current.append(block)
        novel = True
        if size() >= config["target_tokens"] and block["kind"] not in ["heading", "attached_label"]:
            next_heading = i + 1 < len(atoms) and atoms[i + 1]["kind"] == "heading"
            flush("target_size_or_atomic_block_end", not next_heading)
    flush("source_end", False)
    if (len(result) > 1 and count_tokens(text[result[-1][0]:result[-1][1]]) < config["minimum_section_tokens"]
            and count_tokens(text[result[-2][0]:result[-1][1]]) <= config["soft_max_tokens"]):
        result[-2:] = [(result[-2][0], result[-1][1], "merged_small_tail")]
    return result, blocks, fences, headings


def token_structure_spans(text: str, source_type: str = "documentation",
                          config: dict | None = None) -> list[tuple[int, int]]:
    """Return (start, end) spans for candidate A (or ``config``). ``text[start:end]`` is the chunk text."""
    if not text:
        return []
    spans, _, _, _ = ranges(text, config or CANDIDATE_A, source_type)
    return [(a, b) for a, b, _reason in spans if b > a]
