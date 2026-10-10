"""Leakage audit: nothing in the index may reveal an evaluation case's answer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..tokenizer import lexical_tokens
from ..types import Chunk
from .cases import EvalCase


@dataclass
class LeakageReport:
    violations: list[str] = field(default_factory=list)  # hard failures: must be empty
    warnings: list[str] = field(default_factory=list)  # needs a human look

    @property
    def ok(self) -> bool:
        return not self.violations


def _shingles(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0))}


def check_leakage(chunks: list[Chunk], cases: list[EvalCase], *, overlap_threshold: float,
                  shingle_tokens: int) -> LeakageReport:
    report = LeakageReport()
    numbers = {n for c in cases for n in ([c.source_issue_number] + c.duplicate_issue_numbers)
               if n is not None}
    ref = {n: re.compile(rf"(?<![\w/])#{n}\b|/issues/{n}\b") for n in numbers}
    case_shingles = {c.case_id: _shingles(lexical_tokens(f"{c.input.title} {c.input.body}"), shingle_tokens)
                     for c in cases}
    for chunk in chunks:
        if chunk.issue_number in numbers:  # the eval issue itself, or a comment on it
            report.violations.append(f"{chunk.chunk_id}: belongs to evaluation issue "
                                     f"#{chunk.issue_number}")
        for n, pattern in ref.items():
            if pattern.search(chunk.text):
                report.warnings.append(f"{chunk.chunk_id}: links to evaluation issue #{n}")
        tokens = _shingles(lexical_tokens(chunk.text), shingle_tokens)
        if not tokens:
            continue
        for case_id, shingles in case_shingles.items():
            if shingles and len(shingles & tokens) / len(shingles) >= overlap_threshold:
                report.warnings.append(f"{chunk.chunk_id}: near-verbatim overlap with case {case_id}")
    return report
