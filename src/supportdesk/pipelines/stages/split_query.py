"""Several queries from the cleaned report without a model: the title, the Streamlit calls together,
and the report's sentences."""
from __future__ import annotations

import re

from ...types import CaseInput
from .clean_query import CleanQuery

_CODE = re.compile(r"\bst\.\w+|Error|Exception|Traceback")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceSplitQuery:
    """Up to retrieval.max_queries queries: the title, the code lines, then the report's sentences."""

    def __init__(self, max_chars: int, max_queries: int, keep_report: bool = False):
        self.clean, self.max_queries, self.keep_report = CleanQuery(max_chars), max_queries, keep_report

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        report = self.clean(case, env)[0]
        title, *rest = report.split("\n")
        code = [line for line in rest if _CODE.search(line)]
        prose = " ".join(line for line in rest if line not in code)
        sentences = [s.strip() for s in _SENTENCE_END.split(prose) if len(s.split()) >= 4]
        queries = [title] + ([" ".join(code)] if code else []) + sentences
        pieces = list(dict.fromkeys(q for q in queries if q.strip()))[: self.max_queries]
        return [report] + [q for q in pieces if q != report] if self.keep_report else pieces
