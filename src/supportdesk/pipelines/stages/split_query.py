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

    def __init__(self, max_chars: int, max_queries: int):
        self.clean, self.max_queries = CleanQuery(max_chars), max_queries

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        title, *rest = self.clean(case, env)[0].split("\n")
        code = [line for line in rest if _CODE.search(line)]
        prose = " ".join(line for line in rest if line not in code)
        sentences = [s.strip() for s in _SENTENCE_END.split(prose) if len(s.split()) >= 4]
        queries = [title] + ([" ".join(code)] if code else []) + sentences
        return list(dict.fromkeys(q for q in queries if q.strip()))[: self.max_queries]
