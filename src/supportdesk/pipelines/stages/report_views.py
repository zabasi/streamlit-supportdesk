"""Extra views of a report for search, without a model call (retrieval.report_views).

A whole report is a long query: a feature it names once (``st.fragment(run_every=2)``) or the exact error
it shows can be outweighed by its many other words. Two views search for those alone:

- title: the report's title, the reporter's own one-line summary
- code: the Streamlit commands the report calls (``st.*``), the parameter names passed to them, and the
  first error or exception lines

advanced fuses each view's ranking with the report's at a lower weight (retrieval.view_weight).
"""
from __future__ import annotations

import re

from ...types import CaseInput

VIEWS = ("title", "code")
_COMMAND = re.compile(r"\bst\.[A-Za-z_][\w.]*")
_CALL = re.compile(r"\bst\.[A-Za-z_][\w.]*\s*\(([^()]*)\)")
_PARAMETER = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")
_ERROR = re.compile(r"^[^\n]*\b\w*(?:Error|Exception)\b[^\n]*$", re.MULTILINE)


def code_view(text: str, *, max_errors: int = 2, max_chars: int = 300) -> str:
    """The Streamlit commands called, the parameter names passed to them, and the first error lines."""
    commands = list(dict.fromkeys(m.rstrip(".") for m in _COMMAND.findall(text)))
    parameters = list(dict.fromkeys(p for args in _CALL.findall(text) for p in _PARAMETER.findall(args)))
    errors = [" ".join(line.split())[:max_chars] for line in _ERROR.findall(text)][:max_errors]
    return " ".join(commands + parameters + errors).strip()


def report_views(case: CaseInput, kinds: str) -> list[str]:
    """The views named in ``kinds`` (comma-separated: title, code; or none), each a query; empty views
    are left out."""
    wanted = [k.strip() for k in kinds.split(",") if k.strip() and k.strip() != "none"]
    views = []
    for kind in wanted:
        text = " ".join((case.title or "").split()) if kind == "title" else code_view(f"{case.title}\n{case.body}")
        if text and text not in views:
            views.append(text)
    return views


__all__ = ["VIEWS", "code_view", "report_views"]
