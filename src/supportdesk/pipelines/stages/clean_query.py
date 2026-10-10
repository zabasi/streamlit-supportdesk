"""A cleaner search query: the parts of a bug report that say what is wrong, without the GitHub
issue template around them."""
from __future__ import annotations

import re

from ...types import CaseInput

#: Template sections that say nothing about the problem (versions are read separately by intake).
DROPPED_SECTIONS = {"checklist", "is this a regression", "debug info", "are you willing to submit a pr"}

#: Template text that appears in reports regardless of the problem.
BOILERPLATE = ("community voting on feature requests", "if you'd like the streamlit team to prioritize",
               "_no response_", "i have searched the existing issues", "i added a very descriptive title")

_HEADING = re.compile(r"^#{2,4}\s+(.*)$")
_FENCE = re.compile(r"^\s*```")
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]")
_HTML = re.compile(r"<[^>]+>")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_CODE_SIGNAL = re.compile(r"\bst\.\w+|Error|Exception|Traceback|Warning")


def clean_report(text: str) -> str:
    """The report's meaningful lines: prose of the useful sections, and from code only the lines
    that call Streamlit or name an error. Headings, template sections and links are dropped."""
    kept, in_code, skip_section = [], False, False
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if _FENCE.match(line):
            in_code = not in_code
            continue
        heading = None if in_code else _HEADING.match(line)
        if heading:
            name = heading.group(1).strip().rstrip("?").strip().lower()
            skip_section = name in DROPPED_SECTIONS
            continue
        if skip_section or line.strip() == "---":
            continue
        if in_code:
            if _CODE_SIGNAL.search(line):
                kept.append(line.strip())
            continue
        if _CHECKBOX.match(line) or any(b in line.lower() for b in BOILERPLATE):
            continue
        line = _URL.sub("", _LINK.sub(r"\1", _IMAGE.sub("", _HTML.sub("", line)))).strip()
        if line:
            kept.append(line)
    return "\n".join(kept)


class CleanQuery:
    """One query: the title and the meaningful parts of the report, without template, links or debug info."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        parts = [case.title, clean_report(case.body)] + [
            clean_report(h.get("body", "")) for h in case.history if h.get("role", "user") == "user"]
        return ["\n".join(p for p in parts if p)[: self.max_chars]]
