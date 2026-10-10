"""Personal data and secrets. The indexing graph uses sensitive_findings to refuse a corpus that
contains any; the query step uses clean_text to tidy a user's report before searching."""
from __future__ import annotations

import re

from ..config import SECRET_PATTERN

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_BLANKS = re.compile(r"\n{3,}")


def sensitive_findings(text: str) -> list[str]:
    """What kind of personal data or secret ``text`` contains, if any. The indexing graph's screen
    node uses this to refuse a corpus instead of editing it (editing would break chunk offsets)."""
    return [kind for kind, pattern in (("email", _EMAIL), ("secret", SECRET_PATTERN)) if pattern.search(text)]


def clean_text(text: str | None) -> str:
    """Normalise newlines, drop HTML comments (issue-template boilerplate), mask emails/secrets."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT.sub("", text)
    text = _EMAIL.sub("<email>", text)
    text = SECRET_PATTERN.sub("<redacted-secret>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()
