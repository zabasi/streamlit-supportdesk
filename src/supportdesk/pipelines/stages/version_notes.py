"""Version and usage notes on each evidence piece (pipeline.version_notes).

The documentation in the corpus is one current snapshot (Streamlit 1.65), while reports come from versions
as old as 0.84. Without being told, the answer model treated today's documentation as true for every
version, and read a release note as the cause of a symptom. Each piece of evidence gets two short notes,
computed here from facts the corpus already has (no model call):

- version_applicability: what version the piece describes, compared with the reporter's version when the
  report states one (a release newer than the reporter's version is not in it; one older already is);
- usage_constraint: how the piece may be used (documentation: check the option exists in the reporter's
  version and setup; a release note: shows a change, not a cause; a report or comment: another user's
  observation, not verified documentation).

The controlled-routing answer step already reads these two fields and shows them beside each source.
Notes a source already carries are kept.
"""
from __future__ import annotations

import dataclasses
import re

from ...types import Chunk

_VERSION = re.compile(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b")


def versions(text: str | None) -> list[tuple[int, int, int]]:
    """Every Streamlit-style version number in a text ("1.13.0", "0.89.0 to 1.5.0"); [] if none."""
    if not text:
        return []
    return [(int(a), int(b), int(c or 0)) for a, b, c in _VERSION.findall(str(text))]


def show(v: tuple[int, int, int]) -> str:
    return f"{v[0]}.{v[1]}.{v[2]}"


def _date(as_of: str | None) -> str:
    return (as_of or "")[:10]


def _relation(release: tuple[int, int, int], reported: list[tuple[int, int, int]]) -> str:
    low, high = min(reported), max(reported)
    span = show(low) if low == high else f"{show(low)} to {show(high)}"
    if high < release:
        return f" The reporter's version ({span}) is older than this release, so this change is not in it."
    if low >= release:
        return f" The reporter's version ({span}) already includes this change."
    return f" The reporter's versions ({span}) span this release: it may or may not apply."


def notes(chunk: Chunk, reported_version: str | None) -> dict[str, str]:
    """The two notes for one piece of evidence."""
    reported = versions(reported_version)
    own = versions(chunk.version)
    kind, date = chunk.source_type, _date(chunk.as_of)
    if kind == "documentation":
        snapshot = f"Streamlit {show(own[0])}" if own else "the latest Streamlit"
        applicability = f"Current documentation, describing {snapshot}."
        # the documentation is today's snapshot: when its version is not a version number (a commit), a
        # report that states any version is treated as older; a report on the same or a newer version is not
        if reported and (not own or max(reported) < own[0]):
            applicability += (f" The report is on {show(max(reported))}, which is older: options, defaults and "
                              "behaviour described here may not exist or may differ in that version.")
        if not reported:
            # no stated version: no evidence of a mismatch, so no instruction to check one (it only made the
            # model ask for a version it did not need)
            return {"version_applicability": applicability}
        return {"version_applicability": applicability,
                "usage_constraint": ("Before recommending an option from this page, check that it exists in the "
                                     "reporter's version and fits their setup (for example st.navigation pages "
                                     "versus a pages/ directory).")}
    if kind == "release_section":
        applicability = f"Release note for Streamlit {show(own[0]) if own else chunk.version or 'an unknown version'}"
        applicability += f" ({date})." if date else "."
        if reported and own:
            applicability += _relation(own[0], reported)
        return {"version_applicability": applicability,
                "usage_constraint": ("Shows that a change was made in this release; on its own it does not show "
                                     "the cause of the reported problem.")}
    if kind in ("issue", "comment"):
        who = "Another user's report" if kind == "issue" else "A comment on another report"
        applicability = who + (f" on Streamlit {show(own[0])}" if own else "") + (f", {date}." if date else ".")
        return {"version_applicability": applicability,
                "usage_constraint": ("An observation from another case: it may share the symptom without sharing "
                                     "the cause, and it is not verified documentation.")}
    if kind == "linked_reference":
        return {"version_applicability": "Linked material" + (f" captured {date}" if date else "")
                + "; it shows the code or text as it was then.",
                "usage_constraint": "Supporting material; check that it applies to the reporter's version."}
    return {}


def tag(chunk: Chunk, reported_version: str | None) -> str:
    """A one-line version label for a candidate shown to the reranker (retrieval.rerank_versions)."""
    reported, own = versions(reported_version), versions(chunk.version)
    kind = chunk.source_type
    if kind == "documentation":
        label = f"current documentation (Streamlit {show(own[0])})" if own else "current documentation"
        if reported and (not own or max(reported) < own[0]):
            label += f", newer than the reporter's {show(max(reported))}"
        return label
    if kind == "release_section":
        label = f"release note for {show(own[0])}" if own else "release note"
        if reported and own:
            low, high = min(reported), max(reported)
            if high < own[0]:
                label += f", newer than the reporter's {show(high)}: not in their version"
            elif low >= own[0]:
                label += f", already in the reporter's {show(low)}"
        return label
    if kind in ("issue", "comment"):
        return ("another user's report" if kind == "issue" else "a comment on another report") + (
            f", on Streamlit {show(own[0])}" if own else "")
    return "linked material"


def annotate(chunks: list[Chunk], reported_version: str | None) -> list[Chunk]:
    """The chunks with their notes added (a note the source already has is kept)."""
    out = []
    for chunk in chunks:
        meta = dict(chunk.meta or {})
        for key, value in notes(chunk, reported_version).items():
            if not meta.get(key):
                meta[key] = value
        out.append(dataclasses.replace(chunk, meta=meta))
    return out


__all__ = ["annotate", "notes", "tag", "versions"]
