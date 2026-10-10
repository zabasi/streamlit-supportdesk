"""The data contract: one ``corpus.jsonl`` file, one prepared source per line.

Each line: ``{"source_id", "source_type", "text", "metadata", "text_sha256"}``. The text is already
prepared for retrieval (titles and issue context included); ``text_sha256`` must match it, so a
corpus edited after preparation is caught instead of silently indexed.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_TYPES = ("issue", "comment", "documentation", "release_section", "linked_reference")
CORPUS_FILE = "corpus.jsonl"

#: Metadata kept on every chunk (when present): what the model or a reviewer may need to weigh it.
META_KEYS = ("state", "state_reason", "labels", "author_association", "evidence_role", "allowed_use",
             "verification_status", "section_heading", "linked_source_ids", "family_id",
             "related_issue_numbers")


class DataError(ValueError):
    """Input data (the corpus or evaluation cases) is missing or malformed."""


class CorpusError(DataError):
    """The corpus file is missing or a record breaks the contract."""


@dataclass(frozen=True)
class SourceRecord:
    """One source, in the same shape whatever its type."""

    source_id: str
    source_type: str
    title: str
    text: str
    url: str
    parent_source_id: str | None = None
    issue_number: int | None = None
    linked_issue_numbers: frozenset[int] = frozenset()  # issues a linked reference was linked from
    version: str | None = None
    as_of: str | None = None  # when it became public; None = undated (always visible)
    meta: dict = field(default_factory=dict)
    text_sha256: str = ""  # of ``text``, exactly as delivered


@dataclass
class Corpus:
    records: list[SourceRecord]
    path: Path
    sha256: str
    notes: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "records": len(self.records),
                "by_type": dict(Counter(r.source_type for r in self.records)), **self.notes}


def _issue_number(source_id: str) -> int | None:
    prefix, _, rest = source_id.partition(":")
    return int(rest) if prefix == "issue" and rest.isdigit() else None


def _version(meta: dict, *keys: str) -> str | None:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, dict):  # for example {"value": "0.84", "evidence_text": ...}
            value = value.get("value")
        if value:
            return str(value)
    return None


def _record(row: dict, line: int, notes: Counter) -> SourceRecord:
    for key, kind in (("source_id", str), ("source_type", str), ("text", str), ("metadata", dict)):
        if not isinstance(row.get(key), kind):
            raise CorpusError(f"line {line}: {key!r} must be a {kind.__name__}")
    sid, kind, text, meta = row["source_id"], row["source_type"], row["text"], row["metadata"]
    if kind not in SOURCE_TYPES:
        raise CorpusError(f"line {line} ({sid}): unknown source_type {kind!r}; known: {SOURCE_TYPES}")
    if not text.strip():
        raise CorpusError(f"line {line} ({sid}): empty text")
    expected = row.get("text_sha256")
    if expected and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
        raise CorpusError(f"line {line} ({sid}): text does not match text_sha256 (edited after preparation?)")

    url = meta.get("html_url") or meta.get("source_url") or ""
    title = meta.get("title") or ""
    parent, number, linked = meta.get("parent_source_id"), None, frozenset()
    if kind == "issue":
        number, version, as_of = meta.get("number"), _version(meta, "streamlit_version"), meta.get("created_at")
    elif kind == "comment":
        number = meta.get("issue_number") or _issue_number(parent or "")
        title = title or f"Comment on issue #{number}"
        version, as_of = None, meta.get("created_at")
    elif kind == "documentation":
        version, as_of = _version(meta, "declared_streamlit_version", "commit"), None  # pinned commit
    elif kind == "release_section":
        version, as_of = _version(meta, "version"), meta.get("release_date")
        title = title or f"Release {version}" + (f": {meta['section_heading']}" if meta.get("section_heading") else "")
    else:  # linked_reference
        version = _version(meta, "declared_streamlit_version", "commit_or_tag")
        linked = frozenset(n for n in map(_issue_number, meta.get("linked_from_source_ids") or []) if n)
        as_of = meta.get("created_at")
        if not as_of:
            # No publication date: use when it was fetched. That hides it from historical cases
            # filed before then, instead of letting a later page leak into them.
            as_of = meta.get("retrieved_at")
            notes["linked_reference_dated_by_retrieved_at"] += 1
    return SourceRecord(source_id=sid, source_type=kind, title=title, text=text, url=url,
                        parent_source_id=parent, issue_number=number, linked_issue_numbers=linked,
                        version=version, as_of=as_of,
                        text_sha256=expected or hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        meta={k: meta[k] for k in META_KEYS if meta.get(k) is not None})


def load_corpus(directory: str | Path) -> Corpus:
    path = Path(directory) / CORPUS_FILE
    if not path.exists():
        raise CorpusError(f"no corpus at {path}: put corpus.jsonl there, or try the bundled sample "
                          "with `export SUPPORTDESK_CONFIG=configs/sample.toml`")
    raw = path.read_bytes()
    records, seen, notes = [], set(), Counter()
    for line, text in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"line {line}: not JSON ({exc.msg})") from None
        if row.get("metadata", {}).get("embedding_allowed") is False:
            notes["not_embeddable_skipped"] += 1
            continue
        record = _record(row, line, notes)
        if record.source_id in seen:
            raise CorpusError(f"line {line}: duplicate source_id {record.source_id!r}")
        seen.add(record.source_id)
        records.append(record)
    if not records:
        raise CorpusError(f"{path} has no records")
    return Corpus(records, path, hashlib.sha256(raw).hexdigest(), dict(notes))


def write_corpus(directory: str | Path, rows: list[dict]) -> Path:
    """Write rows in the contract format, filling in ``text_sha256``."""
    path = Path(directory) / CORPUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            row = {**row, "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest()}
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
