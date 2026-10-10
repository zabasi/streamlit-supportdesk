"""Plain data types shared by every part of the system. They hold data and convert to and from dicts,
nothing else."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A citable span of one source: ``text == corpus[source_id].text[start_char:end_char]``
    (Python code-point offsets, end exclusive), checked when the index is built."""

    chunk_id: str
    source_id: str
    source_type: str  # issue | comment | documentation | release_section | linked_reference
    title: str
    text: str
    url: str = ""
    parent_source_id: str | None = None
    issue_number: int | None = None
    version: str | None = None  # Streamlit version, documentation commit or release version, if known
    as_of: str | None = None  # ISO timestamp the content became public; None = undated
    meta: dict = field(default_factory=dict)  # state, labels, author_association, ...
    start_char: int | None = None  # inclusive offset into the source text
    end_char: int | None = None  # exclusive
    source_text_sha256: str | None = None  # the parent source's text_sha256

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> Chunk:
        return cls(**row)


@dataclass
class CaseInput:
    """Everything the assistant is allowed to see at the start of a case."""

    case_id: str
    title: str
    body: str
    created_at: str | None = None
    history: list[dict] = field(default_factory=list)  # [{"role", "body", "created_at"}]
    environment: dict = field(default_factory=dict)  # known fields only; missing stays absent

    @classmethod
    def from_dict(cls, row: dict) -> CaseInput:
        return cls(
            case_id=row["case_id"],
            title=row.get("title", ""),
            body=row.get("body", ""),
            created_at=row.get("created_at"),
            history=list(row.get("history") or []),
            environment=dict(row.get("environment") or {}),
        )


@dataclass
class CaseResult:
    """One uniform record for every pipeline, so baseline and final compare like for like."""

    case_id: str
    pipeline: str
    mode: str
    outcome: str  # config.OUTCOMES or config.FAILURE_OUTCOMES
    confidence: str | None = None  # the model's own: high | medium | low
    user_reply: str | None = None
    maintainer_summary: dict = field(default_factory=dict)
    known_facts: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_step: str | None = None
    citations: list[str] = field(default_factory=list)  # chunk_ids, all inside the packet
    violations: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # rank-ordered {chunk_id, source_id, ...}
    packet_hash: str | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
