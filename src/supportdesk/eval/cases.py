"""Evaluation cases: what the assistant sees, what counts as good evidence, which outcomes are
defensible. Gold labels never enter the search index or a prompt."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..data.chunking import Exclusions
from ..types import CaseInput


@dataclass
class EvalCase:
    case_id: str
    family_id: str  # duplicate reports share a family; a family is never split across dev and test
    split: str  # dev | test
    category: str  # answerable | ambiguous | escalation
    input: CaseInput
    gold_evidence: list[str] = field(default_factory=list)  # source_ids acceptable as evidence
    acceptable_outcomes: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)  # facts a good reply should ask for
    source_issue_number: int | None = None  # the real issue this case was derived from
    duplicate_issue_numbers: list[int] = field(default_factory=list)
    notes: str = ""
    kind: str = "issue"  # issue | history_variant | source_usage
    independent: bool = True  # False: a derivative of another case; not counted in sample size
    evidence_spans: list[dict] = field(default_factory=list)  # exact gold spans (source_id, start/end_char)
    rubric: dict = field(default_factory=dict)  # evaluator-only; never shown to the system


def case_from_dict(row: dict) -> EvalCase:
    problems = []
    for key in ("case_id", "family_id", "split", "category", "input", "acceptable_outcomes"):
        if key not in row:
            problems.append(f"missing {key!r}")
    if problems:
        raise ValueError(f"case {row.get('case_id', '?')}: {', '.join(problems)}")
    if row["split"] not in ("dev", "test"):
        raise ValueError(f"case {row['case_id']}: split must be dev or test")
    if row["category"] not in config.CATEGORIES:
        raise ValueError(f"case {row['case_id']}: category must be one of {config.CATEGORIES}")
    bad = set(row["acceptable_outcomes"]) - set(config.OUTCOMES)
    if bad:
        raise ValueError(f"case {row['case_id']}: unknown outcomes {sorted(bad)}")
    return EvalCase(
        case_id=row["case_id"], family_id=row["family_id"], split=row["split"],
        category=row["category"], input=CaseInput.from_dict({"case_id": row["case_id"],
                                                             **row["input"]}),
        gold_evidence=list(row.get("gold_evidence", [])),
        acceptable_outcomes=list(row["acceptable_outcomes"]),
        missing_info=list(row.get("missing_info", [])),
        source_issue_number=row.get("source_issue_number"),
        duplicate_issue_numbers=list(row.get("duplicate_issue_numbers", [])),
        notes=row.get("notes", ""),
        kind=row.get("kind", "issue"),
        independent=row.get("independent", True),
        evidence_spans=list(row.get("evidence_spans") or []))


def load_cases(*paths: str | Path, split: str | None = None) -> list[EvalCase]:
    """A file holds cases in this project's format; a directory holds evaluation files
    (see eval/dataset.py), read for every split present."""
    from .dataset import SPLITS, load_evaluation_set
    cases: list[EvalCase] = []
    for path in map(Path, paths):
        if path.is_dir():
            for name in SPLITS:
                if (path / f"{name}_inputs.jsonl").exists():
                    cases += load_evaluation_set(path, name)
            continue
        with path.open(encoding="utf-8") as handle:
            cases += [case_from_dict(json.loads(line)) for line in handle if line.strip()]
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id across case files")
    return [c for c in cases if split is None or c.split == split]


def exclusions_from_cases(cases: list[EvalCase]) -> Exclusions:
    """Everything an evaluation case came from (and its duplicates) is kept out of the index.
    Pass dev AND test cases: neither may be searchable."""
    numbers: set[int] = set()
    for case in cases:
        if case.source_issue_number is not None:
            numbers.add(case.source_issue_number)
        numbers.update(case.duplicate_issue_numbers)
    return Exclusions(issue_numbers=frozenset(numbers))
