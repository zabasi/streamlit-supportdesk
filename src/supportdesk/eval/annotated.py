"""Load gold evidence from the annotated development dump (selected_spans).

The dump is one JSON object per case: the model input plus ``sources[]`` with the original
corpus record and the reviewer-selected character spans. Gold for retrieval is those
``source_id``s; passage metrics use the spans. Official ``data/eval`` cases (category,
outcomes) are kept when present; only the gold overlay changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from ..types import CaseInput
from .cases import EvalCase

#: Where the annotated development dump is expected (not shipped with the repository); scripts take --gold PATH.
DEFAULT_ANNOTATED = Path("data/eval/development_source_annotated_v1/development.jsonl")


def _issue_number(case_id: str) -> int | None:
    match = re.search(r"(\d+)$", case_id or "")
    return int(match.group(1)) if match else None


def read_annotated(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotated gold file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_from_row(row: dict) -> tuple[list[str], list[dict]]:
    """Unique source ids and the selected spans (source_id, start_char, end_char, text)."""
    gold, spans = [], []
    for block in row.get("sources") or []:
        source = block.get("source") or {}
        source_id = source.get("source_id") or block.get("source_id")
        if not source_id:
            continue
        gold.append(source_id)
        for span in block.get("selected_spans") or []:
            spans.append({
                "source_id": source_id,
                "start_char": span["start_char"],
                "end_char": span["end_char"],
                "text": span.get("text", ""),
                "evidence_id": span.get("span_id"),
            })
    return list(dict.fromkeys(gold)), spans


def case_from_annotated(row: dict, *, official: EvalCase | None = None) -> EvalCase:
    gold, spans = gold_from_row(row)
    incoming = row.get("input") or {}
    if official is not None:
        return replace(official, gold_evidence=gold, evidence_spans=spans)
    case_id = row["case_id"]
    return EvalCase(
        case_id=case_id, family_id=f"family:{case_id}", split="dev", category="answerable",
        input=CaseInput.from_dict({"case_id": case_id, **incoming}),
        gold_evidence=gold, acceptable_outcomes=["answer"],
        source_issue_number=_issue_number(case_id), evidence_spans=spans,
        notes="annotated selected_spans gold",
    )


def load_annotated_cases(path: str | Path, *, official: list[EvalCase] | None = None) -> list[EvalCase]:
    """The annotated cases, with official category/outcomes overlaid when ``official`` is given."""
    by_id = {c.case_id: c for c in official or []}
    return [case_from_annotated(row, official=by_id.get(row["case_id"])) for row in read_annotated(path)]


def case_to_eval_row(case: EvalCase) -> dict:
    """JSON object ``case_from_dict`` can read, including the gold overlay."""
    return {
        "case_id": case.case_id, "family_id": case.family_id, "split": case.split,
        "category": case.category, "input": asdict(case.input),
        "gold_evidence": list(case.gold_evidence),
        "acceptable_outcomes": list(case.acceptable_outcomes),
        "missing_info": list(case.missing_info),
        "source_issue_number": case.source_issue_number,
        "duplicate_issue_numbers": list(case.duplicate_issue_numbers),
        "notes": case.notes, "kind": case.kind, "independent": case.independent,
        "evidence_spans": list(case.evidence_spans),
    }


def write_eval_jsonl(cases: list[EvalCase], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(case_to_eval_row(c), ensure_ascii=False) + "\n" for c in cases),
                    encoding="utf-8")
    return path


__all__ = ["DEFAULT_ANNOTATED", "case_from_annotated", "case_to_eval_row", "gold_from_row",
           "load_annotated_cases", "read_annotated", "write_eval_jsonl"]
