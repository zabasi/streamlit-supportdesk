"""Reads the evaluation files (schema 6.0) for one split, for example ``development`` or ``test``:

    {split}_inputs.jsonl                    + {split}_rubrics.jsonl                  base issue cases
    {split}_history_variant_inputs.jsonl    + {split}_history_variant_rubrics.jsonl  resumed conversations
    {split}_source_usage_inputs.jsonl       + {split}_source_usage_rubrics.jsonl     questions about sources
    {split}_multi_turn_scenarios.json                                                staged scenarios

Inputs are what the system may see; rubrics are evaluator-only and stay in ``EvalCase.rubric``.
The conversation is presented the way ``evaluation_session.py`` delivers it: the
initial conversation, then ``user_report`` as the latest user message.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..types import CaseInput
from .cases import EvalCase

SPLITS = {"development": "dev", "test": "test"}
CATEGORY = {"answerable": "answerable", "needs_clarification": "ambiguous", "needs_escalation": "escalation"}
OUTCOMES = {"answerable": ["answer"], "needs_clarification": ["request_info"], "needs_escalation": ["escalate"]}


class EvaluationSetError(ValueError):
    """An evaluation file is missing, malformed or inconsistent."""


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _joined(directory: Path, split: str, name: str) -> list[tuple[dict, dict]]:
    """(input, rubric) pairs for one kind; [] if that kind's inputs file is absent."""
    inputs = directory / f"{split}_{name}inputs.jsonl"
    if not inputs.exists():
        return []
    rubrics_path = directory / f"{split}_{name}rubrics.jsonl"
    if not rubrics_path.exists():
        raise EvaluationSetError(f"{inputs.name} has no rubric file ({rubrics_path.name})")
    rubrics = {r["case_id"]: r for r in _rows(rubrics_path)}
    pairs = []
    for row in _rows(inputs):
        if row["case_id"] not in rubrics:
            raise EvaluationSetError(f"{inputs.name}: case {row['case_id']} has no rubric")
        pairs.append((row, rubrics[row["case_id"]]))
    return pairs


def _messages(row: dict) -> list[dict]:
    """Initial conversation plus the user report as the newest message (evaluation_session.py)."""
    convo = [{"role": m["role"], "body": m["content"]} for m in row.get("initial_conversation") or []]
    return convo + [{"role": "user", "body": row["user_report"]}]


def _issue_number(*values: str | None) -> int | None:
    for value in values:
        match = re.search(r"(\d+)$", value or "")
        if match:
            return int(match.group(1))
    return None


def _evidence(rubric: dict) -> tuple[list[str], list[dict]]:
    spans = [{k: e[k] for k in ("source_id", "start_char", "end_char", "evidence_sha256", "text")}
             | {"evidence_id": e.get("evidence_id")} for e in rubric.get("acceptable_retrievable_evidence") or []]
    return list(dict.fromkeys(s["source_id"] for s in spans)), spans


def _missing(rubric: dict) -> list[str]:
    return [m["question"] if isinstance(m, dict) else str(m) for m in rubric.get("required_missing_information") or []]


def _case(row: dict, rubric: dict, *, kind: str, split: str) -> EvalCase:
    messages = _messages(row)
    first, rest = messages[0], messages[1:]
    # historical_evaluation false: the case may see the whole corpus, so no date filter applies.
    created_at = row.get("created_at") if row.get("historical_evaluation") else None
    env = {k: v for k, v in (row.get("reported_environment") or {}).items() if v}
    gold, spans = _evidence(rubric)
    base = row.get("base_case_id") or row["case_id"]
    if kind == "issue":
        category = rubric.get("case_category")
        if category not in CATEGORY:
            raise EvaluationSetError(f"case {row['case_id']}: unknown case_category {category!r}")
        mapped, outcomes = CATEGORY[category], OUTCOMES[category]
    else:
        mapped, outcomes = kind, []  # scored by the rubric, not by outcome
    title = row.get("title") or first["body"][:120]
    return EvalCase(
        case_id=row["case_id"], family_id=f"family:{row.get('family_group_ref') or base}", split=split,
        category=mapped,
        input=CaseInput(case_id=row["case_id"], title=title, body=first["body"], created_at=created_at,
                        history=rest, environment=env),
        gold_evidence=gold, acceptable_outcomes=outcomes, missing_info=_missing(rubric),
        source_issue_number=_issue_number(row.get("case_source_id"), base) if kind != "source_usage" else None,
        kind=kind, independent=kind == "issue", evidence_spans=spans, rubric=rubric,
        notes=row.get("input_origin", ""))


def load_evaluation_set(directory: str | Path, split: str) -> list[EvalCase]:
    """All cases of one split (``development`` or ``test``) found in ``directory``."""
    if split not in SPLITS:
        raise EvaluationSetError(f"split must be one of {sorted(SPLITS)}")
    directory = Path(directory)
    cases = ([_case(i, r, kind="issue", split=SPLITS[split]) for i, r in _joined(directory, split, "")]
             + [_case(i, r, kind="history_variant", split=SPLITS[split])
                for i, r in _joined(directory, split, "history_variant_")]
             + [_case(i, r, kind="source_usage", split=SPLITS[split])
                for i, r in _joined(directory, split, "source_usage_")])
    if not cases:
        raise EvaluationSetError(f"no {split}_*inputs.jsonl files in {directory}")
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise EvaluationSetError(f"duplicate case ids in {split}")
    return cases


def load_scenarios(directory: str | Path, split: str) -> list[dict]:
    path = Path(directory) / f"{split}_multi_turn_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def check_spans(cases: list[EvalCase], corpus_texts: dict[str, str]) -> list[str]:
    """Every gold span must be exactly ``corpus[source_id].text[start_char:end_char]``."""
    problems = []
    for case in cases:
        for span in case.evidence_spans:
            text = corpus_texts.get(span["source_id"])
            if text is None:
                problems.append(f"{case.case_id}: {span['source_id']} is not in the corpus")
            elif text[span["start_char"]:span["end_char"]] != span["text"]:
                problems.append(f"{case.case_id}: span {span['evidence_id']} does not match the corpus text")
    return problems
