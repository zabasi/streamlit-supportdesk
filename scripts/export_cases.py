"""Write one JSON file per case of a finished run, with everything about that case in one place.

For each case: the report (title, body, environment, earlier messages), its evaluation label (category and
expected decision), the decision made and whether it matches, the reply to the user, the pipeline's
maintainer summary, the controlled-routing v2 record when there is one (the model's structured assessment,
the routing, the reference map and the package's own maintainer summary), the evidence in the prompt and
what it cost. Read from the run's results.jsonl and summary.json and the case files the run used; no model
call. An index.json lists the cases with their label and decision.

    python scripts/export_cases.py e2e-advanced
    python scripts/export_cases.py e2e-advanced --output-dir docs/cases/e2e-advanced
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supportdesk.eval.cases import load_cases


def case_record(row: dict, case, run_id: str) -> dict:
    attempts = (row.get("trace") or {}).get("attempts") or []
    controlled = next((a["controlled_v2"] for a in attempts if isinstance(a, dict) and "controlled_v2" in a), None)
    expected = (case.acceptable_outcomes or [None])[0] if case else None
    record = {
        "run_id": run_id, "case_id": row["case_id"], "split": row.get("split"),
        "report": None if case is None else {"title": case.input.title, "body": case.input.body,
                                             "created_at": case.input.created_at,
                                             "environment": dict(case.input.environment or {}),
                                             "history": list(case.input.history or [])},
        "label": None if case is None else {"category": case.category, "expected": expected},
        "decision": row.get("outcome"),
        "matches_label": None if expected is None else row.get("outcome") == expected,
        "reply_to_user": row.get("user_reply"),
        "maintainer_summary": row.get("maintainer_summary"),
        "controlled_v2": controlled,
        "evidence_in_prompt": [p["chunk_id"] for p in (row.get("trace") or {}).get("packet", [])],
        "citations": row.get("citations"),
        "error": row.get("error"),
        "usage": row.get("usage"),
    }
    if controlled is not None and "assessment" not in controlled:
        record["note"] = "this run was made before the assessment was saved: replay it to add it (no cost)"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run id (under --runs-dir) or a run directory")
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <run directory>/cases")
    args = parser.parse_args(argv)
    run_dir = Path(args.run) if (Path(args.run) / "results.jsonl").exists() else args.runs_dir / args.run
    if not (run_dir / "results.jsonl").exists():
        parser.error(f"no results.jsonl in {run_dir}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cases = {c.case_id: c for c in load_cases(*(summary.get("case_files") or ["data/eval"]))}
    out = args.output_dir or run_dir / "cases"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = case_record(row, cases.get(row["case_id"]), run_dir.name)
        (out / f"{row['case_id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({k: record[k] for k in ("case_id", "decision", "matches_label")} | {"label": record["label"]})
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    right = sum(1 for i in index if i["matches_label"])
    print(f"wrote {len(index)} case files to {out} ({right} of {len(index)} decisions match their label)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
