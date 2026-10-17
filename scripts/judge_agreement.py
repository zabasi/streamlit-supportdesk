"""Compare the judge's case judgements with a person's, field by field.

Each directory holds one JSON file per case (<case_id>.json) in the judge's format
(configs/prompts/judge_case_system.txt; the person's from docs/evaluation/human_review_form.md). Only cases
judged by both are compared. Prints agreement per field (scores: exact and within one point) and every
disagreement.

    python scripts/judge_agreement.py judgements/model judgements/human
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = [  # (label, path, kind)
    ("verdict", ("overall", "verdict"), "label"),
    ("decision acceptable", ("decision_quality", "decision_acceptable"), "label"),
    ("decision score", ("decision_quality", "score"), "score"),
    ("evidence score", ("evidence_quality", "score"), "score"),
    ("version and uncertainty", ("decision_quality", "version_and_uncertainty_handling"), "score"),
    ("user response", ("response_and_summary", "user_response"), "score"),
    ("maintainer summary", ("response_and_summary", "maintainer_summary"), "score"),
    ("reply matches stored decision", ("operational_success", "reply_matches_stored_decision"), "label"),
    ("failure type", ("failure_analysis", "failure_type"), "label"),
]


def value(record: dict, path: tuple[str, ...]):
    for key in path:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def compare(model_dir: Path, human_dir: Path) -> dict:
    model = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in model_dir.glob("*.json")}
    human = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in human_dir.glob("*.json")}
    both = sorted(set(model) & set(human))
    out = {"cases": both, "fields": {}, "disagreements": []}
    for label, path, kind in FIELDS:
        pairs = [(c, value(model[c], path), value(human[c], path)) for c in both]
        pairs = [(c, m, h) for c, m, h in pairs if m is not None and h is not None]
        exact = sum(m == h for _, m, h in pairs)
        row = {"compared": len(pairs), "exact": exact}
        if kind == "score":
            row["within_one"] = sum(abs(int(m) - int(h)) <= 1 for _, m, h in pairs)
        out["fields"][label] = row
        out["disagreements"] += [{"case_id": c, "field": label, "judge": m, "human": h} for c, m, h in pairs if m != h]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("human_dir", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.model_dir, args.human_dir)
    if not result["cases"]:
        print("no case judged by both", file=sys.stderr)
        return 1
    print(f"cases judged by both: {len(result['cases'])}\n")
    print(f"| {'field':<30} | agree (exact) | within one point |\n|---|---:|---:|")
    for label, row in result["fields"].items():
        n = row["compared"] or 1
        within = f"{row['within_one']}/{row['compared']}" if "within_one" in row else "—"
        print(f"| {label:<30} | {row['exact']}/{row['compared']} ({100 * row['exact'] / n:.0f}%) | {within} |")
    if result["disagreements"]:
        print("\ndisagreements:")
        for d in result["disagreements"]:
            print(f"  {d['case_id']}  {d['field']}: judge {d['judge']!r}, human {d['human']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
