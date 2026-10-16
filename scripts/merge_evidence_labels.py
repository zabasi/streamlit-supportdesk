"""Merge reviewed evidence labels into the evaluation rubrics, optionally making the evaluation historical.

The labels file (one row per case: {"case_id", "input", "sources": [{"source", "selection_scope",
"selected_spans"}]}) replaces only the evidence part of {split}_rubrics.jsonl
(acceptable_retrievable_evidence); the category, reasoning, next steps, forbidden claims and rubric scores
are kept. claim_support references to old evidence are re-pointed to the new spans that cover the same text.

By default the evaluation stays non-historical (every case may see the whole corpus) and all labelled spans
are gold. With --historical, each case's historical_evaluation flag is set, so retrieval only sees dated
sources (issues, comments, release notes, linked sources) from before the case; documentation has no date
and stays visible. Evidence from a source dated on or after its case cannot be found then, so it moves to
the rubric's later_evidence list instead of the gold. --out-dir writes the result elsewhere, for example a
historical copy of the cases beside the main ones. Every evidence entry records its source's date.

Every source and span is checked against the corpus first; nothing is written if a check fails.

    python scripts/merge_evidence_labels.py --labels development.jsonl --dry-run
    python scripts/merge_evidence_labels.py --labels development.jsonl
    python scripts/merge_evidence_labels.py --labels development.jsonl --historical --out-dir data/eval_historical
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from supportdesk.data.corpus import load_corpus

SPLIT_FILES = {"development": "development", "dev": "development", "test": "test"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def merge(labels: list[dict], inputs: list[dict], rubrics: list[dict], corpus: dict, *,
          historical: bool) -> tuple[list[dict], list[dict], dict]:
    """(new inputs, new rubrics, report). Raises ValueError on any mismatch with the corpus or the cases."""
    by_case = {row["case_id"]: row for row in labels}
    if set(by_case) != {r["case_id"] for r in rubrics} or set(by_case) != {r["case_id"] for r in inputs}:
        raise ValueError("the labels, inputs and rubrics must hold the same case ids")
    dates = {r["case_id"]: r.get("created_at") for r in inputs}
    if historical and not all(dates.values()):
        raise ValueError("historical evaluation needs created_at for every case in the inputs file")
    report = {"cases": {}, "unmapped_claim_refs": []}
    new_rubrics = []
    for rubric in rubrics:
        cid = rubric["case_id"]
        template = (rubric.get("acceptable_retrievable_evidence") or [{}])[0]
        kept, later = [], []
        for entry in by_case[cid]["sources"]:
            src = entry["source"]
            record = corpus.get(src["source_id"])
            if record is None or record.text != src["text"]:
                raise ValueError(f"{cid}: source {src['source_id']} is missing from the corpus or its text differs")
            for span in entry["selected_spans"]:
                a, b = span["start_char"], span["end_char"]
                if record.text[a:b] != span["text"]:
                    raise ValueError(f"{cid}: span {span['span_id']} is not the source text at [{a}:{b}]")
                first_line = span["text"].lstrip().splitlines()[0] if span["text"].strip() else ""
                item = {"evidence_id": span["span_id"], "source_id": src["source_id"],
                        "source_type": src["source_type"], "start_char": a, "end_char": b,
                        "end_exclusive": True, "text": span["text"], "evidence_sha256": sha(span["text"]),
                        "source_text_sha256": sha(record.text), "selection_scope": entry.get("selection_scope"),
                        "heading": first_line if first_line.startswith("#") else None,
                        "offset_basis": template.get("offset_basis", "retrieval_corpus.text"),
                        "offset_unit": template.get("offset_unit", "python_unicode_code_points"),
                        "version_scope": template.get("version_scope"), "source_as_of": record.as_of}
                if historical and record.as_of and record.as_of >= dates[cid]:
                    later.append({**item, "case_created_at": dates[cid]})
                else:
                    kept.append(item)
        old = {e["evidence_id"]: e for e in rubric.get("acceptable_retrievable_evidence") or []}
        claims = []
        for claim in rubric.get("claim_support") or []:
            ids = []
            for ref in claim.get("evidence_ids") or []:
                e = old.get(ref)
                covering = [k["evidence_id"] for k in kept if e and k["source_id"] == e["source_id"]
                            and k["start_char"] < e["end_char"] and e["start_char"] < k["end_char"]]
                ids.extend(covering)
                if not covering:
                    report["unmapped_claim_refs"].append((cid, claim.get("claim_id"), ref))
            claims.append({**claim, "evidence_ids": list(dict.fromkeys(ids))})
        new_rubrics.append({**rubric, "acceptable_retrievable_evidence": kept, "later_evidence": later,
                            "claim_support": claims})
        report["cases"][cid] = {"gold": len(kept), "later": len(later),
                                "types": sorted({k["source_type"] for k in kept})}
    new_inputs = [{**r, "historical_evaluation": historical} for r in inputs]
    return new_inputs, new_rubrics, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True, help="the reviewed labels file (one row per case)")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="development", choices=sorted(SPLIT_FILES))
    parser.add_argument("--out-dir", type=Path, default=None, help="write the merged files here instead of --eval-dir")
    parser.add_argument("--historical", action=argparse.BooleanOptionalAction, default=False,
                        help="set historical_evaluation and move evidence dated on or after its case to later_evidence")
    parser.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = parser.parse_args(argv)
    name = SPLIT_FILES[args.split]
    inputs_path, rubrics_path = args.eval_dir / f"{name}_inputs.jsonl", args.eval_dir / f"{name}_rubrics.jsonl"
    corpus = {r.source_id: r for r in load_corpus(args.data_dir).records}
    try:
        inputs, rubrics, report = merge(read_jsonl(args.labels), read_jsonl(inputs_path), read_jsonl(rubrics_path),
                                        corpus, historical=args.historical)
    except ValueError as error:
        print(f"not merged: {error}", file=sys.stderr)
        return 1
    for cid, r in report["cases"].items():
        print(f"{cid:>10}: {r['gold']:>2} gold spans ({', '.join(r['types'])}), {r['later']} moved to later_evidence")
    print(f"total: {sum(r['gold'] for r in report['cases'].values())} gold spans, "
          f"{sum(r['later'] for r in report['cases'].values())} later | historical_evaluation={args.historical}")
    if report["unmapped_claim_refs"]:
        print(f"claim_support references with no covering new span (dropped): {report['unmapped_claim_refs']}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out = args.out_dir or args.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / inputs_path.name, inputs)
    write_jsonl(out / rubrics_path.name, rubrics)
    print(f"wrote {out / inputs_path.name} and {out / rubrics_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
