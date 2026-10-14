"""Evaluate the pre-retrieval Streamlit scope classifier without loading the vector index.

Examples:
    python scripts/eval_scope_classifier.py --mode offline
    python scripts/eval_scope_classifier.py --mode live
    python scripts/eval_scope_classifier.py --mode live --model gpt-4o-mini
    # quick 30-case smoke suite:
    python scripts/eval_scope_classifier.py --mode live --cases data/eval/scope_classifier_dev.jsonl
    python scripts/eval_scope_classifier.py --mode live --classifier combined   # scope + queries, one call

The live command uses the project's normal gateway, cache, pricing, budget ledger and API-key
configuration. It measures only the scope-classification call, so retrieval/generation do not run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from supportdesk.config import Settings
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.stages.scope_classifier import LLMScopeClassifier
from supportdesk.pipelines.stages.scope_rewrite import CombinedScopeClassifier, ScopeRewrite
from supportdesk.types import CaseInput

DEFAULT_CASES = Path("data/eval/scope_classifier_benchmark.jsonl")
DEFAULT_PROMPT = Path("configs/prompts/scope_system.txt")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
                raise ValueError(f"{path}:{line_no}: invalid label {row.get('label')!r}")
            rows.append(row)
    return rows


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["expected"] == "IN_SCOPE" and r["predicted"] == "IN_SCOPE" for r in rows)
    tn = sum(r["expected"] == "OUT_OF_SCOPE" and r["predicted"] == "OUT_OF_SCOPE" for r in rows)
    fp = sum(r["expected"] == "OUT_OF_SCOPE" and r["predicted"] == "IN_SCOPE" for r in rows)
    fn = sum(r["expected"] == "IN_SCOPE" and r["predicted"] == "OUT_OF_SCOPE" for r in rows)
    total = len(rows)
    return {
        "n": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "in_scope_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "in_scope_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--prompt", type=Path, default=None,
                    help="the classifier's prompt (default: prompts.scope, or prompts.scope_rewrite for combined)")
    ap.add_argument("--classifier", choices=("llm", "combined"), default="llm",
                    help="llm: the separate scope call; combined: the call that also writes the concept queries")
    ap.add_argument("--mode", choices=("offline", "replay", "live"), default="offline")
    ap.add_argument("--model", default=None, help="Override models.chat for this benchmark")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = ap.parse_args()

    overrides = {"run.mode": args.mode, "pipeline.name": "scope_gate"}
    if args.model:
        overrides["models.chat"] = args.model
    settings = Settings.load(overrides=overrides)
    clients = build_clients(settings)
    if args.classifier == "combined":
        if args.prompt:
            settings = Settings.load(overrides={**overrides, "prompts.scope_rewrite": str(args.prompt)})
        classifier = CombinedScopeClassifier(ScopeRewrite(clients, settings))
    else:
        classifier = LLMScopeClassifier(clients, args.prompt or DEFAULT_PROMPT)
    source = load_rows(args.cases)
    results = []
    mark = clients.ledger.mark()
    started = time.perf_counter()

    for row in source:
        case = CaseInput(
            case_id=row["case_id"],
            title=row.get("title", ""),
            body=row.get("body", ""),
        )
        t0 = time.perf_counter()
        decision = classifier(case)
        results.append({
            "case_id": row["case_id"],
            "expected": row["label"],
            "predicted": decision.label,
            "correct": decision.label == row["label"],
            "reason": decision.reason,
            "method": decision.method,
            "latency_seconds": round(time.perf_counter() - t0, 4),
            "title": row.get("title", ""),
        })

    summary = {
        "mode": args.mode,
        "classifier": args.classifier,
        "model": settings.models.chat,
        "cases": str(args.cases),
        "metrics": metrics(results),
        "usage": clients.ledger.summarize(mark),
        "wall_seconds": round(time.perf_counter() - started, 4),
        "failures": [r for r in results if not r["correct"]],
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        m = summary["metrics"]
        u = summary["usage"]
        print(f"scope classifier ({args.classifier}): {m['n']} cases | model={summary['model']} | mode={args.mode}")
        print(
            f"accuracy={m['accuracy']:.1%}  IN precision={m['in_scope_precision']:.1%}  "
            f"IN recall={m['in_scope_recall']:.1%}  FN={m['fn']}  FP={m['fp']}"
        )
        print(
            f"calls={u['calls']}  input_tokens={u['input_tokens']}  output_tokens={u['output_tokens']}  "
            f"cost=${u['cost_usd']:.6f}  model_latency={u['latency_seconds']:.2f}s  "
            f"wall={summary['wall_seconds']:.2f}s"
        )
        if summary["failures"]:
            print("\nmisclassified:")
            for r in summary["failures"]:
                print(
                    f"  {r['case_id']} expected={r['expected']} got={r['predicted']} | "
                    f"{r['title']} | {r['reason']}"
                )
        else:
            print("\nall cases classified correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
