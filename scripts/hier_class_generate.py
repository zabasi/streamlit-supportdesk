"""Grounded generation on top of finished hier_class_rag retrieval runs.

Reuses each method's stored packet (no second retrieve). Asks the model with the HW03
delimited <question>/<evidence> schema, parses without repair, validates citations and
claim support, and refuses when the check fails.

    python scripts/hier_class_generate.py --mode live
    python scripts/hier_class_generate.py --mode live --methods advanced,llm_llm
    python scripts/hier_class_generate.py --mode offline   # extractive stand-in; not a real score

Stay on --split dev. Do not commit .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]

from dotenv import load_dotenv

from hier_class_rag import (
    METHOD_LABELS,
    OUT,
    _canonical,
    _cases,
    _load_settings,
    _run_dir,
)
from supportdesk.data.corpus import load_corpus
from supportdesk.eval.grounded_metrics import evaluate_grounded, format_confusion, format_grounded
from supportdesk.generation.grounded import ExtractiveAnswerer, run_grounded_turn
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.packet import Packet
from supportdesk.tokenizer import count_tokens
from supportdesk.types import Chunk

# Current system, best 2x2, only ablation win, the llmchunk+llmfilter combo.
DEFAULT_METHODS = (
    "advanced",
    "llm_llm",
    "advanced_rule_kmeans",
    "advanced_llmchunk_llmfilter",
)
GEN_OUT = OUT / "generation"


def _index_kind(method: str) -> str:
    name = _canonical(method)
    if name in ("rule_kmeans", "rule_llm") or name.startswith("advanced_rule"):
        return "rule"
    if name in ("llm_kmeans", "llm_llm") or "llmchunk" in name:
        return "llm"
    return "existing"


def _chunk_paths(kind: str, settings) -> list[Path]:
    if kind == "rule":
        return [OUT / "indexes" / "rule" / "chunks.jsonl"]
    if kind == "llm":
        return [OUT / "indexes" / "llm" / "chunks.jsonl"]
    return [Path(settings.paths.index_dir) / "chunks.jsonl",
            ROOT / "index" / "chunks.jsonl"]


def load_chunks(kind: str, settings) -> dict[str, Chunk]:
    for path in _chunk_paths(kind, settings):
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            print(f"  chunks from {path} ({len(rows)})")
            return {row["chunk_id"]: Chunk.from_dict(row) for row in rows}
    print(f"  no chunks.jsonl for {kind}; will slice corpus text from stored offsets")
    return {}


def reconstruct_packet(row: dict, chunks: dict[str, Chunk], records: dict) -> Packet:
    packet = Packet()
    for item in row.get("packet") or []:
        cid = item["chunk_id"]
        chunk = chunks.get(cid)
        if chunk is None:
            record = records.get(item.get("source_id"))
            start, end = item.get("start_char"), item.get("end_char")
            text = ""
            if record is not None and start is not None and end is not None:
                text = record.text[start:end]
            chunk = Chunk(
                chunk_id=cid, source_id=item.get("source_id") or cid,
                source_type=getattr(record, "source_type", None) or "documentation",
                title=getattr(record, "title", None) or item.get("source_id") or cid,
                text=text, url=getattr(record, "url", None) or "",
                parent_source_id=item.get("parent_source_id") or getattr(record, "parent_source_id", None),
                start_char=start, end_char=end)
        if chunk.chunk_id in {c.chunk_id for c in packet.chunks}:
            continue
        packet.chunks.append(chunk)
        packet.labels[f"S{len(packet.chunks)}"] = chunk.chunk_id
        packet.token_count += count_tokens(chunk.text)
    return packet


def _write_comparison(rows: dict[str, dict], path: Path) -> None:
    lines = [
        "# Grounded generation on hier-class retrieval",
        "",
        (
            f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            "Same annotated dev gold. Retrieval packets reused; only the answer call is new."
        ),
        "",
        (
            "A grounded assistant is not one that cites; it is one that can be checked, "
            "and refuses when the check fails (Week 3 HW03 / class lab)."
        ),
        "",
        (
            "| method | answer | request_info | escalate | validation_failed | "
            "outcome accuracy | answerable | ambiguous | escalation | claim support | "
            "citation precision |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, body in rows.items():
        h = body["metrics"]["headline"]
        oc = h.get("outcomes") or {}

        def cell(key):
            value = h.get(key)
            return "n/a" if value is None else value

        by = h.get("by_category") or {}

        def cat(key):
            row = by.get(key) or {}
            acc = row.get("outcome_accuracy")
            return "n/a" if acc is None else acc

        lines.append(
            (
                f"| `{name}` | {oc.get('answer', 0)} | {oc.get('request_info', 0)} | "
                f"{oc.get('escalate', 0)} | {oc.get('validation_failed', 0)} | "
                f"{cell('outcome_accuracy')} | {cat('answerable')} | {cat('ambiguous')} | "
                f"{cat('escalation')} | {cell('claim_support_rate')} | "
                f"{cell('citation_precision')} |"
            ))
    lines += [
        "",
        "## How to read the numbers",
        "",
        (
            "- **answer / request_info / escalate**: the official support decision. Gold is the "
            "eval case category (`answerable` → answer, `ambiguous` → request_info, "
            "`escalation` → escalate)."
        ),
        (
            "- **outcome accuracy**: share of cases whose delivered decision is in "
            "`acceptable_outcomes`. Split by category in the next three columns."
        ),
        (
            "- **confusion matrix** (below): rows are gold, columns are what the model "
            "delivered. Diagonal is correct. `validation_failed` is a blocked ungrounded "
            "`answer`, not a gold label."
        ),
        (
            "- **claim support**: share of claims whose words appear in the *cited* packet passages. "
            "HW03 bar was ≥ 0.90."
        ),
        (
            "- **citation precision**: share of cited ids that are in this turn's packet. "
            "HW03 bar was ≥ 0.95; outside packet must be 0."
        ),
        "- **answered after failure** must be 0: a rejected payload is never shown.",
        (
            "- **abstention F1**: should refuse when the packet has no gold source; should answer "
            "when it does. Precision + recall, because refusing everything or answering everything "
            "both look good on one number alone."
        ),
        (
            "- **completeness vs gold spans**: of the reviewer-selected words, how many appear in "
            "the delivered answer. Retrieval can be fine and this still low if the model hedges."
        ),
        (
            "- **answer when packet had gold**: of the cases where retrieval actually put gold in "
            "the packet, how many produced a delivered answer. This is the generation layer, not "
            "recall@6."
        ),
        "",
        (
            "Links: [Pinecone RAG evaluation](https://www.pinecone.io/learn/series/rag/rag-evaluation/) "
            "· [RARR, Gao et al.](https://arxiv.org/abs/2305.14627) · "
            "[F1](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))"
        ),
        "",
        "## Confusion matrices (gold \\ predicted)",
        "",
        "Each cell is a case count. Read a row to see where that gold label went.",
        "",
    ]
    for name, body in rows.items():
        matrix = (body.get("metrics") or {}).get("headline", {}).get("confusion_matrix")
        lines += [f"### `{name}`", ""]
        if matrix:
            lines += [format_confusion(matrix), ""]
        else:
            lines += ["(no labeled outcomes)", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    if os.environ.get("SUPPORTDESK_CONFIG", "").endswith("sample.toml"):
        print("WARNING: SUPPORTDESK_CONFIG points at the sample corpus; unset it for the real eval.",
              file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("offline", "replay", "live"), default="offline")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    from supportdesk.eval.annotated import DEFAULT_ANNOTATED
    gold = args.gold or DEFAULT_ANNOTATED
    wanted = [_canonical(m.strip()) for m in args.methods.split(",") if m.strip()]

    settings = _load_settings(args.mode, "advanced")
    system = (ROOT / "configs" / "prompts" / "grounded_system.txt").read_text(encoding="utf-8")
    developer = (ROOT / "configs" / "prompts" / "grounded_developer.txt").read_text(encoding="utf-8")
    cases = _cases(gold, args.eval_dir)
    if args.limit:
        cases = cases[: args.limit]
    by_id = {c.case_id: c for c in cases}
    records = {r.source_id: r for r in load_corpus(settings.paths.data_dir).records}
    if args.mode == "offline":
        chat = ExtractiveAnswerer()
        print("offline: extractive stand-in (class-lab style). Not a real model score.")
    else:
        chat = build_clients(settings).chat

    rows: dict[str, dict] = {}
    GEN_OUT.mkdir(parents=True, exist_ok=True)
    for method in wanted:
        dest = GEN_OUT / method
        summary_path = dest / "summary.json"
        if summary_path.exists() and not args.rebuild:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows[method] = {"metrics": summary["metrics"], "run": str(dest)}
            print(f"\n=== {method} (reused) ===")
            print(format_grounded(summary["metrics"]))
            continue
        retrieval = _run_dir(method)
        results_path = retrieval / "results.jsonl"
        if not results_path.exists():
            print(f"\n=== {method} SKIP: no retrieval results at {results_path} ===")
            continue
        print(f"\n=== {method} ===")
        print(f"  {METHOD_LABELS.get(method, '')}")
        print(f"  retrieval {retrieval}")
        chunks = load_chunks(_index_kind(method), settings)
        retrieved = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        pairs = []
        dest.mkdir(parents=True, exist_ok=True)
        turns_path = dest / "turns.jsonl"
        started = time.perf_counter()
        with turns_path.open("w", encoding="utf-8") as handle:
            for row in retrieved:
                case = by_id.get(row["case_id"])
                if case is None:
                    continue
                packet = reconstruct_packet(row, chunks, records)
                gold_ids = set(case.gold_evidence)
                turn = run_grounded_turn(
                    case.input, packet, chat, system=system, developer=developer,
                    retries=1 if args.mode != "offline" else 0, gold=gold_ids)
                record = {
                    **turn.to_dict(),
                    "method": method,
                    "category": case.category,
                    "split": case.split,
                    "retrieval_outcome": row.get("outcome"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                pairs.append((case, record))
                print(f"  {case.case_id} {turn.outcome}"
                      f"{' ' + (turn.abstention_reason or '') if turn.abstained else ''}"
                      f" citations={len(turn.citations)} claims={len(turn.claims)}")
        metrics = evaluate_grounded(pairs)
        summary = {
            "run_id": f"{method}_grounded",
            "method": method,
            "mode": args.mode,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_cases": len(pairs),
            "metrics": metrics,
            "retrieval_run": str(retrieval),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        rows[method] = {"metrics": metrics, "run": str(dest)}
        print(format_grounded(metrics))
        print(f"wall {time.perf_counter() - started:.1f}s -> {dest}")

    if not rows:
        raise SystemExit("no generation runs (missing retrieval results?)")
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": args.mode, "methods": {k: v["metrics"]["headline"] for k, v in rows.items()}}
    (GEN_OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    _write_comparison(rows, GEN_OUT / "comparison.md")
    print(f"\nwrote {GEN_OUT / 'results.json'} and {GEN_OUT / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
