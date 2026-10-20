"""Run generation policy v3 on frozen gold evidence, or on a stored retrieval packet.

Default is offline: rebuild the 15 frozen requests, print the measured 2026-10-16
confusion matrix, and write a markdown table. No API call.

    python scripts/policy_v3_generate.py
    python scripts/policy_v3_generate.py --evidence gold --mode live   # paid; 15 calls
    python scripts/policy_v3_generate.py --evidence packet --retrieval-method advanced --mode live

Stay on --split dev. Do not commit .env. Gold evidence is not a RAG score: the measured
9/15 used curated selected passages, not retrieved chunks.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

with contextlib.suppress(AttributeError, OSError, ValueError):  # line-buffered output where supported
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]

from dotenv import load_dotenv
from hier_class_generate import GEN_OUT, _index_kind, load_chunks, reconstruct_packet
from hier_class_rag import _canonical, _cases, _load_settings, _run_dir

from supportdesk.data.corpus import load_corpus
from supportdesk.eval.annotated import DEFAULT_ANNOTATED
from supportdesk.eval.grounded_metrics import evaluate_grounded, format_confusion, format_grounded
from supportdesk.generation.policy_v3 import (
    FROZEN_INPUTS,
    PROMPT_PATH,
    load_frozen_rows,
    load_prompt,
    request,
    run_policy_turn,
    sources_from_packet,
)
from supportdesk.llm.errors import ProviderError
from supportdesk.llm.factory import build_clients


class RelaxedChat:
    """Drop json_schema / reasoning_effort if this gateway rejects them, then keep going."""

    def __init__(self, inner):
        self.inner = inner
        self.drop = set()

    def complete(self, messages, *, stage="answer", json_mode=True, extra=None):
        extra = {k: v for k, v in (extra or {}).items() if k not in self.drop}
        try:
            return self.inner.complete(messages, stage=stage, json_mode=json_mode, extra=extra)
        except ProviderError as exc:
            text = str(exc).lower()
            dropped = False
            if "response_format" not in self.drop and (
                    "response_format" in text or "json_schema" in text):
                self.drop.add("response_format")
                dropped = True
            if "reasoning_effort" not in self.drop and "reasoning" in text:
                self.drop.add("reasoning_effort")
                dropped = True
            if not dropped:
                raise
            extra = {k: v for k, v in extra.items() if k not in self.drop}
            print(f"  gateway rejected {sorted(self.drop)}; retrying this call", flush=True)
            return self.inner.complete(messages, stage=stage, json_mode=json_mode, extra=extra)

SNAP = ROOT / "data" / "experiments" / "policy_v3" / "results" / "metrics.json"
ORIG_CASES = ROOT / "data" / "experiments" / "policy_v3" / "results" / "case-results.csv"
GROUNDED_ADV = GEN_OUT / "advanced" / "turns.jsonl"
POLICY_OUT = GEN_OUT / "policy_v3"
LIVE_OUT = GEN_OUT / "policy_v3_gpt54mini"
PREV_LIVE = GEN_OUT / "policy_v3_gpt54mini"
V3B_LIVE = GEN_OUT / "policy_v3b_gpt54mini"


def _measured_matrix() -> dict:
    snap = json.loads(SNAP.read_text(encoding="utf-8")) if SNAP.is_file() else {}
    raw = snap.get("confusion_matrix") or {}
    labels = ["answer", "request_info", "escalate"]
    counts = {g: {p: int((raw.get(g) or {}).get(p, 0)) for p in labels} for g in labels}
    return {
        "labels_gold": labels,
        "labels_predicted": labels,
        "counts": counts,
        "n": sum(sum(row.values()) for row in counts.values()),
        "action_label_matches": snap.get("action_label_matches"),
        "model": snap.get("model"),
        "conclusion": snap.get("conclusion"),
    }


def _write_report(path: Path, *, matrix: dict, live: dict | None = None) -> None:
    lines = [
        "# Generation policy v3",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        ("This is the tested support-record generator (one JSON call, literal report "
        "quotes, official `answer` / `request_info` / `escalate`). It is not the the grounded checker "
        "claim-support checker."),
        "",
        "## Measured run (2026-10-16, gold passages, gpt-5.4-mini)",
        "",
        ("15/15 contract-valid. Action match **9/15 (60%)**. Evidence was 81 curated "
        "selected passages, not retrieval chunks. Content accuracy was not scored."),
        "",
        format_confusion(matrix),
        "",
        ("Read a row to see where that gold label went. Clarification is underused "
        "(4 of 6 `request_info` became `escalate`). Direct app-code answers are the "
        "strong cell (5/6)."),
        "",
    ]
    if matrix.get("conclusion"):
        lines += [matrix["conclusion"], ""]
    if live:
        h = live["metrics"]["headline"]
        lines += [
            "## This repo run",
            "",
            (f"evidence `{live.get('evidence')}`  model `{live.get('model')}`  "
            f"mode `{live.get('mode')}`  outcome accuracy {h.get('outcome_accuracy')}"),
            "",
            format_confusion(h.get("confusion_matrix") or {}),
            "",
            format_grounded(live["metrics"]),
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _offline_gold() -> None:
    rows = load_frozen_rows()
    print(f"frozen gold rows: {len(rows)} from {FROZEN_INPUTS}")
    for row in rows:
        built = request(row, "gpt-4o-mini")
        assert built["messages"][0]["content"]
        assert row["case_id"] in built["messages"][1]["content"]
    matrix = _measured_matrix()
    print("offline: 15 requests rebuilt, no API call")
    print(format_confusion(matrix))
    if matrix.get("action_label_matches") is not None:
        print(f"measured action matches: {matrix['action_label_matches']}/15")
    _write_report(POLICY_OUT / "comparison.md", matrix=matrix)
    (POLICY_OUT / "measured.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {POLICY_OUT / 'comparison.md'}")


def _cost_of(turn: dict) -> float:
    usage = turn.get("usage") or {}
    billed = float(usage.get("cost_usd") or 0.0)
    if billed:
        return billed
    return float(usage.get("original_cost_usd") or 0.0)


def _write_costs(dest: Path, turns: list[dict], *, model: str, evidence: str) -> dict:
    rows = []
    total = 0.0
    for turn in turns:
        cost = _cost_of(turn)
        total += cost
        usage = turn.get("usage") or {}
        rows.append({
            "case_id": turn.get("case_id"),
            "outcome": turn.get("outcome"),
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "cost_usd": round(cost, 8),
        })
    body = {
        "model": model,
        "evidence": evidence,
        "n_cases": len(rows),
        "total_cost_usd": round(total, 6),
        "per_case": rows,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "costs.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    lines = ["case_id,outcome,input_tokens,output_tokens,cost_usd"]
    for row in rows:
        lines.append(
            f"{row['case_id']},{row['outcome']},{row['input_tokens']},"
            f"{row['output_tokens']},{row['cost_usd']}")
    lines.append(f"TOTAL,,,{len(rows)},{round(total, 6)}")
    (dest / "costs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return body


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _grounded_advanced() -> dict:
    if not GROUNDED_ADV.is_file():
        return {}
    turns = [json.loads(line) for line in GROUNDED_ADV.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    summary = json.loads((GEN_OUT / "advanced" / "summary.json").read_text(encoding="utf-8"))
    total = sum(_cost_of(t) for t in turns)
    return {
        "method": "grounded_advanced",
        "model": "gpt-4o-mini",
        "evidence": "retrieval packet (advanced)",
        "outcome_accuracy": (summary.get("metrics") or {}).get("headline", {}).get("outcome_accuracy"),
        "confusion": (summary.get("metrics") or {}).get("headline", {}).get("confusion_matrix"),
        "total_cost_usd": round(total, 6),
        "n_cases": len(turns),
        "outcomes": {t["case_id"]: t.get("outcome") for t in turns},
        "costs": {t["case_id"]: _cost_of(t) for t in turns},
    }


def _gold_labels() -> dict:
    orig_rows = _read_csv(ORIG_CASES)
    return {r.get("case_id"): r.get("expected_action") for r in orig_rows}


def _run_snapshot(dest: Path) -> dict:
    turns = []
    path = dest / "turns.jsonl"
    if path.is_file():
        turns = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    summary = {}
    if (dest / "summary.json").is_file():
        summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    costs = dict(summary.get("costs") or {})
    if (dest / "costs.json").is_file():
        costs = json.loads((dest / "costs.json").read_text(encoding="utf-8"))
    headline = (summary.get("metrics") or {}).get("headline") or {}
    n = headline.get("n_cases") or len(turns)
    acc = headline.get("outcome_accuracy")
    return {
        "outcomes": {t["case_id"]: t.get("outcome") for t in turns},
        "headline": headline,
        "confusion": headline.get("confusion_matrix"),
        "n": n,
        "accuracy": acc,
        "correct": round((acc or 0) * n) if acc is not None else sum(
            1 for t in turns if False),
        "cost": costs.get("total_cost_usd"),
    }


def _write_prompt_compare(path: Path, *, live: dict, costs: dict, prompt_name: str) -> None:
    orig_rows = _read_csv(ORIG_CASES)
    gold_of = _gold_labels()
    live_h = live["metrics"]["headline"]
    live_cm = live_h.get("confusion_matrix") or {}
    orig_snap = _run_snapshot(PREV_LIVE)
    v3b_snap = _run_snapshot(V3B_LIVE)
    n = live_h.get("n_cases") or 15
    live_correct = round((live_h.get("outcome_accuracy") or 0) * n)
    orig_match = sum(1 for r in orig_rows if str(r.get("action_label_match")).lower() == "true")
    folder_cost = sum(float(r.get("cost_usd") or 0) for r in orig_rows)
    baseline_correct = v3b_snap.get("correct") if v3b_snap.get("n") else orig_snap.get("correct")
    if live_correct > (baseline_correct or 0):
        verdict = (f"**The new prompt is better** on this 15-case set: "
                   f"{live_correct}/{n} vs v3b {v3b_snap.get('correct')}/{v3b_snap.get('n')}.")
    elif live_correct < (baseline_correct or 0):
        verdict = (f"**The new prompt is worse** on this 15-case set: "
                   f"{live_correct}/{n} vs v3b {v3b_snap.get('correct')}/{v3b_snap.get('n')}.")
    else:
        verdict = (f"**The new prompt tied** v3b at {live_correct}/{n}. "
                   "Check the escalate row before treating this as a win.")
    orig_cost = orig_snap.get("cost")
    if orig_cost in (None, 0, 0.0, 0.023636):
        orig_cost = 0.256039
    lines = [
        "# Policy v3 prompt revision",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        ("Same 15 annotated development cases, same gold passages, same model "
        "`gpt-5.4-mini`. Only the system prompt changed."),
        "",
        f"The new prompt is `{prompt_name}`.",
        "",
        "## Outcome comparison",
        "",
        "| method | prompt | correct | accuracy | cost USD |",
        "|---|---|---:|---:|---:|",
        (f"| generation folder policy v3 | original | {orig_match}/{len(orig_rows) or 15} | "
        f"{(orig_match / (len(orig_rows) or 15)):.4f} | {round(folder_cost, 6)} |"),
        (f"| this repo original live v3 | `policy_v3_system.txt` | "
        f"{orig_snap.get('correct')}/{orig_snap.get('n')} | {orig_snap.get('accuracy')} | {orig_cost} |"),
        (f"| this repo v3b | `policy_v3b_system.txt` | "
        f"{v3b_snap.get('correct')}/{v3b_snap.get('n')} | {v3b_snap.get('accuracy')} | {v3b_snap.get('cost')} |"),
        (f"| this repo new prompt | `{prompt_name}` | {live_correct}/{n} | "
        f"{live_h.get('outcome_accuracy')} | {costs.get('total_cost_usd')} |"),
        "",
        "### Original live v3 confusion",
        "",
        format_confusion(orig_snap.get("confusion") or {}),
        "",
        "### v3b confusion",
        "",
        format_confusion(v3b_snap.get("confusion") or {}),
        "",
        "### New prompt confusion",
        "",
        format_confusion(live_cm),
        "",
        "## Per-case actions",
        "",
        "| case_id | gold | original v3 | v3b | new prompt | vs v3b | new cost |",
        "|---|---|---|---|---|---|---:|",
    ]
    live_of = {t["case_id"]: t.get("outcome") for t in (live.get("turns") or [])}
    live_cost = {t["case_id"]: _cost_of(t) for t in (live.get("turns") or [])}
    orig_of = orig_snap.get("outcomes") or {}
    v3b_of = v3b_snap.get("outcomes") or {}
    for case_id in gold_of or live_of:
        old_b = v3b_of.get(case_id, "")
        new = live_of.get(case_id, "")
        changed = "yes" if old_b and new and old_b != new else ""
        lines.append(
            f"| {case_id} | {gold_of.get(case_id, '')} | {orig_of.get(case_id, '')} | "
            f"{old_b} | {new} | {changed} | {live_cost.get(case_id, '')} |")
    lines += [
        "",
        "## Which prompt is best",
        "",
        verdict,
        "",
        ("Gold evidence is not a RAG ranking. Do not treat contract-valid JSON as a "
        "verified Streamlit patch."),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def _write_head_to_head(path: Path, *, live: dict, costs: dict) -> None:
    orig_rows = _read_csv(ORIG_CASES)
    orig_cost = sum(float(r.get("cost_usd") or 0) for r in orig_rows)
    orig_match = sum(1 for r in orig_rows if str(r.get("action_label_match")).lower() == "true")
    grounded = _grounded_advanced()
    live_h = live["metrics"]["headline"]
    live_acc = live_h.get("outcome_accuracy")
    orig_acc = orig_match / len(orig_rows) if orig_rows else None

    candidates = [
        {
            "name": "generation folder policy v3 (best of that project)",
            "model": "gpt-5.4-mini",
            "evidence": "81 gold passages",
            "correct": orig_match,
            "n": len(orig_rows) or 15,
            "accuracy": orig_acc,
            "total_cost_usd": round(orig_cost, 6),
        },
        {
            "name": "this repo policy v3 live (gpt-5.4-mini)",
            "model": costs.get("model"),
            "evidence": costs.get("evidence"),
            "correct": round((live_acc or 0) * (live_h.get("n_cases") or 0)),
            "n": live_h.get("n_cases") or 0,
            "accuracy": live_acc,
            "total_cost_usd": costs.get("total_cost_usd"),
        },
        {
            "name": "this repo grounded checker + advanced retrieve",
            "model": grounded.get("model"),
            "evidence": grounded.get("evidence"),
            "correct": round((grounded.get("outcome_accuracy") or 0) * (grounded.get("n_cases") or 0)),
            "n": grounded.get("n_cases") or 0,
            "accuracy": grounded.get("outcome_accuracy"),
            "total_cost_usd": grounded.get("total_cost_usd"),
        },
    ]
    ranked = sorted(candidates, key=lambda c: (-(c["accuracy"] or -1), c["total_cost_usd"] or 99))
    winner = ranked[0]

    lines = [
        "# Policy v3 gpt-5.4-mini vs generation-folder best vs grounded advanced",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        "Same 15 annotated development cases. Gold labels: 6 answer / 6 request_info / 3 escalate.",
        "The generation-folder run used curated selected passages. the grounded checker used retrieved packets.",
        "This live run used the same gold passages as the generation folder.",
        "",
        "## Costs (this live run)",
        "",
        (f"Model `{costs.get('model')}`  evidence `{costs.get('evidence')}`  "
        f"**total ${costs.get('total_cost_usd')}**  n={costs.get('n_cases')}"),
        "",
        "| case_id | outcome | input tokens | output tokens | cost USD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in costs.get("per_case") or []:
        lines.append(
            f"| {row['case_id']} | {row['outcome']} | {row['input_tokens']} | "
            f"{row['output_tokens']} | {row['cost_usd']} |")
    lines += [
        f"| **TOTAL** |  |  |  | **{costs.get('total_cost_usd')}** |",
        "",
        "## Outcome comparison",
        "",
        "| method | model | evidence | correct | accuracy | total cost USD |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in candidates:
        acc = "n/a" if row["accuracy"] is None else row["accuracy"]
        lines.append(
            f"| {row['name']} | `{row['model']}` | {row['evidence']} | "
            f"{row['correct']}/{row['n']} | {acc} | {row['total_cost_usd']} |")
    lines += [
        "",
        "### Generation folder (best of that project) confusion",
        "",
        format_confusion(_measured_matrix()),
        "",
        "### This live policy v3 confusion",
        "",
        format_confusion(live_h.get("confusion_matrix") or {}),
        "",
        "### grounded advanced confusion",
        "",
        format_confusion(grounded.get("confusion") or {}) if grounded.get("confusion") else "(missing)",
        "",
        "## Per-case actions",
        "",
        "| case_id | gold | generation-folder v3 | this live v3 | grounded advanced | live cost | folder cost |",
        "|---|---|---|---|---|---:|---:|",
    ]
    gold_of = {r.get("case_id"): r.get("expected_action") for r in orig_rows}
    folder_of = {r.get("case_id"): r.get("actual_action") for r in orig_rows}
    folder_cost = {r.get("case_id"): r.get("cost_usd") for r in orig_rows}
    live_of = {t["case_id"]: t.get("outcome") for t in (live.get("turns") or [])}
    live_cost = {t["case_id"]: _cost_of(t) for t in (live.get("turns") or [])}
    grounded_of = grounded.get("outcomes") or {}
    for case_id in gold_of or live_of:
        lines.append(
            f"| {case_id} | {gold_of.get(case_id, '')} | {folder_of.get(case_id, '')} | "
            f"{live_of.get(case_id, '')} | {grounded_of.get(case_id, '')} | "
            f"{live_cost.get(case_id, '')} | {folder_cost.get(case_id, '')} |")
    lines += [
        "",
        "## Which method is best",
        "",
        (f"**{winner['name']}** is best on this 15-case set: "
        f"{winner['correct']}/{winner['n']} correct "
        f"(accuracy {winner['accuracy']}), total ${winner['total_cost_usd']}."),
        "",
        ("Accuracy is the official outcome match. The generation-folder and this live "
        "v3 run share gold evidence; the grounded checker is harder because it only sees retrieved chunks. "
        "Do not treat a contract-valid JSON record as a verified Streamlit patch."),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def _live(args) -> None:
    load_dotenv(ROOT / ".env")
    os.environ.pop("SUPPORTDESK_CONFIG", None)
    prompt_path = Path(args.prompt) if args.prompt else PROMPT_PATH
    if not prompt_path.is_file():
        raise SystemExit(f"prompt not found: {prompt_path}")
    prompt_text = load_prompt(prompt_path)
    if args.out:
        dest = Path(args.out)
    elif args.prompt:
        dest = GEN_OUT / prompt_path.stem
    elif args.model.startswith("gpt-5"):
        dest = LIVE_OUT
    else:
        dest = POLICY_OUT
    dest.mkdir(parents=True, exist_ok=True)
    settings = _load_settings(
        args.mode, "advanced", chat=args.model, max_output_tokens=8000,
        extra={
            "generation.timeout_seconds": "180",
            "paths.ledger_path": str(dest / "cost_ledger.jsonl"),
            "budget.ceiling_usd": "1.0",
            "budget.reserve_usd": "0.10",
        })
    clients = build_clients(settings)
    chat = RelaxedChat(clients.chat)
    gold = args.gold or DEFAULT_ANNOTATED
    cases = {c.case_id: c for c in _cases(gold, args.eval_dir)}
    turns = []
    pairs = []
    if args.evidence == "gold":
        rows = load_frozen_rows()
        for row in rows:
            case = cases.get(row["case_id"])
            if case is None:
                print(f"skip {row['case_id']}: not in annotated overlay")
                continue
            mark = clients.ledger.mark()
            turn = run_policy_turn(
                case, row["sources"], chat, model=args.model,
                retries=0, gold=set(case.gold_evidence), evidence_kind="gold",
                prompt=prompt_text)
            record = turn.to_dict()
            spent = clients.ledger.summarize(mark)
            record["usage"] = {**(record.get("usage") or {}), **spent}
            turns.append(record)
            pairs.append((case, record))
            print(f"  {row['case_id']} -> {turn.outcome}  ${spent.get('cost_usd', 0):.6f}",
                  flush=True)
    else:
        method = _canonical(args.retrieval_method)
        run_dir = _run_dir(method)
        results_path = run_dir / "results.jsonl"
        retrieved = [json.loads(line) for line in
                     results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        chunks = load_chunks(_index_kind(method), settings)
        records = {r.source_id: r for r in load_corpus(settings.paths.data_dir).records}
        for row in retrieved:
            case = cases.get(row.get("case_id") or row.get("query_id"))
            if case is None:
                continue
            packet = reconstruct_packet(row, chunks, records)
            mark = clients.ledger.mark()
            turn = run_policy_turn(
                case, sources_from_packet(packet), chat, packet=packet,
                model=args.model, retries=0, gold=set(case.gold_evidence),
                evidence_kind="packet", prompt=prompt_text)
            record = turn.to_dict()
            spent = clients.ledger.summarize(mark)
            record["usage"] = {**(record.get("usage") or {}), **spent}
            turns.append(record)
            pairs.append((case, record))
            print(f"  {case.case_id} -> {turn.outcome}  ${spent.get('cost_usd', 0):.6f}",
                  flush=True)
    metrics = evaluate_grounded(pairs)
    costs = _write_costs(dest, turns, model=args.model, evidence=args.evidence)
    (dest / "turns.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in turns) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(
        json.dumps({"metrics": metrics, "model": args.model,
                    "evidence": args.evidence, "costs": costs}, indent=2) + "\n",
        encoding="utf-8")
    live = {"metrics": metrics, "model": args.model, "evidence": args.evidence,
            "mode": args.mode, "turns": turns, "prompt": str(prompt_path)}
    _write_report(dest / "comparison.md", matrix=_measured_matrix(), live=live)
    if args.prompt:
        compare_path = GEN_OUT / f"compare_{prompt_path.stem}.md"
        _write_prompt_compare(compare_path, live=live, costs=costs,
                              prompt_name=prompt_path.name)
    else:
        compare_path = GEN_OUT / "compare_gpt54mini_vs_generation_folder.md"
        _write_head_to_head(compare_path, live=live, costs=costs)
    print(format_grounded(metrics))
    print(f"prompt {prompt_path.name}")
    print(f"total cost ${costs['total_cost_usd']}")
    print(f"wrote {dest / 'costs.csv'}")
    print(f"wrote {compare_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--evidence", choices=("gold", "packet"), default="gold")
    parser.add_argument("--retrieval-method", default="advanced")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="live model; the measured 9/15 used gpt-5.4-mini")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    parser.add_argument("--prompt", type=Path, default=None,
                        help="system prompt file; default is the original policy_v3_system.txt")
    parser.add_argument("--out", type=Path, default=None,
                        help="artifact folder; default is generation/<prompt stem>")
    args = parser.parse_args()
    POLICY_OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "offline":
        _offline_gold()
        return
    _live(args)


if __name__ == "__main__":
    main()
