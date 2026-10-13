"""A/B latency check for current advanced retrieval with vs. without the scope classifier.

Uses the same advanced profile/settings and same retriever for both arms. A calls build_advanced
(ungated); B calls build_advanced_scoped (gated). This avoids changing the public pipeline registry
just to obtain a control arm.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

from supportdesk.config import Settings
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.advanced import build_advanced
from supportdesk.pipelines.scope_gate import build_advanced_scoped
from supportdesk.registry import make_retriever
from supportdesk.retrieval.index import VectorIndex
from supportdesk.types import CaseInput

CASES = [
    ("in_data_editor_key", "IN_SCOPE", "I get an error when assigning a value to the same key used by st.data_editor."),
    ("in_form_rerun", "IN_SCOPE", "My Streamlit page reruns on every form interaction; I want all inputs submitted together."),
    ("in_implicit_widget_pages", "IN_SCOPE", "My checkbox keeps forgetting its value whenever I switch between pages. How can I keep it?"),
    ("in_session_state_button", "IN_SCOPE", "Why does st.session_state reset after clicking a button?"),
    ("in_streamlit_with_noise_code", "IN_SCOPE", "I am debugging a Streamlit multipage app. The widget value disappears between pages. This unrelated helper is also in the project: numbers = [5,1,4,2,3]; numbers.sort(). Why is the widget state lost?"),
    ("out_flask", "OUT_OF_SCOPE", "How do I create a REST API with Flask?"),
    ("out_general_knowledge", "OUT_OF_SCOPE", "What is the capital of Germany?"),
    ("out_pandas", "OUT_OF_SCOPE", "How do I merge two pandas DataFrames on multiple columns?"),
    ("out_python_code_only", "OUT_OF_SCOPE", "Why does this not work? items = [3,1,2]; items.sort(); print(items)"),
    ("out_python_list", "OUT_OF_SCOPE", "How do I reverse a Python list?"),
]


def result_row(case_id, expected, arm, result, elapsed):
    usage = result.usage or {}
    trace = result.trace or {}
    extra = trace.get("extra", {}) or {}
    scope = extra.get("scope", {}) or {}
    retrieved = result.retrieved or []
    return {
        "case_id": case_id,
        "expected": expected,
        "arm": arm,
        "wall_s": float(usage.get("wall_seconds", elapsed) or elapsed),
        "latency_s": float(usage.get("latency_seconds", 0) or 0),
        "calls": int(usage.get("calls", 0) or 0),
        "cost_usd": float(usage.get("cost_usd", 0) or 0),
        "scope_label": scope.get("label"),
        "n_queries": int(trace.get("n_queries", 0) or 0),
        "retrieved_count": len(retrieved),
        "top_chunk": retrieved[0].get("chunk_id") if retrieved else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(CASES))
    ap.add_argument("--out", default="artifacts/ab_scope_latency")
    args = ap.parse_args()

    settings = Settings.load(overrides={"pipeline.name": "advanced", "run.mode": "live"})
    index = VectorIndex.load(settings.paths.index_dir, expect_model=settings.models.embedding)

    # Keep A/B caches and ledgers separate so the second arm cannot look faster just because
    # the first arm warmed a shared model-response cache.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths_a = dataclasses.replace(
        settings.paths, cache_path=out / "cache_a.sqlite", ledger_path=out / "ledger_a.jsonl")
    paths_b = dataclasses.replace(
        settings.paths, cache_path=out / "cache_b.sqlite", ledger_path=out / "ledger_b.jsonl")
    settings_a = dataclasses.replace(settings, paths=paths_a)
    settings_b = dataclasses.replace(settings, paths=paths_b)
    clients_a = build_clients(settings_a)
    clients_b = build_clients(settings_b)
    retriever_a = make_retriever(settings_a.pipeline.retriever, index, clients_a, settings_a)
    retriever_b = make_retriever(settings_b.pipeline.retriever, index, clients_b, settings_b)
    arm_a = build_advanced(retriever_a, clients_a, settings_a)
    arm_b = build_advanced_scoped(retriever_b, clients_b, settings_b)

    rows = []
    for i, (case_id, expected, body) in enumerate(CASES[: args.limit]):
        case = CaseInput(case_id=case_id, title=body, body="")
        order = [("A", arm_a), ("B", arm_b)] if i % 2 == 0 else [("B", arm_b), ("A", arm_a)]
        for arm, pipe in order:
            t0 = time.perf_counter()
            result = pipe.run(case)
            rows.append(result_row(case_id, expected, arm, result, time.perf_counter() - t0))

    pairs = []
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["arm"]] = r
    for case_id, arms in by_case.items():
        if "A" not in arms or "B" not in arms:
            continue
        a, b = arms["A"], arms["B"]
        pairs.append({
            "case_id": case_id,
            "expected": a["expected"],
            "a_wall": a["wall_s"], "b_wall": b["wall_s"],
            "delta_wall": b["wall_s"] - a["wall_s"],
            "a_cost": a["cost_usd"], "b_cost": b["cost_usd"],
            "delta_cost": b["cost_usd"] - a["cost_usd"],
            "a_calls": a["calls"], "b_calls": b["calls"],
            "delta_calls": b["calls"] - a["calls"],
            "scope_label": b["scope_label"],
            "scope_correct": b["scope_label"] == a["expected"],
            "a_top_chunk": a["top_chunk"], "b_top_chunk": b["top_chunk"],
            "top1_same": a["top_chunk"] == b["top_chunk"],
            "b_retrieved_count": b["retrieved_count"],
        })

    ins = [p for p in pairs if p["expected"] == "IN_SCOPE"]
    outs = [p for p in pairs if p["expected"] == "OUT_OF_SCOPE"]
    avg = lambda xs, key: sum(x[key] for x in xs) / len(xs) if xs else 0.0
    summary = {
        "cases": len(pairs),
        "classifier_accuracy_pct": 100.0 * sum(p["scope_correct"] for p in pairs) / len(pairs) if pairs else 0.0,
        "in_scope": {
            "n": len(ins),
            "a_avg_wall_s": avg(ins, "a_wall"),
            "b_avg_wall_s": avg(ins, "b_wall"),
            "avg_added_wall_s": avg(ins, "delta_wall"),
            "avg_added_cost_usd": avg(ins, "delta_cost"),
            "avg_added_calls": avg(ins, "delta_calls"),
            "top1_same_pct": 100.0 * sum(p["top1_same"] for p in ins) / len(ins) if ins else 0.0,
        },
        "out_of_scope": {
            "n": len(outs),
            "a_avg_wall_s": avg(outs, "a_wall"),
            "b_avg_wall_s": avg(outs, "b_wall"),
            "avg_time_saved_s": -avg(outs, "delta_wall"),
            "avg_cost_saved_usd": -avg(outs, "delta_cost"),
            "avg_calls_saved": -avg(outs, "delta_calls"),
            "all_stopped_before_retrieval": all(p["b_retrieved_count"] == 0 for p in outs),
        },
        "overall": {
            "a_avg_wall_s": avg(pairs, "a_wall"),
            "b_avg_wall_s": avg(pairs, "b_wall"),
            "avg_net_wall_delta_s": avg(pairs, "delta_wall"),
            "a_total_cost_usd": sum(p["a_cost"] for p in pairs),
            "b_total_cost_usd": sum(p["b_cost"] for p in pairs),
            "net_cost_delta_usd": sum(p["delta_cost"] for p in pairs),
        },
        "pairs": pairs,
    }

    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
