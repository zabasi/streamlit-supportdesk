"""Run a pipeline over evaluation cases and write the artifacts under ``<artifacts_dir>/runs/<run_id>/``."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, redact
from ..llm.errors import BudgetExhausted, CallLimitReached, GatewayError
from ..llm.factory import Clients
from ..pipelines.parsing import PayloadError
from ..pipelines.staged import StagedPipeline
from ..types import CaseResult
from .cases import EvalCase
from .metrics import evaluate, evaluate_retrieval

INDEX_FIELDS = ("built_at", "embedding_model", "chunk_count", "chunk_ids_sha256", "exclusions")


def _run_dir(settings: Settings, run_id: str, out_root: Path | None) -> Path:
    path = Path(out_root or settings.paths.artifacts_dir / "runs") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(out_dir: Path, rows: list[dict], *, kind: str, pipeline: StagedPipeline, clients: Clients,
           settings: Settings, run_id: str, index_manifest: dict | None, case_files: list[str] | None,
           warnings: list[str] | None, metrics: dict, ledger_mark: int, note: str) -> dict:
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "run_id": run_id, "kind": kind, "pipeline": pipeline.name, "mode": clients.mode,
        "retriever": settings.pipeline.retriever,
        # set only when --retriever (or another override) replaced the pipeline's own retriever
        "retriever_override_of": settings.retriever_override(),
        "stages": pipeline.describe(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings.public(), "case_files": case_files or [], "warnings": list(warnings or []),
        "index": {k: (index_manifest or {}).get(k) for k in INDEX_FIELDS},
        "metrics": metrics, "ledger_by_stage": clients.ledger.by_stage(),
        "budget_remaining_usd": clients.ledger.remaining_usd(), "note": note,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    clients.ledger.write_csv(out_dir / "cost_ledger.csv", since=ledger_mark)
    return summary


def run_eval(pipeline: StagedPipeline, cases: list[EvalCase], clients: Clients, settings: Settings, *,
             run_id: str, index_manifest: dict | None = None, case_files: list[str] | None = None,
             out_root: Path | None = None, warnings: list[str] | None = None) -> dict:
    """Run the whole pipeline on each case."""
    out_dir, mark = _run_dir(settings, run_id, out_root), clients.ledger.mark()
    pairs, stopped = [], False
    for case in cases:
        if stopped:  # once the budget is gone, every remaining case is recorded as such
            result = CaseResult(case.case_id, pipeline.name, clients.mode, "budget_stopped",
                                error="budget exhausted earlier in this run")
        else:
            result = pipeline.run(case.input)
            stopped = result.outcome == "budget_stopped"
        pairs.append((case, {**result.to_dict(), "category": case.category, "split": case.split}))
    note = "" if clients.mode == "live" else (
        "replay/offline numbers are not real spend; compare 'cost.replayed_usd' with 'cost.total_usd'")
    return _write(out_dir, [row for _, row in pairs], kind="full", pipeline=pipeline, clients=clients,
                  settings=settings, run_id=run_id, index_manifest=index_manifest,
                  case_files=case_files, warnings=warnings, ledger_mark=mark, note=note,
                  metrics=evaluate(pairs, settings.eval.recall_ks))


def run_retrieval_eval(pipeline: StagedPipeline, cases: list[EvalCase], clients: Clients,
                       settings: Settings, *, run_id: str, index_manifest: dict | None = None,
                       case_files: list[str] | None = None, out_root: Path | None = None,
                       warnings: list[str] | None = None) -> dict:
    """Measure only the retrieval half (query, retrieve, rerank, packet): no answers, so almost free."""
    out_dir, ledger = _run_dir(settings, run_id, out_root), clients.ledger
    run_mark, pairs, stopped = ledger.mark(), [], False
    for case in cases:
        row = {"case_id": case.case_id, "pipeline": pipeline.name, "mode": clients.mode,
               "category": case.category, "split": case.split, "outcome": "retrieval_only",
               "retrieved": [], "packet": [], "citations": [], "violations": [], "error": None}
        mark, started = ledger.mark(), time.perf_counter()
        if stopped:
            row.update(outcome="budget_stopped", error="budget exhausted earlier in this run")
        else:
            try:
                r = pipeline.retrieve(case.input)
                row["retrieved"] = [{"chunk_id": h.chunk.chunk_id, "source_id": h.chunk.source_id,
                                     "parent_source_id": h.chunk.parent_source_id, "rank": h.rank,
                                     "score": round(h.score, 6), "found_by": h.source} for h in r.hits]
                row["packet"] = [{"chunk_id": c.chunk_id, "source_id": c.source_id,
                                  "parent_source_id": c.parent_source_id} for c in r.packet.chunks]
                row["n_queries"] = len(r.queries)
            except (BudgetExhausted, CallLimitReached) as exc:
                stopped = True
                row.update(outcome="budget_stopped", error=redact(exc))
            except (GatewayError, PayloadError) as exc:
                row.update(outcome="error", error=redact(exc))
        row["usage"] = {**ledger.summarize(mark), "wall_seconds": round(time.perf_counter() - started, 4)}
        pairs.append((case, row))
    return _write(out_dir, [row for _, row in pairs], kind="retrieval", pipeline=pipeline,
                  clients=clients, settings=settings, run_id=run_id, index_manifest=index_manifest,
                  case_files=case_files, warnings=warnings, ledger_mark=run_mark,
                  note="retrieval only: no answers were generated, so decision metrics are absent",
                  metrics=evaluate_retrieval(pairs, settings.eval.recall_ks))
