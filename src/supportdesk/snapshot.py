"""A Markdown snapshot of one pipeline: its stages, its settings, a stage-by-stage walkthrough of one
case, and its latest evaluation results."""
from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .eval.metrics import format_metrics
from .pipelines.staged import STAGE_NAMES, StagedPipeline
from .tokenizer import count_tokens
from .types import CaseInput

# "pipeline" holds the design's switches (retriever, scope gate, answer step, its model, package and fixes)
SETTING_SECTIONS = ("pipeline", "models", "prompts", "chunking", "retrieval", "context", "generation")


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fence(text: str, lang: str = "") -> str:
    """A code block that cannot be closed early by backticks inside ``text``."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text


def _describe(stage) -> str:
    doc = inspect.getdoc(stage if inspect.isfunction(stage) else type(stage)) or ""
    return doc.splitlines()[0] if doc else ""


def _implementation(stage) -> str:
    return getattr(stage, "__name__", None) or type(stage).__name__


def _hits_table(hits, gold: set[str]) -> list[str]:
    rows = ["| rank | score | source | type | chunk | gold |", "|---|---|---|---|---|---|"]
    for h in hits:
        is_gold = h.chunk.source_id in gold or (h.chunk.parent_source_id or "") in gold
        rows.append(f"| {h.rank} | {h.score:.4f} | {_cell(h.chunk.source_id)} | {h.chunk.source_type} | "
                    f"`{h.chunk.chunk_id}` | {'✓' if is_gold else ''} |")
    return rows


def latest_runs(runs_dir: Path, pipeline: str) -> dict[str, tuple[str, dict]]:
    """The newest run of each kind (full, retrieval) for this pipeline: kind -> (run_id, summary)."""
    found: dict[str, tuple[str, dict]] = {}
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("pipeline") != pipeline:
            continue
        kind = summary.get("kind", "full")
        if kind not in found or summary["created_at"] >= found[kind][1]["created_at"]:
            found[kind] = (path.parent.name, summary)
    return found


def _attempts(attempts: list[dict]) -> str:
    """The answer's attempts: a status for the single-call step; for the controlled step, which model and
    package wrote the assessment, the routing reason and any retries of a dropped connection."""
    parts = []
    for attempt in attempts or []:
        if "controlled_v2" in attempt:
            routing = attempt["controlled_v2"].get("routing") or {}
            retries = attempt.get("retries") or 0
            parts.append(f"assessment by `{attempt.get('model', '?')}` ({attempt.get('package', 'controlled_v2')}), "
                         f"routed by `{routing.get('reason', '?')}`"
                         + (f", {retries} {'retry' if retries == 1 else 'retries'}" if retries else ""))
        else:
            parts.append(str(attempt.get("status", "?")))
    return ", ".join(parts) or "n/a"


def take_snapshot(pipeline: StagedPipeline, settings: Settings, case: CaseInput, *,
                  gold: list[str] | None = None, runs_dir: Path | None = None) -> str:
    """Run ``pipeline`` on ``case`` once, recording every step, and describe it as Markdown."""
    steps: dict[str, object] = {}
    ledger = pipeline.clients.ledger
    mark = ledger.mark()
    result = pipeline.run(case, observe=lambda step, output: steps.__setitem__(step, output))
    calls = ledger.rows[mark:]
    gold_set = set(gold or [])
    out: list[str] = [f"# Pipeline snapshot: {pipeline.name}", ""]

    out += ["| | |", "|---|---|",
            f"| Taken | {datetime.now(timezone.utc).isoformat(timespec='seconds')} |",
            f"| Profile | {settings.profile or '(none)'} |",
            f"| Mode | {settings.mode}"
            + (" (fake models: shows the flow, not answer quality)" if settings.mode == "offline" else "")
            + " |",
            f"| Retriever | `{settings.pipeline.retriever}`"
            + (f" (**override**: {pipeline.name}'s own is `{settings.retriever_override()}`)"
               if settings.retriever_override() else "") + " |",
            f"| Models | chat `{settings.models.chat}`, embedding `{settings.models.embedding}`"
            + (f", answer `{settings.pipeline.answer_model}` (controlled package "
               f"{settings.pipeline.controlled_version})" if settings.pipeline.generator == "controlled_v2" else "")
            + " |",
            f"| Config fingerprint | `{settings.fingerprint()}` |", ""]

    out += ["## Stages", "", "| # | Stage | Implementation | What it does |", "|---|---|---|---|"]
    for number, name in enumerate(STAGE_NAMES, 1):
        stage = getattr(pipeline, name)
        out.append(f"| {number} | {name} | `{_implementation(stage)}` | {_cell(_describe(stage))} |")

    out += ["", "## Settings", "",
            "Values marked ★ come from this pipeline's own profile.", "",
            "| Setting | Value | Source |", "|---|---|---|"]
    for section, key, value, source, _ in settings.rows():
        if section in SETTING_SECTIONS:
            star = " ★" if settings.profile and settings.profile in source else ""
            shown = "(unset)" if value is None else value
            out.append(f"| `{section}.{key}` | {_cell(shown)} | {_cell(source)}{star} |")
    out.append(f"\nDerived: retrieval pool {settings.pool_size}, "
               f"chunks allowed in the prompt {settings.prompt_chunks}.")

    out += ["", "## Walkthrough of one case", "",
            f"**{_cell(case.title)}**" + (f" (filed {case.created_at})" if case.created_at else ""), "",
            "> " + _short(case.body, 600).replace("\n", "\n> ") if case.body else "> (no body)", ""]
    if gold_set:
        out += [f"Gold evidence for this case: {', '.join(f'`{g}`' for g in sorted(gold_set))}", ""]

    env = steps.get("environment", {})
    out += ["### Environment read from the report", "",
            ", ".join(f"{k} = `{v}`" for k, v in sorted(env.items())) or "nothing stated", ""]
    if "query" in steps:
        out += [f"### 1. query: `{_implementation(pipeline.query)}`", ""]
        out += [f"- {_short(q, 300)}" for q in steps["query"]] + [""]
    if "retriever" in steps:
        hits = steps["retriever"]
        out += [f"### 2. retriever: `{_implementation(pipeline.retriever)}`", "",
                f"{len(hits)} candidates (pool {settings.pool_size}):", ""] + _hits_table(hits, gold_set) + [""]
    if "reranker" in steps:
        before = [h.chunk.chunk_id for h in steps.get("retriever", [])][: settings.retrieval.top_k]
        after = steps["reranker"]
        out += [f"### 3. reranker: `{_implementation(pipeline.reranker)}`", ""]
        out += (["Order unchanged; kept the top " + str(len(after)) + "."]
                if [h.chunk.chunk_id for h in after] == before
                else ["New order:", ""] + _hits_table(after, gold_set)) + [""]
    if "packet" in steps:
        packet = steps["packet"]
        out += [f"### 4. packet: `{_implementation(pipeline.packet)}`", "",
                f"{len(packet.chunks)} chunks, {packet.token_count} tokens shown to the model:", "",
                "| label | chunk | tokens |", "|---|---|---|"]
        label_of = {cid: label for label, cid in packet.labels.items()}
        out += [f"| {label_of[c.chunk_id]} | `{c.chunk_id}` | {count_tokens(c.text)} |" for c in packet.chunks]
        if packet.dropped:
            out += ["", "Dropped: " + ", ".join(f"`{d['chunk_id']}` ({d['reason']})" for d in packet.dropped)]
        out.append("")
    if "generator" in steps:
        answer = steps["generator"]
        out += [f"### 5. generator: `{_implementation(pipeline.generator)}`", "",
                (f"Attempts: {_attempts(answer.attempts)}. "
                 f"Proposed outcome: **{answer.outcome}**. Raw model output:"), "",
                _fence(_pretty(answer.raw_output)[:4000], "json"), ""]
    if "validator" in steps:
        checked = steps["validator"]
        proposed = steps["generator"].outcome if "generator" in steps else checked.outcome
        change = (f"Outcome changed from **{proposed}** to **{checked.outcome}**."
                  if checked.outcome != proposed else f"Outcome kept: **{checked.outcome}**.")
        out += [f"### 6. validator: `{_implementation(pipeline.validator)}`", "", change,
                f"Violations: {', '.join(checked.violations) or 'none'}.", ""]
    if "summarizer" in steps:
        out += [f"### 7. summarizer: `{_implementation(pipeline.summarizer)}`", "",
                _fence(json.dumps(steps["summarizer"], indent=2, ensure_ascii=False), "json"), ""]

    out += ["### Result", "", f"Outcome: **{result.outcome}**"]
    if result.error:
        out.append(f"\nStopped with an error: `{_cell(result.error)}`")
    if result.user_reply:
        out += ["", "Reply to the reporter:", "", "> " + result.user_reply.replace("\n", "\n> ")]
    out += ["", f"Citations: {', '.join(f'`{c}`' for c in result.citations) or 'none'}", "",
            "Model calls during this run:", "",
            "| stage | type | mode | tokens in | tokens out | cost (USD) | status |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {r.get('stage')} | {r.get('request_type')} | {r.get('mode')} | {r.get('input_tokens', 0)} | "
            f"{r.get('output_tokens', 0)} | {r.get('cost_usd', 0.0):.6f} | {r.get('status')} |" for r in calls]
    out.append(f"\nTime: {result.usage.get('wall_seconds', 0):.3f}s")

    if runs_dir is not None:
        runs = latest_runs(runs_dir, pipeline.name)
        out += ["", "## Latest evaluation results", ""]
        if not runs:
            out.append(f"No runs yet. Try `supportdesk eval --pipeline {pipeline.name} --split dev`.")
        for kind in ("full", "retrieval"):
            if kind in runs:
                run_id, summary = runs[kind]
                out += [f"**{kind}** run `{run_id}` ({summary['created_at']}, mode {summary['mode']}):", "",
                        _fence(format_metrics(summary["metrics"])), ""]
    return "\n".join(out).rstrip() + "\n"
