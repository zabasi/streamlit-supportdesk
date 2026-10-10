"""Command line interface. Every command reads one resolved configuration (see ``config.py``)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .casework import CaseWorkflow, Tracker, TrackerError
from .config import ConfigError, Settings
from .data.chunking import Exclusions
from .data.corpus import DataError
from .eval.cases import exclusions_from_cases, load_cases
from .eval.gold import check_gold, describe_cases
from .eval.leakage import check_leakage
from .eval.metrics import format_metrics
from .eval.runner import run_eval, run_retrieval_eval
from .eval.splits import assign_splits, check_no_family_overlap
from .indexing import run_indexing
from .llm.factory import build_clients
from .registry import PIPELINES, make_pipeline, pipeline_description, require
from .retrieval.index import IndexUnusable, VectorIndex
from .snapshot import take_snapshot
from .types import CaseInput

#: Which setting each command-line option sets.
SHORTCUTS = {"mode": "run.mode", "data_dir": "paths.data_dir", "index_dir": "paths.index_dir",
             "artifacts_dir": "paths.artifacts_dir", "pipeline": "pipeline.name",
             "retriever": "pipeline.retriever", "port": "web.port",
             "test_fraction": "eval.test_fraction", "seed": "eval.split_seed"}


def _settings(args, pipeline_name: str | None = None) -> Settings:
    """Resolve the configuration; ``pipeline_name`` re-resolves it for that pipeline's profile."""
    overrides = [f"{key}={getattr(args, attr)}" for attr, key in SHORTCUTS.items()
                 if getattr(args, attr, None) is not None]
    if getattr(args, "cases", None):
        overrides.append(f"eval.case_files={json.dumps(args.cases)}")
    overrides += list(getattr(args, "set", None) or [])
    if pipeline_name is not None:
        overrides.append(f"pipeline.name={pipeline_name}")
    return Settings.load(config_files=getattr(args, "config", None) or (), overrides=overrides)


def _abs(path) -> str:
    return str(Path(path).resolve())


def _header(s: Settings) -> str:
    """What is about to run: printed first by every command that runs a pipeline."""
    own = s.retriever_override()
    retriever = (s.pipeline.retriever if own is None
                 else f"{s.pipeline.retriever} (OVERRIDE: {s.pipeline.name}'s own is {own})")
    return (f"pipeline: {s.pipeline.name} | retriever: {retriever} | top_k: "
            f"{s.retrieval.top_k} | profile: {s.profile or '(none)'} | mode: {s.mode}")


def _load_cases(s: Settings, *, required: bool = True):
    files = [Path(f) for f in s.eval.case_files]
    missing = [str(f) for f in files if not f.exists()]  # a file, or a folder of evaluation files
    if missing and required:
        raise DataError(f"evaluation case file not found: {', '.join(missing)} "
                        "(setting eval.case_files, or pass --cases FILE)")
    return [] if missing else load_cases(*files)


def _load(s: Settings):
    require(s.pipeline.name, s.pipeline.retriever)
    clients = build_clients(s)
    index = VectorIndex.load(s.paths.index_dir, expect_model=clients.embedding_model,
                             expect_chunking=s.chunking)
    return clients, index, make_pipeline(s.pipeline.name, s.pipeline.retriever, index, clients, s)


def _leakage(s: Settings, chunks, cases):
    return check_leakage(chunks, cases, overlap_threshold=s.eval.leakage_overlap_threshold,
                         shingle_tokens=s.eval.leakage_shingle_tokens)


def cmd_status(args) -> int:
    s = _settings(args)
    pub = s.public()
    print(json.dumps({
        "mode": s.mode, "api_key_present": pub["api_key_present"], "gateway_host": pub["gateway_host"],
        "config_files": pub["config_files"], "config_sha256": pub["config_sha256"],
        "paths": {k: _abs(v) for k, v in vars(s.paths).items()},
        "data_dir_exists": s.paths.data_dir.is_dir(),
        "index_exists": (s.paths.index_dir / "manifest.json").exists(),
    }, indent=2))
    return 0


def cmd_config(args) -> int:
    s = _settings(args)
    if args.json:
        print(json.dumps({"config": s.to_dict(), "sources": s.sources,
                          "config_files": list(s.config_files), "sha256": s.fingerprint()}, indent=2))
        return 0
    print(_header(s))
    print(f"config files: {', '.join(s.config_files) or '(none: defaults and environment only)'}")
    print(f"fingerprint:  {s.fingerprint()}\n")
    section = None
    for sec, key, value, source, help_text in s.rows():
        if sec != section:
            section = sec
            print(f"[{sec}]")
        shown = _abs(value) if isinstance(value, Path) else value
        print(f"  {key:<27}= {shown!s:<28} ({source})")
        if args.describe:
            print(f"  {'':<27}  {help_text}")
    print(f"\nderived: retrieval pool = {s.pool_size}, chunks allowed in the prompt = {s.prompt_chunks}")
    return 0


def cmd_pipelines(args) -> int:
    """What can be selected, and how each is configured."""
    selected = _settings(args).pipeline.name
    rows = []
    for name in PIPELINES:
        s = _settings(args, name)
        rows.append((name, s.pipeline.retriever, str(s.retrieval.top_k), s.profile or "(none)",
                     pipeline_description(name)))
    head = ("PIPELINE", "RETRIEVER", "TOP_K", "PROFILE", "DESCRIPTION")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    for row in [head, *rows]:
        print("  ".join(row[i].ljust(widths[i]) for i in range(4)) + "  " + row[4])
    print(f"\nselected: {selected}   (change with --pipeline NAME, or pipeline.name in a config file)")
    return 0


def cmd_index(args) -> int:
    """Run the indexing graph: load -> clean -> chunk -> audit (leakage) -> embed -> store."""
    s = _settings(args)
    cases = _load_cases(s, required=False)
    if not cases:
        print(f"WARNING: no evaluation cases found ({', '.join(map(str, s.eval.case_files))}); "
              "nothing is excluded from the index", file=sys.stderr)
    exclusions = exclusions_from_cases(cases) if cases else Exclusions()
    clients = build_clients(s)
    state = run_indexing(s, clients, cases=cases, exclusions=exclusions, save_to=s.paths.index_dir)
    if state["sensitive"]:
        for source_id, kinds in state["sensitive"][:20]:
            print(f"SENSITIVE: {source_id} contains {', '.join(kinds)}", file=sys.stderr)
        print(f"stopped: {len(state['sensitive'])} source(s) contain personal data or secrets; fix the "
              "corpus (its text is never edited here). Nothing was written or spent.", file=sys.stderr)
        return 2
    report = state["leakage"]
    for line in report.warnings:
        print(f"WARNING leakage: {line}", file=sys.stderr)
    if not report.ok:
        print("\n".join(f"LEAK: {v}" for v in report.violations), file=sys.stderr)
        print("stopped before embedding: nothing was written or spent", file=sys.stderr)
        return 2
    empty = state["sources_without_chunks"]
    if empty:
        print(f"WARNING: {len(empty)} source(s) produced no chunks and are not searchable "
              f"(first: {', '.join(empty[:5])})", file=sys.stderr)
    index = state["index"]
    print(json.dumps({k: index.manifest[k] for k in
                      ("chunk_count", "chunks_by_type", "embedding_model", "chunking", "exclusions")},
                     indent=2))
    print(f"index written to {_abs(s.paths.index_dir)} (qdrant/ vector store, chunks.jsonl, manifest.json)")
    print(f"spent this run: ${clients.ledger.total_cost_usd:.6f}")
    return 0


def cmd_ask(args) -> int:
    s = _settings(args)
    print(_header(s), file=sys.stderr)
    _, _, pipeline = _load(s)
    row = json.loads(Path(args.json).read_text()) if args.json else {}
    row.setdefault("case_id", "adhoc")
    row.setdefault("title", args.title or "")
    row.setdefault("body", args.body or "")
    row.setdefault("created_at", args.created_at)
    result = pipeline.run(CaseInput.from_dict(row))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.outcome not in config.FAILURE_OUTCOMES else 1


def _run_eval(args, runner, kind: str) -> int:
    if args.split == "test" and not args.final:
        print("Refusing to run the test split without --final. Tune on dev; run test once.",
              file=sys.stderr)
        return 2
    s = _settings(args)
    info = sys.stderr if args.json else sys.stdout  # keep stdout pure JSON with --json
    print(_header(s), file=info)
    clients, index, pipeline = _load(s)
    all_cases = _load_cases(s)
    leak = _leakage(s, index.chunks, all_cases)
    if not leak.ok:
        print("\n".join(f"LEAK: {v}" for v in leak.violations), file=sys.stderr)
        print("The index contains evaluation issues; rebuild it with `supportdesk index`.",
              file=sys.stderr)
        return 2
    cases = [c for c in all_cases if c.split == args.split]
    if args.limit:
        cases = cases[: args.limit]
    problems = check_gold(cases, index.chunks, historical=s.retrieval.historical_filter)
    for line in problems[:10]:
        print(f"WARNING gold label: {line}", file=sys.stderr)
    if len(problems) > 10:
        print(f"WARNING: {len(problems) - 10} more gold-label problems (run `supportdesk check-cases`)",
              file=sys.stderr)
    run_id = args.run_id or f"{pipeline.name}-{kind}{args.split}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    summary = runner(pipeline, cases, clients, s, run_id=run_id, index_manifest=index.manifest,
                     case_files=[str(f) for f in s.eval.case_files], warnings=problems)
    print(json.dumps(summary["metrics"], indent=2) if args.json else format_metrics(summary["metrics"]))
    print(f"\nartifacts: {_abs(s.paths.artifacts_dir / 'runs' / run_id)}", file=info)
    return 0


def cmd_eval(args) -> int:
    return _run_eval(args, run_eval, "")


def cmd_eval_retrieval(args) -> int:
    return _run_eval(args, run_retrieval_eval, "retrieval-")


def cmd_check_cases(args) -> int:
    """Validate the evaluation set against the index: counts, gold labels, splits, leakage."""
    s = _settings(args)
    cases = _load_cases(s)
    index = VectorIndex.load(s.paths.index_dir)
    info = describe_cases(cases)
    print(f"{info['total']} cases in {info['families']} families")
    for split, b in sorted(info["by_split"].items()):
        cats = ", ".join(f"{k} {v}" for k, v in sorted(b["categories"].items()))
        print(f"  {split:<5} {b['n']:>3} cases ({b['with_gold']} with gold evidence)   {cats}")
    leak = _leakage(s, index.chunks, cases)
    problems = (check_gold(cases, index.chunks, historical=s.retrieval.historical_filter)
                + check_no_family_overlap(cases) + [f"leak: {v}" for v in leak.violations])
    for line in problems:
        print(f"FAIL: {line}")
    for line in leak.warnings:
        print(f"WARN: {line}")
    no_gold = [c.case_id for c in cases if not c.gold_evidence]
    if no_gold:
        print(f"note: {len(no_gold)} case(s) have no gold evidence and are excluded from recall: "
              f"{', '.join(no_gold[:8])}{' ...' if len(no_gold) > 8 else ''}")
    print("ok: every gold label is reachable and the index is free of evaluation issues"
          if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 2


def _snapshot_one(args, s: Settings) -> str:
    print(_header(s))
    _, _, pipeline = _load(s)
    gold = None
    if args.title:
        case = CaseInput(case_id="snapshot", title=args.title, body=args.body or "",
                         created_at=args.created_at)
    else:
        cases = _load_cases(s, required=False)
        picked = [c for c in cases if (c.case_id == args.case_id if args.case_id else c.split == "dev")]
        if not picked:
            raise DataError("no case to run: pass --title/--body, or --case-id, or add dev cases "
                            "to eval.case_files")
        case, gold = picked[0].input, picked[0].gold_evidence
    out_dir = Path(args.output_dir) if args.output_dir else s.paths.artifacts_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{s.pipeline.name}.md"
    path.write_text(take_snapshot(pipeline, s, case, gold=gold, runs_dir=s.paths.artifacts_dir / "runs"),
                    encoding="utf-8")
    return _abs(path)


def cmd_snapshot(args) -> int:
    """One Markdown file per pipeline: stages, settings, a stage-by-stage run, latest results."""
    base = _settings(args)
    names = list(PIPELINES) if args.all else [base.pipeline.name]
    failed = 0
    for name in names:
        try:
            print(f"wrote {_snapshot_one(args, _settings(args, name) if args.all else base)}\n")
        except (ConfigError, DataError, IndexUnusable) as exc:
            failed += 1
            print(f"{name}: error: {config.redact(exc)}", file=sys.stderr)
    return 1 if failed else 0


def _workflow(s: Settings) -> CaseWorkflow:
    _, _, pipeline = _load(s)
    return CaseWorkflow(pipeline, Tracker(s.paths.artifacts_dir / "tracker.sqlite"),
                        s.paths.artifacts_dir / "checkpoints.sqlite")


def cmd_case(args) -> int:
    """Open a case (--title and --body) or add the user's follow-up (--body), then run the case graph.
    Its proposal (answer, question or escalation) waits for `review`; nothing is posted before."""
    s = _settings(args)
    print(_header(s), file=sys.stderr)
    flow = _workflow(s)
    if args.title:
        flow.tracker.open_case(args.case_id, args.title, args.body or "")
    elif args.body:
        flow.tracker.add_user_message(args.case_id, args.body)
    else:
        raise TrackerError("open a case with --title and --body, or send a follow-up with --body")
    out = flow.run_turn(args.case_id)
    if out["status"] == "awaiting_review":
        p = out["proposal"]
        print(f"{p['outcome']} waiting for a maintainer (proposal {p['proposal_id']})\n"
              f"  proposed reply: {p['reply']}\n"
              f"decide with: supportdesk review {args.case_id} --approve | --edit TEXT | --reject")
    else:
        print(json.dumps({"outcome": out["outcome"], **(out["action"] or {})}, indent=2))
    return 0


def cmd_review(args) -> int:
    """Show the proposal waiting on a case, or decide it: --approve, --edit TEXT or --reject."""
    flow = _workflow(_settings(args))
    if not (args.approve or args.reject or args.edit is not None):
        print(json.dumps(flow.pending(args.case_id) or {"pending": None}, indent=2, ensure_ascii=False))
        return 0
    decision = "approve" if args.approve else "reject" if args.reject else "edit"
    action = flow.review(args.case_id, decision, text=args.edit, expected_proposal=args.proposal)
    print(json.dumps(action, indent=2))
    return 0


def cmd_tracker(args) -> int:
    """The local tracker: every case, one case in full (messages, notes, reviews, events), or reset."""
    s = _settings(args)
    tracker = Tracker(s.paths.artifacts_dir / "tracker.sqlite")
    if args.reset:
        tracker.reset()
        (s.paths.artifacts_dir / "checkpoints.sqlite").unlink(missing_ok=True)
        print("tracker and checkpoints reset")
        return 0
    print(json.dumps(tracker.show(args.case_id) if args.case_id else tracker.cases(), indent=2, ensure_ascii=False))
    return 0


def cmd_web(args) -> int:
    from .web.server import serve  # imported lazily: the CLI works without the web layer
    s = _settings(args)
    return serve(s, port=s.web.port, settings_for=lambda name: _settings(args, name))


def cmd_report(args) -> int:
    from .eval.report import write_report
    s = _settings(args)
    out = Path(args.output) if args.output else s.paths.artifacts_dir / "reports" / "report.html"
    write_report([s.paths.artifacts_dir / "runs" / r for r in args.runs], out)
    print(f"wrote {_abs(out)}")
    return 0


def cmd_split(args) -> int:
    s = _settings(args)
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    mapping = assign_splits([r["family_id"] for r in rows], test_fraction=s.eval.test_fraction,
                            seed=s.eval.split_seed)
    for row in rows:
        row["split"] = mapping[row["family_id"]]
    out = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    Path(args.output).write_text(out) if args.output else sys.stdout.write(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="supportdesk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, fn, help_text, *, pipeline=False, retriever=False, cases=False):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(fn=fn)
        p.add_argument("--config", action="append", metavar="FILE",
                       help="TOML config file layered over the defaults (repeatable)")
        p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                       help="override any setting, for example --set retrieval.top_k=8")
        p.add_argument("--mode", choices=config.MODES, help="run.mode: offline | replay | live")
        p.add_argument("--data-dir", help="paths.data_dir")
        p.add_argument("--index-dir", help="paths.index_dir")
        p.add_argument("--artifacts-dir", help="paths.artifacts_dir")
        if pipeline:
            p.add_argument("--pipeline", help="pipeline.name: which pipeline to use, and whose settings "
                                              "profile to load (see `supportdesk pipelines`)")
        if retriever:
            p.add_argument("--retriever", help="pipeline.retriever: dense | bm25 | hybrid")
        if cases:
            p.add_argument("--cases", nargs="+", metavar="FILE",
                           help="eval.case_files: evaluation case files (dev + test)")
        return p

    command("status", cmd_status, "Resolved paths, and whether the data and index exist", pipeline=True)
    p = command("config", cmd_config, "Show every setting and where its value came from", pipeline=True)
    p.add_argument("--describe", action="store_true", help="also describe each setting")
    p.add_argument("--json", action="store_true")
    command("pipelines", cmd_pipelines, "List the pipelines you can select and how each is configured",
            pipeline=True)
    command("index", cmd_index, "Build the vector index, leaving evaluation issues out",
            pipeline=True, cases=True)
    p = command("ask", cmd_ask, "Ask one question", pipeline=True, retriever=True)
    p.add_argument("--title")
    p.add_argument("--body")
    p.add_argument("--created-at")
    p.add_argument("--json", help="a case as JSON: title, body, history, ...")
    for name, fn, text in (("eval", cmd_eval, "Run a pipeline on the evaluation cases"),
                           ("eval-retrieval", cmd_eval_retrieval,
                            "Measure retrieval only (recall@k, MRR); no answers, almost free")):
        p = command(name, fn, text, pipeline=True, retriever=True, cases=True)
        p.add_argument("--split", choices=("dev", "test"), default="dev")
        p.add_argument("--final", action="store_true", help="required to run the test split")
        p.add_argument("--limit", type=int, help="only the first N cases")
        p.add_argument("--run-id")
        p.add_argument("--json", action="store_true", help="print the full metrics as JSON")
    p = command("snapshot", cmd_snapshot, "Write a Markdown snapshot of a pipeline: its stages, settings, "
                "a stage-by-stage run of one case, and its latest results",
                pipeline=True, retriever=True, cases=True)
    p.add_argument("--all", action="store_true", help="one snapshot per registered pipeline")
    p.add_argument("--title", help="run this report instead of the first dev case")
    p.add_argument("--body")
    p.add_argument("--created-at")
    p.add_argument("--case-id", help="run this evaluation case")
    p.add_argument("--output-dir", help="default: <artifacts_dir>/snapshots")
    p = command("case", cmd_case, "Open a support case or add the user's follow-up; the assistant's "
                "proposal waits for review", pipeline=True)
    p.add_argument("case_id")
    p.add_argument("--title", help="opens a new case")
    p.add_argument("--body", help="the report, or the user's follow-up message")
    p = command("review", cmd_review, "Show or decide the proposal waiting on a case", pipeline=True)
    p.add_argument("case_id")
    decide = p.add_mutually_exclusive_group()
    decide.add_argument("--approve", action="store_true")
    decide.add_argument("--reject", action="store_true")
    decide.add_argument("--edit", metavar="TEXT", help="post this text instead of the proposed reply")
    p.add_argument("--proposal", help="only decide if this proposal id is the one waiting")
    p = command("tracker", cmd_tracker, "Show the local tracker (all cases, or one case), or --reset it")
    p.add_argument("case_id", nargs="?")
    p.add_argument("--reset", action="store_true")
    command("check-cases", cmd_check_cases, "Validate the evaluation set against the index",
            pipeline=True, cases=True)
    p = command("web", cmd_web, "Local web UI (loopback only); runs each pipeline with its own retriever",
                pipeline=True)
    p.add_argument("--port", type=int, help="web.port")
    p = command("report", cmd_report, "HTML comparison of finished runs")
    p.add_argument("runs", nargs="+", help="run ids under <artifacts_dir>/runs/")
    p.add_argument("--output")
    p = command("split", cmd_split, "Assign dev/test splits by family to a cases file")
    p.add_argument("input")
    p.add_argument("--output")
    p.add_argument("--test-fraction", type=float, help="eval.test_fraction")
    p.add_argument("--seed", type=int, help="eval.split_seed")
    return parser


def load_dotenv_file() -> None:
    """Load ./.env (API key, base URL, SUPPORTDESK__ overrides) into the environment. Variables that
    are already set win. SUPPORTDESK_NO_DOTENV=1 turns this off (the test suite does)."""
    if os.environ.get("SUPPORTDESK_NO_DOTENV"):
        return
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:  # output piped into a command such as `head`, which closed early
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (ConfigError, DataError, IndexUnusable, TrackerError, ValueError) as exc:
        print(f"error: {config.redact(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
