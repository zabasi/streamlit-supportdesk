"""Hierarchical-class RAG experiment: 2 chunkers x 2 classifiers, vs advanced / advanced_hier.

The four new methods share one retrieve path (class-routed dense search, documentation places
kept). advanced and advanced_hier run as they are on the existing index. The combo attaches only
the winning classifier to advanced on that same index, so chunking is not confounded.

    python scripts/hier_class_rag.py --mode live
    python scripts/hier_class_rag.py --mode live --methods rule_kmeans,advanced,advanced_hier
    python scripts/hier_class_rag.py --mode offline   # sample-sized smoke; not a real score

Stay on --split dev. Gold is the annotated selected_spans dump. Do not commit .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from dotenv import load_dotenv

from supportdesk.config import Settings
from supportdesk.data.chunking import build_chunks, check_chunks
from supportdesk.data.corpus import load_corpus
from supportdesk.data.langchain_chunkers import LangChainLLMChunker, LangChainRuleChunker
from supportdesk.eval.annotated import DEFAULT_ANNOTATED, load_annotated_cases, write_eval_jsonl
from supportdesk.eval.cases import exclusions_from_cases, load_cases
from supportdesk.eval.metrics import evaluate_retrieval, format_metrics
from supportdesk.eval.runner import run_retrieval_eval
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.advanced import build_advanced
from supportdesk.pipelines.hierarchical import build_advanced_hier
from supportdesk.pipelines.staged import StagedPipeline
from supportdesk.pipelines.stages import default_stages
from supportdesk.registry import make_retriever
from supportdesk.retrieval.class_routing import (
    ClassRoutedRetriever,
    classify_chunks_llm,
    heuristic_label,
    kmeans_labels,
)
from supportdesk.retrieval.clusters import index_vectors
from supportdesk.retrieval.index import VectorIndex, index_from_vectors

NEW_METHODS = ("rule_kmeans", "rule_llm", "llm_kmeans", "llm_llm")
BASELINES = ("advanced", "advanced_hier")
FULL_COMBOS = ("advanced_llm_llm",)
ALL_METHODS = NEW_METHODS + BASELINES + FULL_COMBOS
DISPLAY_ORDER = NEW_METHODS + BASELINES + ("advanced+llm", "advanced+kmeans") + FULL_COMBOS
OUT = ROOT / "artifacts" / "hier_class_rag"


def _load_settings(mode: str, pipeline: str) -> Settings:
    return Settings.load(overrides={
        "run.mode": mode,
        "pipeline.name": pipeline,
        "pipeline.scope_gate": "off",
    })


def _cases(gold_path: Path, eval_dir: Path) -> list:
    official = load_cases(eval_dir) if eval_dir.exists() else []
    cases = load_annotated_cases(gold_path, official=official)
    if not cases:
        raise SystemExit(f"no annotated cases in {gold_path}")
    return cases


def _query_pipeline(name: str, retriever, clients, settings: Settings) -> StagedPipeline:
    return StagedPipeline(name=name, clients=clients, settings=settings,
                          **default_stages(retriever, clients, settings))


def _chunker(kind: str, clients, settings: Settings):
    prompt = Path(ROOT / "configs" / "prompts" / "llm_chunk_system.txt").read_text(encoding="utf-8")
    if kind == "rule":
        return LangChainRuleChunker()
    return LangChainLLMChunker(clients.chat, prompt=prompt, embedder=clients.embedding)


def _build_or_load_index(kind: str, clients, settings: Settings, exclusions, records, *,
                         rebuild: bool) -> VectorIndex:
    dest = OUT / "indexes" / kind
    marker = dest / "chunker.json"
    lock = dest / "qdrant" / ".lock"
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass
    if dest.joinpath("manifest.json").exists() and not rebuild:
        print(f"reusing index {dest}")
        return VectorIndex.load(dest)
    print(f"chunking + embedding ({kind})...")
    chunker = _chunker(kind, clients, settings)
    chunks = build_chunks(records, exclusions, chunker=chunker)
    check_chunks(records, chunks)
    vectors, batch = [], settings.retrieval.embed_batch_size
    for i in range(0, len(chunks), batch):
        vecs, _ = clients.embedding.embed([c.text for c in chunks[i:i + batch]], stage="index")
        vectors.extend(vecs)
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}")
    index = index_from_vectors(
        chunks, vectors, embedding_model=clients.embedding_model, chunking=settings.chunking,
        exclusions=exclusions, mode=clients.mode, notes={"hier_class_chunker": kind})
    dest.mkdir(parents=True, exist_ok=True)
    index.save(dest)
    marker.write_text(json.dumps({"chunker": kind, "chunks": len(chunks)}, indent=2), encoding="utf-8")
    print(f"index written to {dest} ({len(chunks)} chunks)")
    return index  # keep the in-memory store; reloading the same Qdrant folder can lock on Windows


def _labels_for(method: str, index: VectorIndex, clients, prompt: str, *, rebuild: bool) -> list[str]:
    path = OUT / "labels" / f"{method}.jsonl"
    if path.exists() and not rebuild:
        saved = [json.loads(line)["label"] for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if len(saved) == len(index.chunks):
            print(f"reusing labels {path}")
            return saved
    clf = method.split("_", 1)[1]
    print(f"labelling chunks ({clf})...")
    if clf == "kmeans":
        vectors = index_vectors(index)
        labels, _ = kmeans_labels(vectors, [c.source_type for c in index.chunks], k=8,
                                  method="doc_anchored")
    elif clients.mode == "offline":
        labels = [heuristic_label(f"{c.title}\n{c.text}") for c in index.chunks]
    else:
        labels = classify_chunks_llm(index.chunks, clients.chat, prompt=prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "label": lab}) + "\n"
                            for c, lab in zip(index.chunks, labels)), encoding="utf-8")
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    print(f"  classes: {counts}")
    return labels


def _new_retriever(method: str, index: VectorIndex, clients, settings: Settings,
                   labels: list[str], prompt: str) -> ClassRoutedRetriever:
    clf = method.split("_", 1)[1]
    return ClassRoutedRetriever(
        index, clients.embedding, labels=labels, doc_slots=max(settings.retrieval.doc_slots, 3),
        top=2, mode="filter", chat=clients.chat, prompt=prompt,
        route="llm" if clf == "llm" else "centroid",
    )


def _pick(metrics: dict) -> float:
    return float((metrics.get("recall_at_k") or {}).get("6") or 0.0)


def _factor_view(rows: dict[str, dict]) -> dict:
    def avg(names):
        vals = [_pick(rows[n]["metrics"]) for n in names if n in rows]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "rule_chunker": avg(["rule_kmeans", "rule_llm"]),
        "llm_chunker": avg(["llm_kmeans", "llm_llm"]),
        "kmeans_classifier": avg(["rule_kmeans", "llm_kmeans"]),
        "llm_classifier": avg(["rule_llm", "llm_llm"]),
    }


def _winner(rows: dict[str, dict]) -> str | None:
    scored = [(name, _pick(row["metrics"])) for name, row in rows.items() if name in NEW_METHODS]
    if not scored:
        return None
    scored.sort(key=lambda kv: (-kv[1], 0 if kv[0].endswith("kmeans") else 1, kv[0]))
    return scored[0][0]


def _combo_retriever(winner: str, index: VectorIndex, clients, settings: Settings, prompt: str,
                     *, rebuild: bool):
    clf = winner.split("_", 1)[1]
    labels = _labels_for(f"existing_{clf}", index, clients, prompt, rebuild=rebuild)
    return ClassRoutedRetriever(
        index, clients.embedding, labels=labels, doc_slots=settings.retrieval.doc_slots or 10,
        top=2, mode="filter", chat=clients.chat, prompt=prompt,
        route="llm" if clf == "llm" else "centroid",
    ), clf


def _write_comparison(rows: dict[str, dict], winner: str | None, factors: dict, path: Path) -> None:
    ks = ("1", "3", "6")
    lines = [
        "# Hierarchical-class RAG comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "Gold = annotated selected_spans. Split = dev.",
        "",
        "| method | recall@1 | recall@3 | recall@6 | hit@6 | MRR | passage_recall@6 | chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = [n for n in DISPLAY_ORDER if n in rows] + [n for n in rows if n not in DISPLAY_ORDER]
    for name in ordered:
        row = rows[name]
        m = row["metrics"]
        rec, hit = m.get("recall_at_k") or {}, m.get("hit_at_k") or {}
        pr = (m.get("passage_recall_at_k") or {}).get("6")
        lines.append(
            f"| `{name}` | {rec.get('1')} | {rec.get('3')} | {rec.get('6')} | {hit.get('6')} | "
            f"{m.get('mrr')} | {pr} | {row.get('chunks', '')} |")
    lines += ["", "## One-factor view (mean recall@6 of the two methods that share the factor)", ""]
    for key, value in factors.items():
        lines.append(f"- **{key}**: {value}")
    if winner:
        lines += ["", f"Winner among the four new methods: `{winner}`.",
                  "`advanced+llm` = LLM classifier on the existing index. "
                  "`advanced_llm_llm` = advanced rewrite/rerank on the LLM-chunked index with LLM topics."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    if os.environ.get("SUPPORTDESK_CONFIG", "").endswith("sample.toml"):
        print("WARNING: SUPPORTDESK_CONFIG points at the sample corpus; unset it for the real eval.",
              file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("offline", "replay", "live"), default="offline")
    ap.add_argument("--gold", type=Path, default=DEFAULT_ANNOTATED)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="comma list from " + ",".join(ALL_METHODS))
    ap.add_argument("--combo", action=argparse.BooleanOptionalAction, default=True,
                    help="after the four methods, run advanced + the winning classifier (default on)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")

    settings = _load_settings(args.mode, "advanced")
    clients = build_clients(settings)
    cases = _cases(args.gold, args.eval_dir)
    if args.limit:
        cases = cases[: args.limit]
    write_eval_jsonl(cases, OUT / "annotated_dev_cases.jsonl")
    print(f"{len(cases)} annotated dev cases -> {OUT / 'annotated_dev_cases.jsonl'}")
    print(f"gold sources / case: " +
          ", ".join(f"{c.case_id}={len(c.gold_evidence)}" for c in cases))

    official = load_cases(args.eval_dir) if args.eval_dir.exists() else cases
    exclusions = exclusions_from_cases(official)
    records = load_corpus(settings.paths.data_dir).records
    prompt = Path(ROOT / "configs" / "prompts" / "topic_classify_system.txt").read_text(encoding="utf-8")
    out_root = OUT / "runs"
    rows: dict[str, dict] = {}

    for method in wanted:
        started = time.perf_counter()
        print(f"\n=== {method} ===")
        if method in BASELINES:
            s = _load_settings(args.mode, method)
            c = build_clients(s)
            index = VectorIndex.load(s.paths.index_dir, expect_model=c.embedding_model,
                                     expect_chunking=s.chunking)
            retriever = make_retriever(s.pipeline.retriever, index, c, s)
            pipeline = (build_advanced_hier if method == "advanced_hier" else build_advanced)(
                retriever, c, s)
            summary = run_retrieval_eval(
                pipeline, cases, c, s, run_id=method, out_root=out_root,
                case_files=[str(args.gold)], warnings=["annotated selected_spans gold"])
            rows[method] = {"metrics": summary["metrics"], "chunks": len(index.chunks),
                            "run": str(out_root / method)}
            clients = c
        elif method == "advanced_llm_llm":
            s = _load_settings(args.mode, "advanced")
            c = build_clients(s)
            index = _build_or_load_index("llm", c, s, exclusions, records, rebuild=args.rebuild)
            labels = _labels_for("llm_llm", index, c, prompt, rebuild=args.rebuild)
            retriever = ClassRoutedRetriever(
                index, c.embedding, labels=labels, doc_slots=s.retrieval.doc_slots or 10,
                top=2, mode="filter", chat=c.chat, prompt=prompt, route="llm")
            pipeline = build_advanced(retriever, c, s)
            pipeline.name = method
            summary = run_retrieval_eval(
                pipeline, cases, c, s, run_id=method, out_root=out_root,
                case_files=[str(args.gold)],
                warnings=["annotated gold; advanced stages on LLM chunks + LLM topics"])
            rows[method] = {"metrics": summary["metrics"], "chunks": len(index.chunks),
                            "run": str(out_root / method)}
            clients = c
        else:
            simple = _load_settings(args.mode, "baseline")
            kind = method.split("_", 1)[0]
            index = _build_or_load_index(kind, clients, settings, exclusions, records,
                                         rebuild=args.rebuild)
            labels = _labels_for(method, index, clients, prompt, rebuild=args.rebuild)
            retriever = _new_retriever(method, index, clients, simple, labels, prompt)
            pipeline = _query_pipeline(method, retriever, clients, simple)
            summary = run_retrieval_eval(
                pipeline, cases, clients, settings, run_id=method, out_root=out_root,
                case_files=[str(args.gold)], warnings=["annotated selected_spans gold"])
            rows[method] = {"metrics": summary["metrics"], "chunks": len(index.chunks),
                            "run": str(out_root / method)}
        print(format_metrics(rows[method]["metrics"]))
        print(f"wall {time.perf_counter() - started:.1f}s")

    winner = _winner(rows)
    if args.combo and winner:
        print(f"\n=== combo advanced+{winner.split('_', 1)[1]} (winner {winner}) ===")
        s = _load_settings(args.mode, "advanced")
        c = build_clients(s)
        index = VectorIndex.load(s.paths.index_dir, expect_model=c.embedding_model,
                                 expect_chunking=s.chunking)
        retriever, clf = _combo_retriever(winner, index, c, s, prompt, rebuild=args.rebuild)
        name = f"advanced+{clf}"
        pipeline = build_advanced(retriever, c, s)
        pipeline.name = name
        summary = run_retrieval_eval(
            pipeline, cases, c, s, run_id=name, out_root=out_root,
            case_files=[str(args.gold)], warnings=["annotated gold; routing on existing index"])
        rows[name] = {"metrics": summary["metrics"], "chunks": len(index.chunks),
                      "run": str(out_root / name), "winner": winner}
        print(format_metrics(summary["metrics"]))

    prev_path = OUT / "results.json"
    previous_winner = None
    if prev_path.exists():
        previous_file = json.loads(prev_path.read_text(encoding="utf-8"))
        previous_winner = previous_file.get("winner")
        for name, body in (previous_file.get("methods") or {}).items():
            if name not in rows and body.get("metrics"):
                rows[name] = body
    winner = winner or previous_winner or _winner(rows)
    factors = _factor_view(rows)
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": args.mode, "n_cases": len(cases), "winner": winner,
              "factors": factors, "methods": {k: {"metrics": v["metrics"], "chunks": v.get("chunks"),
                                                  "run": v.get("run")} for k, v in rows.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_comparison(rows, winner, factors, OUT / "comparison.md")
    print(f"\nfactors (mean recall@6): {factors}")
    print(f"winner: {winner}")
    print(f"wrote {OUT / 'results.json'} and {OUT / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
