"""Hierarchical-class RAG experiment: 2 chunkers x 2 classifiers, vs advanced / advanced_hier.

The four new methods share one retrieve path (class-routed dense search, documentation places
kept). advanced and advanced_hier run as they are on the existing index. The combo attaches only
the winning classifier to advanced on that same index, so chunking is not confounded.

``--ablation`` runs the 3x3 of advanced x {existing, rule, llmchunk} x {none, kmeans, llmfilter}
and reuses any finished ``runs/<name>/summary.json``. Old ids (``advanced_llm``,
``advanced_llm_llm``, ``advanced+llm``) are aliases.

Name parts: ``llmchunk`` = LLM splitter, ``llmfilter`` = LLM topic filter, ``kmeans`` =
k-means filter, ``exist`` / ``rule`` = which index. So ``advanced_llmchunk`` is LLM
**chunking only**; ``advanced_exist_llmfilter`` is LLM **filter only**.

    python scripts/hier_class_rag.py --mode live
    python scripts/hier_class_rag.py --mode live --ablation --no-combo
    python scripts/hier_class_rag.py --mode live --methods rule_kmeans,advanced,advanced_hier
    python scripts/hier_class_rag.py --mode offline   # sample-sized smoke; not a real score

Stay on --split dev. Gold is the annotated selected_spans dump. Do not commit .env.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

with contextlib.suppress(AttributeError, ValueError):  # line-buffered output where supported
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
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
from supportdesk.eval.metrics import format_metrics
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
from supportdesk.retrieval.retriever import DocSlotsRetriever

NEW_METHODS = ("rule_kmeans", "rule_llm", "llm_kmeans", "llm_llm")
BASELINES = ("advanced", "advanced_hier")
# Ids spell both factors: llmchunk = splitter, llmfilter = topic router.
# advanced × {existing, rule, llmchunk} × {none, kmeans, llmfilter}
ADVANCED_ABLATION = {
    "advanced": ("existing", "none"),
    "advanced_exist_kmeans": ("existing", "kmeans"),
    "advanced_exist_llmfilter": ("existing", "llm"),
    "advanced_rule": ("rule", "none"),
    "advanced_rule_kmeans": ("rule", "kmeans"),
    "advanced_rule_llmfilter": ("rule", "llm"),
    "advanced_llmchunk": ("llm", "none"),
    "advanced_llmchunk_kmeans": ("llm", "kmeans"),
    "advanced_llmchunk_llmfilter": ("llm", "llm"),
}
METHOD_ALIASES = {
    "advanced+kmeans": "advanced_exist_kmeans",
    "advanced+llm": "advanced_exist_llmfilter",
    "advanced_rule_llm": "advanced_rule_llmfilter",
    "advanced_llm": "advanced_llmchunk",
    "advanced_llm_kmeans": "advanced_llmchunk_kmeans",
    "advanced_llm_llm": "advanced_llmchunk_llmfilter",
    "advanced_llmchunk_llm": "advanced_llmchunk_llmfilter",
}
METHOD_LABELS = {
    "rule_kmeans": "rule chunker + k-means filter (simple retrieve)",
    "rule_llm": "rule chunker + LLM topic filter (simple retrieve)",
    "llm_kmeans": "LLM chunker + k-means filter (simple retrieve)",
    "llm_llm": "LLM chunker + LLM topic filter (simple retrieve)",
    "advanced": "existing chunks, no filter (current advanced)",
    "advanced_hier": "existing chunks + section-summary hierarchy",
    "advanced_exist_kmeans": "existing chunks + k-means filter",
    "advanced_exist_llmfilter": "existing chunks + LLM topic filter",
    "advanced_rule": "rule chunker, no filter",
    "advanced_rule_kmeans": "rule chunker + k-means filter",
    "advanced_rule_llmfilter": "rule chunker + LLM topic filter",
    "advanced_llmchunk": "LLM chunker, no filter",
    "advanced_llmchunk_kmeans": "LLM chunker + k-means filter",
    "advanced_llmchunk_llmfilter": "LLM chunker (gpt-4o-mini) + LLM topic filter (gpt-4o-mini)",
    "advanced_llmchunk_llmfilter_gpt5mini": (
        "LLM chunker (gpt-4o-mini) + LLM topic filter (gpt-5-mini)"),
}
# Same retrieve as advanced_llmchunk_llmfilter; only the topic model is stronger.
# gpt-5o-mini does not exist; gpt-5-mini is the cheap GPT-5 tier.
EXTRA_METHODS = {
    "advanced_llmchunk_llmfilter_gpt5mini": ("llm", "llm", "gpt-5-mini"),
}
FULL_COMBOS = tuple(n for n in ADVANCED_ABLATION if n not in BASELINES)
ALL_METHODS = NEW_METHODS + BASELINES + FULL_COMBOS + tuple(EXTRA_METHODS)
DISPLAY_ORDER = NEW_METHODS + BASELINES + FULL_COMBOS + tuple(EXTRA_METHODS)
OUT = ROOT / "artifacts" / "hier_class_rag"


def _canonical(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def _load_settings(mode: str, pipeline: str, *, chat: str | None = None,
                   max_output_tokens: int | None = None, extra: dict | None = None) -> Settings:
    overrides = {
        "run.mode": mode,
        "pipeline.name": pipeline,
        "pipeline.scope_gate": "off",
    }
    if chat:
        overrides["models.chat"] = chat
    if max_output_tokens:
        overrides["generation.max_output_tokens"] = str(max_output_tokens)
    if extra:
        overrides.update(extra)
    return Settings.load(overrides=overrides)


def _cases(gold_path: Path | None, eval_dir: Path) -> list:
    """The development cases. By default their gold is the repository's rubrics (data/eval, which hold
    the reviewed labels); --gold overlays an annotated dump instead."""
    official = load_cases(eval_dir) if eval_dir.exists() else []
    if gold_path is None:
        cases = [c for c in official if c.split == "dev"]
        if not cases:
            raise SystemExit(f"no development cases in {eval_dir}")
        return cases
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
    if clf == "kmeans" or clf.startswith("kmeans"):
        vectors = index_vectors(index)
        labels, _ = kmeans_labels(vectors, [c.source_type for c in index.chunks], k=8,
                                  method="doc_anchored")
    elif clients.mode == "offline":
        labels = [heuristic_label(f"{c.title}\n{c.text}") for c in index.chunks]
    else:
        batch = 8 if "gpt-5" in method else 16
        labels = classify_chunks_llm(index.chunks, clients.chat, prompt=prompt, batch=batch)
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
        route="llm" if clf.split("_", 1)[0] in {"llm", "llmfilter"} or clf.startswith("llm") else "centroid",
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


def _run_dir(name: str) -> Path:
    """Prefer the canonical folder; fall back to an old alias folder."""
    dest = OUT / "runs" / name
    if (dest / "summary.json").exists():
        return dest
    for old, new in METHOD_ALIASES.items():
        if new == name:
            alias = OUT / "runs" / old
            if (alias / "summary.json").exists():
                return alias
    return dest


def _load_finished_run(name: str) -> dict | None:
    summary_path = _run_dir(name) / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"metrics": summary["metrics"],
            "chunks": (summary.get("index") or {}).get("chunk_count"),
            "run": str(summary_path.parent)}


def _advanced_retriever(chunk: str, clf: str, index: VectorIndex, clients, settings: Settings,
                        prompt: str, *, rebuild: bool, classifier_clients=None,
                        classifier_model: str | None = None):
    if clf == "none":
        return DocSlotsRetriever(index, clients.embedding, doc_slots=settings.retrieval.doc_slots or 10)
    router = classifier_clients or clients
    suffix = f"_{classifier_model}" if classifier_model else ""
    label_key = f"{'existing' if chunk == 'existing' else chunk}_{clf}{suffix}"
    labels = _labels_for(label_key, index, router, prompt, rebuild=rebuild)
    return ClassRoutedRetriever(
        index, clients.embedding, labels=labels, doc_slots=settings.retrieval.doc_slots or 10,
        top=2, mode="filter", chat=router.chat, prompt=prompt,
        route="llm" if clf == "llm" else "centroid",
    )


def _run_advanced_ablation(name: str, chunk: str, clf: str, *, mode: str, cases, gold: Path,
                           exclusions, records, prompt: str, rebuild: bool,
                           classifier_model: str | None = None) -> dict:
    finished = None if rebuild else _load_finished_run(name)
    if finished:
        print(f"reusing finished run {finished['run']}")
        return finished
    s = _load_settings(mode, "advanced")
    c = build_clients(s)
    router = c
    if classifier_model and classifier_model != s.models.chat:
        print(f"  classifier model {classifier_model} (chunk/rewrite/rerank stay {s.models.chat})")
        router = build_clients(_load_settings(
            mode, "advanced", chat=classifier_model, max_output_tokens=2500))
    if chunk == "existing":
        index = VectorIndex.load(s.paths.index_dir, expect_model=c.embedding_model,
                                 expect_chunking=s.chunking)
    else:
        index = _build_or_load_index(chunk, c, s, exclusions, records, rebuild=rebuild)
    retriever = _advanced_retriever(
        chunk, clf, index, c, s, prompt, rebuild=rebuild,
        classifier_clients=router, classifier_model=classifier_model)
    pipeline = build_advanced(retriever, c, s)
    pipeline.name = name
    summary = run_retrieval_eval(
        pipeline, cases, c, s, run_id=name, out_root=OUT / "runs",
        case_files=[str(gold)],
        warnings=[
            f"annotated gold; advanced + chunk={chunk} + classifier={clf}"
            + (f" + classifier_model={classifier_model}" if classifier_model else "")])
    return {"metrics": summary["metrics"], "chunks": len(index.chunks),
            "run": str(OUT / "runs" / name)}


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
    lines = [
        "# Hierarchical-class RAG comparison",
        "",
        (f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
         "Gold = the development rubrics (data/eval) or --gold. Split = dev."),
        "",
        "## Name key",
        "",
        "- `llmchunk` = LLM **chunker** (how documentation is split). Not a filter.",
        "- `llmfilter` = LLM **topic filter** (which classes are searched). Not a chunker.",
        "- `kmeans` = k-means **filter** on whichever index the first part names.",
        "- `exist` / `rule` / `llmchunk` after `advanced_` = which index `advanced` searches.",
        "- So `advanced_llmchunk` is LLM chunking only. `advanced_exist_llmfilter` is LLM filter only.",
        ("- `advanced_llmchunk_llmfilter` is both. Old ids: `advanced_llm` → `advanced_llmchunk`, "
         "`advanced+llm` → `advanced_exist_llmfilter`, `advanced_llm_llm` → `advanced_llmchunk_llmfilter`."),
        "",
        "| method | meaning | recall@1 | recall@3 | recall@6 | hit@6 | MRR | passage_recall@6 | chunks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    remapped = {_canonical(n): rows[n] for n in rows}
    ordered = [n for n in DISPLAY_ORDER if n in remapped] + [n for n in remapped if n not in DISPLAY_ORDER]
    for name in ordered:
        row = remapped[name]
        m = row["metrics"]
        rec, hit = m.get("recall_at_k") or {}, m.get("hit_at_k") or {}
        pr = (m.get("passage_recall_at_k") or {}).get("6")
        meaning = METHOD_LABELS.get(name, "")
        lines.append(
            f"| `{name}` | {meaning} | {rec.get('1')} | {rec.get('3')} | {rec.get('6')} | "
            f"{hit.get('6')} | {m.get('mrr')} | {pr} | {row.get('chunks', '')} |")
    rows = remapped
    lines += ["", "## One-factor view (mean recall@6 of the two simple methods that share the factor)", ""]
    lines.append("These four numbers are the 2×2 only (no rewrite, no rerank). They are **not** the advanced ablation.")
    lines.append("")
    for key, value in factors.items():
        lines.append(f"- **{key}**: {value}")
    if winner:
        lines += ["", f"Winner among the four new methods: `{winner}`."]
    base = _pick(rows["advanced"]["metrics"]) if "advanced" in rows else None
    if base is not None and any(n in rows for n in ADVANCED_ABLATION):
        lines += [
            "",
            "## Advanced ablation (Δ recall@6 vs `advanced`)",
            "",
            "Every cell keeps `advanced` retrieve (clean + up to 4 queries + dense_doc_slots + RRF + rerank + keep 6).",
            "Only the index and the class filter change.",
            "",
            "| method | chunker | classifier | recall@6 | Δ vs advanced | hit@6 | MRR |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        by_chunk: dict[str, list[float]] = {}
        by_clf: dict[str, list[float]] = {}
        for name, (chunk, clf) in ADVANCED_ABLATION.items():
            if name not in rows:
                continue
            metrics = rows[name]["metrics"]
            score = _pick(metrics)
            delta = round(score - base, 4)
            sign = f"+{delta}" if delta > 0 else str(delta)
            hit = (metrics.get("hit_at_k") or {}).get("6")
            mrr = metrics.get("mrr")
            chunk_name = "llmchunk" if chunk == "llm" else chunk
            clf_name = "llmfilter" if clf == "llm" else clf
            lines.append(
                f"| `{name}` | {chunk_name} | {clf_name} | {score} | {sign} | {hit} | {mrr} |")
            by_chunk.setdefault(chunk, []).append(score)
            by_clf.setdefault(clf, []).append(score)
        if by_chunk or by_clf:
            lines += [
                "",
                "Mean recall@6 inside the 3×3 (so each factor is averaged over the other two):",
                "",
                "| factor | mean recall@6 |",
                "|---|---:|",
            ]
            chunk_labels = {"existing": "existing chunks", "rule": "rule chunks", "llm": "llm chunks"}
            clf_labels = {"none": "no classifier", "kmeans": "kmeans", "llm": "llm classifier"}
            for key in ("existing", "rule", "llm"):
                vals = by_chunk.get(key)
                if vals:
                    lines.append(f"| {chunk_labels[key]} | {round(sum(vals) / len(vals), 4)} |")
            for key in ("none", "kmeans", "llm"):
                vals = by_clf.get(key)
                if vals:
                    lines.append(f"| {clf_labels[key]} | {round(sum(vals) / len(vals), 4)} |")
        extra = [n for n in EXTRA_METHODS if n in rows]
        if extra:
            lines += [
                "",
                "## Stronger classifier on `advanced_llmchunk_llmfilter`",
                "",
                "`gpt-5o-mini` does not exist. Chunking / rewrite / rerank stay `gpt-4o-mini`.",
                "Only chunk labels and query routing use the listed classifier model.",
                "",
                "| method | classifier model | recall@6 | Δ vs `advanced_llmchunk_llmfilter` | hit@6 | MRR |",
                "|---|---|---:|---:|---:|---:|",
            ]
            base_llm = _pick(rows["advanced_llmchunk_llmfilter"]["metrics"]) if "advanced_llmchunk_llmfilter" in rows else None
            for name in extra:
                metrics = rows[name]["metrics"]
                score = _pick(metrics)
                model = EXTRA_METHODS[name][2]
                if base_llm is None:
                    delta = "—"
                else:
                    d = round(score - base_llm, 4)
                    delta = f"+{d}" if d > 0 else str(d)
                lines.append(
                    f"| `{name}` | {model} | {score} | {delta} | "
                    f"{(metrics.get('hit_at_k') or {}).get('6')} | {metrics.get('mrr')} |")
        lines += [
            "",
            "Passage recall is **not** comparable across chunkers. Gold spans were annotated on the original",
            "corpus windows; remapped rule/LLM chunk ids rarely overlap those spans even when the source id",
            "is correct. Use source recall / hit / MRR to compare cells.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    if os.environ.get("SUPPORTDESK_CONFIG", "").endswith("sample.toml"):
        print("WARNING: SUPPORTDESK_CONFIG points at the sample corpus; unset it for the real eval.",
              file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("offline", "replay", "live"), default="offline")
    ap.add_argument("--gold", type=Path, default=None,
                    help="an annotated dump to use as gold instead of data/eval "
                         f"(for example {DEFAULT_ANNOTATED})")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="comma list from " + ",".join(ALL_METHODS))
    ap.add_argument("--combo", action=argparse.BooleanOptionalAction, default=True,
                    help="after the four methods, run advanced + the winning classifier (default on)")
    ap.add_argument("--ablation", action="store_true",
                    help="run every advanced × chunker × classifier cell; reuse finished runs")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    wanted = list(ADVANCED_ABLATION) if args.ablation else [
        _canonical(m.strip()) for m in args.methods.split(",") if m.strip()]
    if args.ablation:
        args.combo = False
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
    print("gold sources / case: " +
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
        label = METHOD_LABELS.get(method)
        if label:
            print(f"  {label}")
        if not args.rebuild:
            finished = _load_finished_run(method)
            if finished:
                rows[method] = finished
                print(f"reusing finished run {finished['run']}")
                print(format_metrics(finished["metrics"]))
                continue
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
        elif method in EXTRA_METHODS:
            chunk, clf, clf_model = EXTRA_METHODS[method]
            rows[method] = _run_advanced_ablation(
                method, chunk, clf, mode=args.mode, cases=cases, gold=args.gold,
                exclusions=exclusions, records=records, prompt=prompt, rebuild=args.rebuild,
                classifier_model=clf_model)
        elif method in ADVANCED_ABLATION:
            chunk, clf = ADVANCED_ABLATION[method]
            rows[method] = _run_advanced_ablation(
                method, chunk, clf, mode=args.mode, cases=cases, gold=args.gold,
                exclusions=exclusions, records=records, prompt=prompt, rebuild=args.rebuild)
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
        print(f"\n=== combo existing-index + winner classifier (winner {winner}) ===")
        s = _load_settings(args.mode, "advanced")
        c = build_clients(s)
        index = VectorIndex.load(s.paths.index_dir, expect_model=c.embedding_model,
                                 expect_chunking=s.chunking)
        retriever, clf = _combo_retriever(winner, index, c, s, prompt, rebuild=args.rebuild)
        name = "advanced_exist_llmfilter" if clf == "llm" else "advanced_exist_kmeans"
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
            key = _canonical(name)
            if key not in rows and body.get("metrics"):
                rows[key] = body
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
