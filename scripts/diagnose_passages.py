"""Where does each gold passage get lost? Traces every gold passage of the evaluation cases through a
pipeline's retrieval: its candidates (before reranking), the chunks it keeps (what the model sees),
and the rank of the chunk that holds the passage in a search over all documentation.

    python scripts/diagnose_passages.py                       # advanced, dev cases, settings as configured
    python scripts/diagnose_passages.py --mode live --output docs/reports/passage-funnel.md
    python scripts/diagnose_passages.py --pipeline clean_doc_slots --set retrieval.top_k=6

Read-only: it changes nothing. Query embeddings and model calls come from the cache when they were
made before (for example by `supportdesk eval-retrieval --pipeline advanced`), so it usually costs $0.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportdesk.cli import _load, load_dotenv_file
from supportdesk.config import Settings
from supportdesk.eval.cases import load_cases
from supportdesk.eval.metrics import passage_coverage

STAGES = ("in the prompt", "dropped by the reranker", "right page, other chunk", "page not among the candidates")


def _span_view(chunk) -> dict:
    return {"source_id": chunk.source_id, "start_char": chunk.start_char, "end_char": chunk.end_char}


def classify(span: dict, candidates: list[dict], kept: list[dict], threshold: float) -> str:
    """Where one gold passage ends up: one of STAGES."""
    if (passage_coverage(kept, span) or 0) >= threshold:
        return STAGES[0]
    if (passage_coverage(candidates, span) or 0) >= threshold:
        return STAGES[1]
    if any(c["source_id"] == span["source_id"] for c in candidates):
        return STAGES[2]
    return STAGES[3]


def holding_chunk(span: dict, chunks: list) -> tuple[str | None, float]:
    """The chunk that covers most of the passage, and how much of it."""
    best, best_cover = None, 0.0
    for chunk in chunks:
        if chunk.source_id == span["source_id"]:
            cover = passage_coverage([_span_view(chunk)], span) or 0.0
            if cover > best_cover:
                best, best_cover = chunk.chunk_id, cover
    return best, best_cover


def diagnose(settings: Settings, split: str) -> tuple[Counter, list[dict]]:
    clients, index, pipeline = _load(settings)
    threshold = settings.eval.passage_found_fraction
    cases = [c for c in load_cases(*settings.eval.case_files, split=split) if c.evidence_spans]
    docs = [i for i, c in enumerate(index.chunks) if c.source_type == "documentation"]
    funnel, rows = Counter(), []
    for case in cases:
        as_of = case.input.created_at if settings.retrieval.historical_filter else None
        query = pipeline.query(case.input, {})[0]
        candidates = [_span_view(h.chunk) for h in pipeline.retriever.search(query, k=settings.pool_size, as_of=as_of)]
        kept = [_span_view(h.chunk) for h in pipeline.retrieve(case.input).hits]
        vector, _ = clients.embedding.embed([query], stage="query")
        found = dict(index.search(vector[0], limit=len(docs), as_of=as_of, source_types=("documentation",)))
        ranked = sorted(found, key=lambda i: (-round(found[i], 6), i))
        doc_rank = {index.chunks[i].chunk_id: r for r, i in enumerate(ranked, start=1)}
        for span in case.evidence_spans:
            stage = classify(span, candidates, kept, threshold)
            chunk_id, cover = holding_chunk(span, index.chunks)
            funnel[stage] += 1
            heading = next((line.strip() for line in span["text"].splitlines() if line.strip()), "")[:60]
            rows.append({"case": case.case_id, "page": span["source_id"].rsplit("/", 1)[-1], "passage": heading,
                         "stage": stage, "chunk": chunk_id, "cover": cover,
                         "doc_rank": doc_rank.get(chunk_id)})
    return funnel, rows


def render(funnel: Counter, rows: list[dict], pipeline: str, split: str) -> str:
    total = sum(funnel.values()) or 1
    out = [f"# Where gold passages are lost: {pipeline}, {split} cases", "",
           "| Where the gold passage ends up | Passages | Share |", "|---|---|---|"]
    out += [f"| {stage} | {funnel[stage]} | {funnel[stage] / total:.0%} |" for stage in STAGES]
    legend = ("Doc rank: the rank of the chunk holding the passage in a search over all documentation "
              "(the pipeline keeps the top documentation candidates). Cover: how much of the passage that "
              "one chunk holds.")
    out += ["", legend, "",
            "| Case | Page | Passage starts with | Where it ends up | Doc rank | Cover |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (STAGES.index(r["stage"]), r["case"])):
        rank = "-" if r["doc_rank"] is None else str(r["doc_rank"])
        passage = r["passage"].replace("|", "\\|")
        out.append(f"| {r['case']} | {r['page']} | {passage} | {r['stage']} | {rank} | {r['cover']:.0%} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", default="advanced")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mode", choices=("offline", "replay", "live"))
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--output", type=Path, help="also write the Markdown report here")
    args = parser.parse_args(argv)
    load_dotenv_file()
    overrides = [f"pipeline.name={args.pipeline}", *args.set] + ([f"run.mode={args.mode}"] if args.mode else [])
    settings = Settings.load(overrides=overrides)
    funnel, rows = diagnose(settings, args.split)
    if not rows:
        print("no evaluation cases with gold passages (evidence_spans) in this split", file=sys.stderr)
        return 1
    report = render(funnel, rows, args.pipeline, args.split)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
