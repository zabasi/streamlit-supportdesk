"""Label every chunk of an index with a topic class, for the class_doc_slots retriever.

  --method llm     the model assigns each chunk one of the fixed Streamlit topics in
                   prompts.topic_classify (batches of 16; unusable answers fall back to keywords);
                   with --overlap, 1 to 3 topics per chunk; with --topics FILE, the topics of FILE
                   (from scripts/discover_topics.py), 1 to 3 per chunk
  --method kmeans  spherical k-means, documentation-anchored, into --k clusters (no model call)

Writes paths.class_labels (one {"chunk_id", "label"} line per chunk) and a manifest beside it. Build once
per index: the retriever refuses labels built for another index.

    python scripts/build_class_labels.py --mode live --method llm
    python scripts/build_class_labels.py --mode live --method llm --overlap \
      --set paths.class_labels=data/classes/fixed_overlap.jsonl
    python scripts/build_class_labels.py --mode live --method llm --topics data/classes/topics.json \
      --set paths.class_labels=data/classes/discovered.jsonl
    python scripts/build_class_labels.py --mode live --method kmeans --set paths.class_labels=data/classes/kmeans.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from supportdesk.config import Settings
from supportdesk.llm.factory import build_clients
from supportdesk.retrieval.class_routing import (
    FIXED_TOPICS,
    classify_chunks_llm,
    classify_chunks_multi,
    kmeans_labels,
    topic_prompt,
    topics_file,
)
from supportdesk.retrieval.clusters import index_vectors
from supportdesk.retrieval.index import VectorIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--method", choices=["llm", "kmeans"], default="llm")
    parser.add_argument("--k", type=int, default=8, help="kmeans: number of classes")
    parser.add_argument("--overlap", action="store_true", help="llm: 1 to 3 of the fixed topics per chunk")
    parser.add_argument("--topics", type=Path, default=None,
                        help="llm: a topic list from scripts/discover_topics.py, 1 to 3 topics per chunk")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    clients = build_clients(settings)
    index = VectorIndex.load(settings.paths.index_dir, expect_model=clients.embedding_model)
    chunks = index.chunks
    mark = clients.ledger.mark()
    topics = None
    if args.method == "llm" and (args.overlap or args.topics):
        topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"] if args.topics else FIXED_TOPICS
        multi = classify_chunks_multi(chunks, clients.chat, prompt=topic_prompt(topics),
                                      allowed=[t["name"] for t in topics])
        rows = [{"chunk_id": c.chunk_id, "labels": labs} for c, labs in zip(chunks, multi)]
        labels = [lab for labs in multi for lab in labs]
    elif args.method == "llm":
        prompt = Path(settings.prompts.topic_classify).read_text(encoding="utf-8")
        labels = classify_chunks_llm(chunks, clients.chat, prompt=prompt)
        rows = [{"chunk_id": c.chunk_id, "label": lab} for c, lab in zip(chunks, labels)]
    else:
        labels, _ = kmeans_labels(index_vectors(index), [c.source_type for c in chunks], k=args.k)
        rows = [{"chunk_id": c.chunk_id, "label": lab} for c, lab in zip(chunks, labels)]
    out = Path(settings.paths.class_labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    sidecar = topics_file(out)  # the topic list routing must use with these labels, or none
    if topics is not None:
        sidecar.write_text(json.dumps({"topics": topics}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif sidecar.exists():
        sidecar.unlink()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": args.method,
                "overlap": bool(topics), "topics": str(args.topics) if args.topics else None,
                "chunks": len(chunks), "index_dir": str(settings.paths.index_dir),
                "model": settings.models.chat if args.method == "llm" else None}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = Counter(labels)
    docs = Counter(lab for c, r in zip(chunks, rows) if c.source_type == "documentation"
                   for lab in (r["labels"] if "labels" in r else [r["label"]]))
    for label, n in counts.most_common():
        print(f"  {label:<22} {n:>4} chunks ({docs.get(label, 0)} documentation)")
    spent = clients.ledger.summarize(mark)
    per_chunk = [len(r["labels"] if "labels" in r else [r["label"]]) for r in rows]
    unlabelled = sum(n == 0 for n in per_chunk)
    print(f"topics per chunk: mean {sum(per_chunk) / len(per_chunk):.2f} | without a topic: {unlabelled}")
    if unlabelled > 0.05 * len(rows):
        print(f"WARNING: {unlabelled} chunks have no topic: with class_mode = filter they can never be found; "
              "check the topic list and the labelling answers before using these labels")
    print(f"wrote {out} ({len(chunks)} chunks, {args.method}) | cost ${spent.get('cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
