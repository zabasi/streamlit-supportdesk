"""Label every chunk of an index with a topic class, for the class_doc_slots retriever.

  --method llm     the model assigns each chunk one of the fixed Streamlit topics in
                   prompts.topic_classify (batches of 16; unusable answers fall back to keywords)
  --method kmeans  spherical k-means, documentation-anchored, into --k clusters (no model call)

Writes paths.class_labels (one {"chunk_id", "label"} line per chunk) and a manifest beside it. Build once
per index: the retriever refuses labels built for another index.

    python scripts/build_class_labels.py --mode live --method llm
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
from supportdesk.retrieval.class_routing import classify_chunks_llm, kmeans_labels
from supportdesk.retrieval.clusters import index_vectors
from supportdesk.retrieval.index import VectorIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--method", choices=["llm", "kmeans"], default="llm")
    parser.add_argument("--k", type=int, default=8, help="kmeans: number of classes")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    clients = build_clients(settings)
    index = VectorIndex.load(settings.paths.index_dir, expect_model=clients.embedding_model)
    chunks = index.chunks
    mark = clients.ledger.mark()
    if args.method == "llm":
        prompt = Path(settings.prompts.topic_classify).read_text(encoding="utf-8")
        labels = classify_chunks_llm(chunks, clients.chat, prompt=prompt)
    else:
        labels, _ = kmeans_labels(index_vectors(index), [c.source_type for c in chunks], k=args.k)
    out = Path(settings.paths.class_labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "label": lab}) + "\n" for c, lab in zip(chunks, labels)),
                   encoding="utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": args.method,
                "chunks": len(chunks), "index_dir": str(settings.paths.index_dir),
                "model": settings.models.chat if args.method == "llm" else None}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = Counter(labels)
    docs = Counter(lab for c, lab in zip(chunks, labels) if c.source_type == "documentation")
    for label, n in counts.most_common():
        print(f"  {label:<22} {n:>4} chunks ({docs.get(label, 0)} documentation)")
    spent = clients.ledger.summarize(mark)
    print(f"wrote {out} ({len(chunks)} chunks, {args.method}) | cost ${spent.get('cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
