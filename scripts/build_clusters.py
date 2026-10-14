"""Group the index's chunks into topic clusters and label each with keywords (cluster routing experiment).

k-means on the chunk embeddings makes clusters of about retrieval.cluster_size chunks; one model call per
cluster names it and lists its keywords from the chunks nearest its centre (unusable output falls back to
the cluster's most distinctive words). Writes paths.clusters and a manifest beside it; the
cluster_doc_slots retriever reads them. Build once per index (about $0.003 in live mode).

    python scripts/build_clusters.py --mode live
    python scripts/build_clusters.py --mode live --set retrieval.cluster_size=60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from supportdesk.config import Settings
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.parsing import NotJson, extract_json
from supportdesk.retrieval.clusters import distinctive_words, index_vectors, kmeans
from supportdesk.retrieval.index import VectorIndex

SAMPLES = 30         # chunks shown to the model per cluster: those nearest its centre
SAMPLE_CHARS = 300   # characters of each


def label_cluster(clients, prompt: str, samples: list) -> tuple[str, list[str]] | None:
    """(label, keywords) from the model, or None when its answer is unusable."""
    listing = "\n".join(f"- [{c.source_type}] {c.title}: {' '.join(c.text.split())[:SAMPLE_CHARS]}"
                        for c in samples)
    text, _ = clients.chat.complete([{"role": "system", "content": prompt},
                                     {"role": "user", "content": listing}], stage="cluster_label", json_mode=True)
    try:
        payload = extract_json(text)
    except (NotJson, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    label = " ".join(str(payload.get("label", "")).split())[:80]
    keywords = [" ".join(str(k).split())[:60] for k in payload.get("keywords") or [] if str(k).strip()]
    return (label, list(dict.fromkeys(keywords))[:20]) if label and keywords else None


def build(settings: Settings, index: VectorIndex, clients, *, seed: int = 0) -> Path:
    size = settings.retrieval.cluster_size
    chunks = index.chunks
    vectors = index_vectors(index)
    k = max(1, math.ceil(len(chunks) / size))
    labels = kmeans(vectors, k, seed=seed)
    unit = vectors / np.where(np.linalg.norm(vectors, axis=1, keepdims=True) == 0, 1.0,
                              np.linalg.norm(vectors, axis=1, keepdims=True))
    prompt = Path(settings.prompts.label_cluster).read_text(encoding="utf-8")
    rows = []
    for c in range(int(labels.max()) + 1):
        inside = [i for i in range(len(chunks)) if labels[i] == c]
        centre = unit[inside].mean(axis=0)
        nearest = sorted(inside, key=lambda i: (-float(unit[i] @ centre), i))[:SAMPLES]
        named = label_cluster(clients, prompt, [chunks[i] for i in nearest])
        if named is None:  # unusable model answer: the cluster's own most distinctive words
            words = distinctive_words([chunks[i].text for i in inside],
                                      [chunks[i].text for i in range(len(chunks)) if labels[i] != c])
            named = (" ".join(words[:4]), words)
            how = "words"
        else:
            how = "model"
        types = {}
        for i in inside:
            types[chunks[i].source_type] = types.get(chunks[i].source_type, 0) + 1
        rows.append({"cluster": c, "label": named[0], "keywords": named[1], "labelled_by": how,
                     "size": len(inside), "types": types, "chunk_ids": [chunks[i].chunk_id for i in inside]})
    out = Path(settings.paths.clusters)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "chunks": len(chunks), "clusters": len(rows), "cluster_size": size, "seed": seed,
                "embedding_model": index.manifest.get("embedding_model"), "label_model": settings.models.chat,
                "index_dir": str(settings.paths.index_dir)}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    parser.add_argument("--seed", type=int, default=0, help="k-means start (the same seed gives the same clusters)")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    clients = build_clients(settings)
    index = VectorIndex.load(settings.paths.index_dir, expect_model=clients.embedding_model)
    mark = clients.ledger.mark()
    out = build(settings, index, clients, seed=args.seed)
    for row in (json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()):
        kinds = ", ".join(f"{t} {n}" for t, n in sorted(row["types"].items()))
        print(f"cluster {row['cluster']}: {row['size']:>3} chunks ({kinds}) | {row['label']}")
        print(f"            {', '.join(row['keywords'][:10])}")
    spent = clients.ledger.summarize(mark)
    print(f"wrote {out} | cost ${spent.get('cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
