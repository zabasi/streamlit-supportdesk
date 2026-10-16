"""Discover routing topics from the whole corpus with an LLM, for review and then labelling.

Reads the documentation section summaries (paths.summaries) and a seeded sample of issues, comments,
release notes and linked sources, grows a topic list batch by batch and merges it into 10 to 15 topics
(retrieval/topic_discovery.py). Writes the list as JSON to review and edit before labelling the chunks:

    python scripts/discover_topics.py --mode live
    python scripts/build_class_labels.py --mode live --method llm --topics data/classes/topics.json \
      --set paths.class_labels=data/classes/discovered.jsonl

A handful of calls (about $0.005).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from supportdesk.config import Settings
from supportdesk.data.corpus import load_corpus
from supportdesk.llm.factory import build_clients
from supportdesk.retrieval.topic_discovery import discover_topics, sample_items

PER_TYPE = {"issue": 25, "comment": 25, "release_section": 12, "linked_reference": 8}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    parser.add_argument("--output", type=Path, default=Path("data/classes/topics.json"))
    parser.add_argument("--seed", type=int, default=0, help="which issues and comments are sampled")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    clients = build_clients(settings)
    summaries = [json.loads(line) for line in Path(settings.paths.summaries).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    records = load_corpus(settings.paths.data_dir).records
    items = sample_items(records, summaries, per_type=PER_TYPE, seed=args.seed)
    mark = clients.ledger.mark()
    topics = discover_topics(items, clients.chat,
                             grow_prompt=Path(settings.prompts.topic_discover).read_text(encoding="utf-8"),
                             refine_prompt=Path(settings.prompts.topic_refine).read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "topics": topics,
        "made": {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": settings.models.chat,
                 "summaries": len(summaries), "sampled": PER_TYPE, "seed": args.seed}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for t in topics:
        print(f"  {t['name']:<28} {t['definition']}")
    spent = clients.ledger.summarize(mark)
    print(f"{len(topics)} topics from {len(items)} texts -> {args.output} | cost ${spent.get('cost_usd', 0):.4f}")
    print("Review and edit the list, then label the chunks with build_class_labels.py --topics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
