"""Discover routing topics from the corpus with an LLM (TopicGPT-style), for the class_doc_slots retriever.

The model reads the corpus in compact form, batch by batch, and grows a topic list (adding a topic when
several texts fit none of the existing ones); a last pass merges near-duplicates into the target range.
The input mixes the documentation section summaries (all of the documentation, compactly) with a sample of
issues, comments, release notes and linked sources (their kinds of problem are not all documented).
"""
from __future__ import annotations

import json
import random
import re

from ..pipelines.parsing import NotJson, extract_json

_NAME = re.compile(r"[^a-z0-9_]+")


def clean_topics(payload, most: int) -> list[dict] | None:
    """The {"name", "definition"} topics in a model answer (names made snake_case, duplicates dropped), or
    None when the answer holds no usable topic."""
    rows = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    out, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _NAME.sub("_", str(row.get("name", "")).strip().lower()).strip("_")
        definition = " ".join(str(row.get("definition", "")).split())
        if name and definition and name not in seen:
            seen.add(name)
            out.append({"name": name, "definition": definition})
    return out[:most] or None


def sample_items(records, summaries: list[dict], *, per_type: dict[str, int], seed: int = 0,
                 chars: int = 300) -> list[str]:
    """The texts to read: every documentation section summary, plus a seeded sample of each other source
    type (its title and first characters), shuffled together."""
    items = [f"[documentation] {s.get('title', '')} > {s.get('heading', '')}: {s['summary']}" for s in summaries]
    rng = random.Random(seed)
    for kind, n in per_type.items():
        pool = sorted((r for r in records if r.source_type == kind), key=lambda r: r.source_id)
        for r in rng.sample(pool, min(n, len(pool))):
            items.append(f"[{kind}] {r.title}: {' '.join(r.text.split())[:chars]}")
    rng.shuffle(items)
    return items


def discover_topics(items: list[str], chat, *, grow_prompt: str, refine_prompt: str, batch: int = 45,
                    most: int = 20, low: int = 10, high: int = 15) -> list[dict]:
    """Grow a topic list over the batches, then merge it into ``low``..``high`` topics. An unusable answer
    keeps the list as it was; an "other" topic is always kept for texts that fit none."""
    topics: list[dict] = []
    grow = grow_prompt.replace("{max_topics}", str(most))
    for start in range(0, len(items), batch):
        listing = "\n".join(f"{n}. {text}" for n, text in enumerate(items[start:start + batch]))
        user = f"Current topics:\n{json.dumps({'topics': topics}, ensure_ascii=False)}\n\nTexts:\n{listing}"
        try:
            raw, _ = chat.complete([{"role": "system", "content": grow}, {"role": "user", "content": user}],
                                   stage="topic_discover", json_mode=True)
            topics = clean_topics(extract_json(raw), most) or topics
        except (NotJson, ValueError, TypeError):
            continue
    if topics:
        refine = refine_prompt.replace("{low}", str(low)).replace("{high}", str(high))
        try:
            raw, _ = chat.complete([{"role": "system", "content": refine},
                                    {"role": "user", "content": json.dumps({"topics": topics}, ensure_ascii=False)}],
                                   stage="topic_refine", json_mode=True)
            topics = clean_topics(extract_json(raw), most) or topics
        except (NotJson, ValueError, TypeError):
            pass
    if not any(t["name"] == "other" for t in topics):
        topics.append({"name": "other", "definition": "none of the other topics, or too mixed to pick one"})
    return topics


__all__ = ["clean_topics", "discover_topics", "sample_items"]
