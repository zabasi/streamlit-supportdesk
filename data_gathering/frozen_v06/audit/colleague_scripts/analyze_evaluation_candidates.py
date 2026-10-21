from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "candidate_cases.json"

TOP_N_PER_TOPIC = 8

TOPICS = {
    "session_state": [
        "session_state",
        "session state",
        "st.session_state",
    ],
    "rerun_execution": [
        "st.rerun",
        "rerun",
        "rerun",
        "execution",
    ],
    "widgets": [
        "widget",
        "widgets",
        "button",
        "selectbox",
        "multiselect",
        "checkbox",
        "radio",
        "slider",
        "text_input",
        "text_area",
        "file_uploader",
    ],
    "forms_callbacks": [
        "form",
        "forms",
        "callback",
        "on_click",
        "on_change",
    ],
    "caching": [
        "cache_data",
        "cache_resource",
        "caching",
        "st.cache",
    ],
    "multipage_navigation": [
        "multipage",
        "pages",
        "navigation",
        "st.navigation",
        "st.page",
    ],
}


def load_records() -> list[dict]:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        records = data.get("records", data.get("corpus", data))
    else:
        records = data

    if not isinstance(records, list):
        raise ValueError("Could not find a list of corpus records.")

    return [r for r in records if isinstance(r, dict)]


def searchable_text(record: dict) -> str:
    parts = [
        str(record.get("title", "")),
        str(record.get("text", "")),
    ]

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        parts.extend(str(v) for v in metadata.values())

    return "\n".join(parts).lower()


def topic_matches(text: str) -> list[str]:
    matched = []

    for topic, terms in TOPICS.items():
        if any(term.lower() in text for term in terms):
            matched.append(topic)

    return matched


def score_record(record: dict, text: str) -> int:
    score = 0
    source_type = record.get("source_type", "")
    content_type = record.get("content_type", "")

    if source_type == "issue":
        score += 5
    elif source_type == "comment":
        score += 4
    elif source_type == "document":
        score += 3
    elif source_type == "release_note":
        score += 2

    if content_type == "issue":
        score += 2
    elif content_type == "issue_comment":
        score += 2
    elif content_type == "documentation":
        score += 2
    elif content_type == "release_note":
        score += 1

    if record.get("parent_id"):
        score += 1

    if len(text) >= 300:
        score += 1

    if len(text) >= 1000:
        score += 1

    return score


def compact_record(record: dict, topics: list[str], text: str) -> dict:
    return {
        "id": record.get("id"),
        "source_type": record.get("source_type"),
        "content_type": record.get("content_type"),
        "parent_id": record.get("parent_id"),
        "title": record.get("title", ""),
        "url": record.get("url", ""),
        "topics": topics,
        "score": score_record(record, text),
        "text_preview": re.sub(r"\s+", " ", record.get("text", ""))[:500],
    }


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    records = load_records()

    topic_records: dict[str, list[dict]] = defaultdict(list)
    topic_counts = Counter()
    source_counts = Counter()
    content_type_counts = Counter()

    for record in records:
        text = searchable_text(record)
        topics = topic_matches(text)

        source_counts[record.get("source_type", "unknown")] += 1
        content_type_counts[record.get("content_type", "unknown")] += 1

        for topic in topics:
            topic_counts[topic] += 1
            topic_records[topic].append(
                compact_record(record, topics, text)
            )

    candidates = {}

    for topic, items in topic_records.items():
        items.sort(
            key=lambda x: (
                x["score"],
                len(x.get("text_preview", "")),
            ),
            reverse=True,
        )

        # Prefer diverse source types among the top candidates.
        selected = []
        seen_sources = set()

        for item in items:
            if item["source_type"] not in seen_sources:
                selected.append(item)
                seen_sources.add(item["source_type"])

            if len(selected) >= 4:
                break

        for item in items:
            if len(selected) >= TOP_N_PER_TOPIC:
                break
            if item["id"] not in {x["id"] for x in selected}:
                selected.append(item)

        candidates[topic] = selected[:TOP_N_PER_TOPIC]

    report = {
        "schema_version": "1.0",
        "dataset_version": "runtime_state_widgets-v01",
        "purpose": (
            "Candidate evidence discovery for manual/assisted evaluation-case "
            "construction. This file is not the final evaluation set."
        ),
        "corpus_total": len(records),
        "source_counts": dict(source_counts),
        "content_type_counts": dict(content_type_counts),
        "topic_counts": dict(topic_counts),
        "topics": candidates,
        "selection_policy": {
            "top_n_per_topic": TOP_N_PER_TOPIC,
            "prefer_source_diversity": True,
            "no_final_case_generation": True,
            "evidence_must_be_verified_before_use": True,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("EVALUATION CANDIDATE ANALYSIS COMPLETED")
    print(f"Corpus records: {len(records)}")
    print()
    print("Source counts:")
    for key, value in source_counts.most_common():
        print(f"  {key}: {value}")

    print()
    print("Topic counts:")
    for key, value in topic_counts.most_common():
        print(f"  {key}: {value}")

    print()
    print(f"Candidate file:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
