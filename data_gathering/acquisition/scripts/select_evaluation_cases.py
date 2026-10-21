import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
CANDIDATE_PATH = ROOT / "data" / "evaluation" / "candidate_cases.json"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "selected_candidates.json"

TARGET_MIN = 40
TARGET_MAX = 50


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_records(corpus):
    if isinstance(corpus, dict):
        records = corpus.get("records")

        if records is None:
            records = corpus.get("corpus")

        if records is None:
            raise ValueError(
                "Could not find corpus records. "
                "Expected 'records' or 'corpus'."
            )

        return records

    if isinstance(corpus, list):
        return corpus

    raise ValueError("Unsupported corpus format.")


def get_issue_number(record):
    metadata = record.get("metadata", {})

    value = (
        metadata.get("issue_number")
        or metadata.get("issue")
        or record.get("issue_number")
    )

    if value is None:
        parent_id = record.get("parent_id", "")

        match = re.match(r"issue-(\d+)", parent_id)

        if match:
            return int(match.group(1))

    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def is_issue(record):
    return record.get("source_type") == "issue"


def is_comment(record):
    return record.get("source_type") == "comment"


def is_document(record):
    return record.get("source_type") == "document"


def is_release_note(record):
    return record.get("source_type") == "release_note"


# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------

TROUBLESHOOTING_TERMS = {
    "error": 8,
    "exception": 8,
    "traceback": 8,
    "bug": 7,
    "broken": 7,
    "breaks": 7,
    "broken behavior": 8,
    "unexpected": 7,
    "unexpected behavior": 8,
    "incorrect": 6,
    "wrong": 6,
    "fails": 7,
    "failed": 7,
    "failure": 7,
    "does not work": 8,
    "doesn't work": 8,
    "not working": 8,
    "not work": 7,
    "missing": 5,
    "disappear": 7,
    "disappears": 7,
    "lost": 7,
    "drops": 7,
    "dropped": 7,
    "reset": 6,
    "resets": 6,
    "rerun": 5,
    "state": 4,
    "session state": 7,
    "widget state": 7,
    "regression": 9,
    "reproduce": 7,
    "reproduction": 7,
    "workaround": 9,
    "stack trace": 8,
    "crash": 8,
    "crashes": 8,
    "issue": 4,
    "problem": 5,
    "cannot": 5,
    "can't": 5,
    "unable": 6,
    "incorrectly": 6,
    "silently": 6,
    "double input": 8,
    "disappears": 7,
    "duplicate": 4,
    "stale": 5,
}


FEATURE_REQUEST_TERMS = {
    "add option": -9,
    "feature request": -10,
    "request feature": -10,
    "proposal": -8,
    "enhancement": -7,
    "would be nice": -7,
    "please add": -9,
    "support for": -5,
    "add support": -7,
    "new feature": -8,
    "coming soon": -8,
    "preload": -4,
    "make streamlit": -5,
}


HIGH_VALUE_COMMENT_TERMS = {
    "root cause": 12,
    "cause": 7,
    "because": 7,
    "workaround": 12,
    "solution": 8,
    "fix": 7,
    "fixed": 7,
    "reproduce": 8,
    "reproduction": 8,
    "confirmed": 6,
    "expected": 4,
    "actual": 4,
    "implementation": 6,
    "code": 3,
}


def keyword_score(text, terms):
    normalized = normalize_text(text)

    score = 0

    for term, weight in terms.items():
        if term in normalized:
            score += weight

    return score


def topic_score(text, topics):
    normalized = normalize_text(text)

    score = 0

    topic_terms = {
        "session_state": [
            "session_state",
            "session state",
            "sessionstate",
        ],
        "widgets": [
            "widget",
            "button",
            "selectbox",
            "multiselect",
            "checkbox",
            "radio",
            "slider",
            "text_input",
            "text input",
            "data_editor",
            "file_uploader",
            "file uploader",
        ],
        "rerun_execution": [
            "rerun",
            "st.rerun",
            "execution",
            "run",
            "script run",
        ],
        "forms_callbacks": [
            "form",
            "callback",
            "on_click",
            "on_change",
        ],
        "caching": [
            "cache",
            "caching",
            "st.cache",
            "cache_data",
            "cache_resource",
        ],
        "multipage_navigation": [
            "multipage",
            "multi-page",
            "page",
            "navigation",
            "pages",
        ],
    }

    for topic in topics:
        for term in topic_terms.get(topic, []):
            if term in normalized:
                score += 3

    return score


def classify_issue(record):
    title = record.get("title", "")
    text = f"{title}\n{record.get('text', '')}"

    score = keyword_score(text, TROUBLESHOOTING_TERMS)
    score += keyword_score(text, FEATURE_REQUEST_TERMS)

    return score


def classify_comment(record):
    text = record.get("text", "")

    score = keyword_score(text, HIGH_VALUE_COMMENT_TERMS)

    if len(text) > 150:
        score += 2

    if "```" in text:
        score += 2

    return score


# ---------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------

def extract_candidate_ids(candidate_data):
    ids = []

    if not isinstance(candidate_data, dict):
        return ids

    topics = candidate_data.get("topics", {})

    if isinstance(topics, dict):
        for topic_items in topics.values():
            if not isinstance(topic_items, list):
                continue

            for item in topic_items:
                if not isinstance(item, dict):
                    continue

                record_id = item.get("id")

                if record_id:
                    ids.append(record_id)

    return list(dict.fromkeys(ids))


def build_topic_map(candidate_data):
    topic_map = defaultdict(set)

    topics = candidate_data.get("topics", {})

    if not isinstance(topics, dict):
        return topic_map

    for topic, items in topics.items():
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            record_id = item.get("id")

            if record_id:
                topic_map[record_id].add(topic)

    return topic_map


# ---------------------------------------------------------
# Corpus indexing
# ---------------------------------------------------------

def build_indexes(records):
    by_id = {}
    issues_by_number = {}
    comments_by_issue = defaultdict(list)
    docs = []
    release_notes = []

    for record in records:
        record_id = record.get("id")

        if not record_id:
            continue

        by_id[record_id] = record

        if is_issue(record):
            issue_number = get_issue_number(record)

            if issue_number is not None:
                issues_by_number[issue_number] = record

        elif is_comment(record):
            issue_number = get_issue_number(record)

            if issue_number is not None:
                comments_by_issue[issue_number].append(record)

        elif is_document(record):
            docs.append(record)

        elif is_release_note(record):
            release_notes.append(record)

    return (
        by_id,
        issues_by_number,
        comments_by_issue,
        docs,
        release_notes,
    )


# ---------------------------------------------------------
# Issue centered cases
# ---------------------------------------------------------

def create_issue_candidate(
    issue,
    topic_map,
    comments_by_issue,
    by_id,
):
    issue_id = issue["id"]
    issue_number = get_issue_number(issue)

    title = issue.get("title", "")
    body = issue.get("text", "")

    topics = sorted(topic_map.get(issue_id, set()))

    if not topics:
        combined = f"{title}\n{body}"

        possible_topics = [
            "session_state",
            "widgets",
            "rerun_execution",
            "forms_callbacks",
            "caching",
            "multipage_navigation",
        ]

        for topic in possible_topics:
            if topic_score(combined, [topic]) > 0:
                topics.append(topic)

    issue_score = classify_issue(issue)

    related_comments = []

    if issue_number is not None:
        comments = comments_by_issue.get(issue_number, [])

        scored_comments = []

        for comment in comments:
            score = classify_comment(comment)

            if score > 0:
                scored_comments.append((score, comment))

        scored_comments.sort(
            key=lambda x: (
                x[0],
                len(x[1].get("text", "")),
            ),
            reverse=True,
        )

        for _, comment in scored_comments[:3]:
            related_comments.append(comment)

    evidence_ids = [issue_id]

    for comment in related_comments:
        if comment["id"] in by_id:
            evidence_ids.append(comment["id"])

    return {
        "candidate_id": f"candidate-{issue_number}",
        "family_id": f"family-{issue_number}",
        "primary_issue_id": issue_id,
        "issue_number": issue_number,
        "title": title,
        "topics": topics,
        "score": issue_score,
        "case_type": "troubleshooting",
        "multi_turn_recommended": bool(related_comments),
        "evidence_ids": evidence_ids,
        "supporting_comment_ids": [
            comment["id"]
            for comment in related_comments
        ],
        "selection_reason": [
            "Primary evidence is a real Streamlit issue.",
            "Related comments are linked through issue_number.",
            "Evidence IDs were verified against corpus.",
        ],
    }


# ---------------------------------------------------------
# Comment centered cases
# ---------------------------------------------------------

def create_comment_candidate(
    comment,
    topic_map,
    issues_by_number,
    by_id,
):
    issue_number = get_issue_number(comment)

    if issue_number is None:
        return None

    issue = issues_by_number.get(issue_number)

    if issue is None:
        return None

    comment_score = classify_comment(comment)

    if comment_score <= 0:
        return None

    issue_score = classify_issue(issue)

    topics = set(topic_map.get(comment["id"], set()))
    topics.update(topic_map.get(issue["id"], set()))

    combined = (
        issue.get("title", "")
        + "\n"
        + issue.get("text", "")
        + "\n"
        + comment.get("text", "")
    )

    possible_topics = [
        "session_state",
        "widgets",
        "rerun_execution",
        "forms_callbacks",
        "caching",
        "multipage_navigation",
    ]

    for topic in possible_topics:
        if topic_score(combined, [topic]) > 0:
            topics.add(topic)

    total_score = (
        issue_score
        + comment_score
        + 8
    )

    return {
        "candidate_id": f"comment-case-{comment['id']}",
        "family_id": f"family-{issue_number}",
        "primary_issue_id": issue["id"],
        "issue_number": issue_number,
        "primary_comment_id": comment["id"],
        "title": issue.get("title", ""),
        "topics": sorted(topics),
        "score": total_score,
        "case_type": "issue_with_technical_comment",
        "multi_turn_recommended": True,
        "evidence_ids": [
            issue["id"],
            comment["id"],
        ],
        "supporting_comment_ids": [
            comment["id"],
        ],
        "selection_reason": [
            "Issue is real troubleshooting evidence.",
            "Comment contains technical information, explanation, "
            "workaround, reproduction details, or implementation context.",
            "Issue/comment relationship was verified.",
        ],
    }


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------

def deduplicate_candidates(candidates):
    best_by_issue = {}

    for candidate in candidates:
        issue_number = candidate.get("issue_number")

        if issue_number is None:
            continue

        previous = best_by_issue.get(issue_number)

        if previous is None:
            best_by_issue[issue_number] = candidate
            continue

        previous_score = previous.get("score", 0)
        current_score = candidate.get("score", 0)

        if current_score > previous_score:
            best_by_issue[issue_number] = candidate

    return list(best_by_issue.values())


# ---------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------

def select_diverse(candidates):
    candidates = sorted(
        candidates,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    selected = []

    topic_counts = Counter()

    # First pass:
    # make sure different topics are represented.
    for candidate in candidates:
        topics = candidate.get("topics", [])

        if not topics:
            continue

        rare_topic = min(
            topics,
            key=lambda topic: topic_counts[topic],
        )

        if topic_counts[rare_topic] < 8:
            selected.append(candidate)

            for topic in topics:
                topic_counts[topic] += 1

    # Second pass:
    # fill remaining slots by score.
    selected_ids = {
        candidate["candidate_id"]
        for candidate in selected
    }

    for candidate in candidates:
        if len(selected) >= TARGET_MAX:
            break

        if candidate["candidate_id"] in selected_ids:
            continue

        selected.append(candidate)
        selected_ids.add(candidate["candidate_id"])

    # Remove accidental duplicates again.
    unique = {}

    for candidate in selected:
        issue_number = candidate.get("issue_number")

        if issue_number not in unique:
            unique[issue_number] = candidate

    selected = list(unique.values())

    selected.sort(
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    return selected[:TARGET_MAX]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def verify_evidence(selected, by_id):
    verified = []

    for candidate in selected:
        evidence_ids = candidate.get("evidence_ids", [])

        valid = True

        for evidence_id in evidence_ids:
            if evidence_id not in by_id:
                valid = False
                break

        if not valid:
            continue

        candidate["evidence_verified"] = True
        verified.append(candidate)

    return verified


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found:\n{CORPUS_PATH}"
        )

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{CANDIDATE_PATH}"
        )

    corpus_data = load_json(CORPUS_PATH)
    candidate_data = load_json(CANDIDATE_PATH)

    records = get_records(corpus_data)

    (
        by_id,
        issues_by_number,
        comments_by_issue,
        _docs,
        _release_notes,
    ) = build_indexes(records)

    topic_map = build_topic_map(candidate_data)

    candidate_ids = extract_candidate_ids(candidate_data)

    candidates_from_topic_file = []

    for record_id in candidate_ids:
        record = by_id.get(record_id)

        if record is None:
            continue

        candidates_from_topic_file.append(record)

    issue_candidates = []

    # -----------------------------------------------------
    # 1. Existing topic candidates
    # -----------------------------------------------------

    for record in candidates_from_topic_file:
        if not is_issue(record):
            continue

        score = classify_issue(record)

        # Ignore obvious feature-only records.
        if score < 3:
            continue

        candidate = create_issue_candidate(
            record,
            topic_map,
            comments_by_issue,
            by_id,
        )

        issue_candidates.append(candidate)

    # -----------------------------------------------------
    # 2. Add high-value comments from existing candidates
    # -----------------------------------------------------

    comment_candidates = []

    for record in candidates_from_topic_file:
        if not is_comment(record):
            continue

        candidate = create_comment_candidate(
            record,
            topic_map,
            issues_by_number,
            by_id,
        )

        if candidate is not None:
            comment_candidates.append(candidate)

    # -----------------------------------------------------
    # 3. If fewer than TARGET_MIN candidates exist,
    #    scan the complete corpus for more real issues.
    # -----------------------------------------------------

    all_issue_records = [
        record
        for record in records
        if is_issue(record)
    ]

    if len(issue_candidates) < TARGET_MIN:
        additional = []

        for issue in all_issue_records:
            issue_score = classify_issue(issue)

            if issue_score < 5:
                continue

            candidate = create_issue_candidate(
                issue,
                topic_map,
                comments_by_issue,
                by_id,
            )

            additional.append(candidate)

        issue_candidates.extend(additional)

    # -----------------------------------------------------
    # 4. Merge
    # -----------------------------------------------------

    all_candidates = (
        issue_candidates
        + comment_candidates
    )

    # -----------------------------------------------------
    # 5. Remove weak candidates
    # -----------------------------------------------------

    filtered = []

    for candidate in all_candidates:
        if candidate.get("score", 0) < 5:
            continue

        if not candidate.get("primary_issue_id"):
            continue

        if not candidate.get("evidence_ids"):
            continue

        filtered.append(candidate)

    # -----------------------------------------------------
    # 6. Deduplicate by issue family
    # -----------------------------------------------------

    deduplicated = deduplicate_candidates(filtered)

    # -----------------------------------------------------
    # 7. Verify evidence
    # -----------------------------------------------------

    verified = verify_evidence(
        deduplicated,
        by_id,
    )

    # -----------------------------------------------------
    # 8. Select diverse candidates
    # -----------------------------------------------------

    selected = select_diverse(verified)

    # -----------------------------------------------------
    # 9. Add rank
    # -----------------------------------------------------

    for index, candidate in enumerate(selected, start=1):
        candidate["rank"] = index

    # -----------------------------------------------------
    # 10. Statistics
    # -----------------------------------------------------

    topic_counts = Counter()

    for candidate in selected:
        for topic in candidate.get("topics", []):
            topic_counts[topic] += 1

    case_type_counts = Counter(
        candidate.get("case_type")
        for candidate in selected
    )

    multi_turn_count = sum(
        1
        for candidate in selected
        if candidate.get("multi_turn_recommended")
    )

    output = {
        "dataset_version": candidate_data.get(
            "dataset_version",
            "unknown",
        ),
        "selection_policy": {
            "target_min": TARGET_MIN,
            "target_max": TARGET_MAX,
            "primary_evidence": "real_issue",
            "prefer_troubleshooting": True,
            "prefer_issue_comment_relationship": True,
            "remove_duplicate_issue_families": True,
            "verify_evidence_against_corpus": True,
            "feature_requests_are_deprioritized": True,
            "no_final_case_generation": True,
        },
        "statistics": {
            "corpus_records": len(records),
            "topic_file_candidates": len(candidate_ids),
            "initial_issue_candidates": len(issue_candidates),
            "initial_comment_candidates": len(comment_candidates),
            "deduplicated_candidates": len(deduplicated),
            "verified_candidates": len(verified),
            "selected_candidates": len(selected),
            "multi_turn_recommended": multi_turn_count,
            "topic_counts": dict(topic_counts),
            "case_type_counts": dict(case_type_counts),
        },
        "candidates": selected,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("FINAL EVALUATION CANDIDATE SELECTION")
    print("=" * 60)

    print(f"Corpus records: {len(records)}")
    print(f"Topic-file candidates: {len(candidate_ids)}")
    print(f"Verified candidates: {len(verified)}")
    print(f"Selected candidates: {len(selected)}")
    print(f"Recommended multi-turn: {multi_turn_count}")

    print()
    print("Topic counts:")

    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")

    print()
    print("Top selected cases:")

    for candidate in selected[:10]:
        print(
            f"  {candidate['rank']:02d}. "
            f"issue-{candidate['issue_number']} | "
            f"score={candidate['score']} | "
            f"{candidate['title']}"
        )

    print()
    print(f"Output: {OUTPUT_PATH}")
    print()

    if len(selected) < TARGET_MIN:
        print(
            "WARNING: Fewer than 40 candidates were found. "
            "The corpus may not contain enough strong troubleshooting cases."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()