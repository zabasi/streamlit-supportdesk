import json
import re
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
SELECTED_PATH = ROOT / "data" / "evaluation" / "selected_candidates.json"
TEMPLATE_PATH = ROOT / "data" / "evaluation" / "cases.template.json"

DEV_PATH = ROOT / "data" / "evaluation" / "dev.json"
TEST_PATH = ROOT / "data" / "evaluation" / "test.json"

TOTAL_CASES = 30
DEV_CASES = 15
TEST_CASES = 15

MIN_MULTI_TURN = 10
MIN_TEST_MULTI_TURN = 5

RANDOM_SEED = 20261006


# =========================================================
# JSON
# =========================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# Corpus
# =========================================================

def get_records(corpus_data):
    if isinstance(corpus_data, list):
        return corpus_data

    if isinstance(corpus_data, dict):
        if "records" in corpus_data:
            return corpus_data["records"]

        if "corpus" in corpus_data:
            return corpus_data["corpus"]

    raise ValueError(
        "Could not find corpus records. "
        "Expected a list or 'records'/'corpus'."
    )


def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"`[^`]+`",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9_ ]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 4
    }


def build_corpus_index(records):
    by_id = {}

    issues = {}
    comments_by_issue = {}

    documents = []
    release_notes = []

    for record in records:
        record_id = record.get("id")

        if not record_id:
            continue

        by_id[record_id] = record

        source_type = record.get("source_type")

        if source_type == "issue":
            issue_number = get_issue_number(record)

            if issue_number is not None:
                issues[issue_number] = record

        elif source_type == "comment":
            issue_number = get_issue_number(record)

            if issue_number is not None:
                comments_by_issue.setdefault(
                    issue_number,
                    [],
                ).append(record)

        elif source_type == "document":
            documents.append(record)

        elif source_type == "release_note":
            release_notes.append(record)

    return (
        by_id,
        issues,
        comments_by_issue,
        documents,
        release_notes,
    )


def get_issue_number(record):
    metadata = record.get("metadata", {})

    value = (
        metadata.get("issue_number")
        or metadata.get("issue")
        or record.get("issue_number")
    )

    if value is None:
        parent_id = record.get(
            "parent_id",
            "",
        )

        match = re.match(
            r"issue-(\d+)",
            parent_id,
        )

        if match:
            return int(match.group(1))

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# =========================================================
# Selected candidates
# =========================================================

def get_selected_candidates(data):
    if isinstance(data, dict):
        candidates = data.get(
            "candidates",
            [],
        )

        if isinstance(candidates, list):
            return candidates

    if isinstance(data, list):
        return data

    raise ValueError(
        "selected_candidates.json does not contain "
        "a valid candidates list."
    )


# =========================================================
# Topic detection
# =========================================================

TOPIC_TERMS = {
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
        "data_editor",
        "dataframe",
        "text_input",
        "file_uploader",
    ],
    "rerun_execution": [
        "rerun",
        "st.rerun",
        "script run",
        "execution",
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
        "navigation",
        "page",
        "pages",
    ],
}


def detect_topics(text):
    normalized = normalize_text(text)

    topics = []

    for topic, terms in TOPIC_TERMS.items():
        for term in terms:
            if term in normalized:
                topics.append(topic)
                break

    return topics


# =========================================================
# Related evidence
# =========================================================

def similarity_score(source_text, target_text):
    source_tokens = tokenize(source_text)
    target_tokens = tokenize(target_text)

    if not source_tokens or not target_tokens:
        return 0

    intersection = source_tokens & target_tokens

    return len(intersection)


def find_related_documents(
    issue,
    documents,
    limit=2,
):
    issue_text = (
        issue.get("title", "")
        + "\n"
        + issue.get("text", "")
    )

    issue_topics = set(
        detect_topics(issue_text)
    )

    scored = []

    for document in documents:
        document_text = (
            document.get("title", "")
            + "\n"
            + document.get("text", "")
        )

        document_topics = set(
            detect_topics(document_text)
        )

        topic_overlap = len(
            issue_topics & document_topics
        )

        token_overlap = similarity_score(
            issue_text,
            document_text,
        )

        score = (
            topic_overlap * 10
            + min(token_overlap, 20)
        )

        if score > 0:
            scored.append(
                (
                    score,
                    document,
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored[:limit]
    ]


def find_related_release_notes(
    issue,
    release_notes,
    limit=1,
):
    issue_text = (
        issue.get("title", "")
        + "\n"
        + issue.get("text", "")
    )

    issue_topics = set(
        detect_topics(issue_text)
    )

    scored = []

    for note in release_notes:
        note_text = (
            note.get("title", "")
            + "\n"
            + note.get("text", "")
        )

        note_topics = set(
            detect_topics(note_text)
        )

        topic_overlap = len(
            issue_topics & note_topics
        )

        token_overlap = similarity_score(
            issue_text,
            note_text,
        )

        score = (
            topic_overlap * 8
            + min(token_overlap, 10)
        )

        if score >= 10:
            scored.append(
                (
                    score,
                    note,
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        note
        for _, note in scored[:limit]
    ]


# =========================================================
# Case generation
# =========================================================

def clean_issue_body(text):
    if not text:
        return ""

    text = text.strip()

    # Avoid enormous evaluation prompts.
    if len(text) > 3000:
        text = text[:3000] + "\n[issue body truncated]"

    return text


def build_user_question(issue):
    title = issue.get(
        "title",
        "Unknown issue",
    )

    body = clean_issue_body(
        issue.get("text", "")
    )

    if body:
        return (
            "I am investigating this Streamlit issue:\n\n"
            f"Title: {title}\n\n"
            f"Reported details:\n{body}\n\n"
            "Based only on the available evidence, explain "
            "what is reported, what is supported by the evidence, "
            "and what information is still missing."
        )

    return (
        "I am investigating this Streamlit issue:\n\n"
        f"Title: {title}\n\n"
        "Based only on the available evidence, explain "
        "what is reported and what information is still missing. "
        "Do not assume details that are not present in the evidence."
    )


def build_followup_question(issue):
    title = issue.get(
        "title",
        "this issue",
    )

    return (
        f"For the issue '{title}', what additional technical "
        "information should I provide to distinguish between "
        "the possible explanations? "
        "Please separate information that is already known "
        "from information that is still unknown."
    )


def build_expected_behavior(issue, comments):
    expected = [
        "Identify the reported problem without changing its meaning.",
        "Use the selected evidence to support factual claims.",
        "Distinguish reported facts from documented explanations "
        "and hypotheses.",
        "Do not treat the issue's closed/open state as proof "
        "that the problem is solved.",
        "Do not invent missing technical details.",
    ]

    if comments:
        expected.append(
            "Use the linked technical comment when it provides "
            "additional reproduction, explanation, workaround, "
            "or implementation evidence."
        )

    return expected


def build_must_not_claim(issue):
    return [
        "Do not claim that a closed issue is automatically solved.",
        "Do not claim a root cause unless the evidence supports it.",
        "Do not invent Streamlit versions, operating systems, "
        "Python versions, or reproduction steps.",
        "Do not claim that a workaround is guaranteed to solve "
        "the user's problem unless the evidence supports that claim.",
    ]


def build_must_ask(issue):
    body = normalize_text(
        issue.get("text", "")
    )

    questions = []

    missing_indicators = [
        (
            "version",
            "What Streamlit version are you using?"
        ),
        (
            "python",
            "What Python version are you using?"
        ),
        (
            "operating system",
            "What operating system are you using?"
        ),
        (
            "reproduce",
            "Can you provide the smallest reproducible example?"
        ),
    ]

    for keyword, question in missing_indicators:
        if keyword not in body:
            questions.append(question)

    # Keep the evaluation case concise.
    return questions[:2]


def build_case(
    candidate,
    issue,
    comments,
    related_docs,
    related_release_notes,
    split,
    multi_turn,
    case_number,
):
    evidence_ids = []

    # Primary issue is always first.
    issue_id = issue.get("id")

    if issue_id:
        evidence_ids.append(issue_id)

    # Use the candidate's verified comments first.
    candidate_comment_ids = set(
        candidate.get(
            "supporting_comment_ids",
            [],
        )
    )

    selected_comments = []

    for comment in comments:
        if comment.get("id") in candidate_comment_ids:
            selected_comments.append(comment)

    # If candidate comments are unavailable, use the strongest
    # comments belonging to the same issue.
    if not selected_comments:
        selected_comments = sorted(
            comments,
            key=lambda c: len(
                c.get("text", "")
            ),
            reverse=True,
        )[:2]

    for comment in selected_comments[:2]:
        comment_id = comment.get("id")

        if comment_id and comment_id not in evidence_ids:
            evidence_ids.append(comment_id)

    # Documentation is only included when it was found by
    # textual/topic relevance.
    for document in related_docs[:1]:
        document_id = document.get("id")

        if (
            document_id
            and document_id not in evidence_ids
        ):
            evidence_ids.append(document_id)

    # Release notes are optional evidence.
    for note in related_release_notes[:1]:
        note_id = note.get("id")

        if (
            note_id
            and note_id not in evidence_ids
        ):
            evidence_ids.append(note_id)

    topics = candidate.get(
        "topics",
        [],
    )

    if not topics:
        topics = detect_topics(
            issue.get("title", "")
            + "\n"
            + issue.get("text", "")
        )

    if not topics:
        topics = ["widgets"]

    turn_1 = {
        "turn_id": 1,
        "user": build_user_question(issue),
        "gold_evidence_ids": evidence_ids,
        "expected_behavior": build_expected_behavior(
            issue,
            selected_comments,
        ),
        "must_ask": build_must_ask(issue),
        "must_not_claim": build_must_not_claim(issue),
    }

    turns = [turn_1]

    if multi_turn:
        turn_2 = {
            "turn_id": 2,
            "user": build_followup_question(issue),
            "gold_evidence_ids": evidence_ids,
            "expected_behavior": [
                "Do not repeat checks or explanations already "
                "established in turn 1.",
                "Maintain the same issue context.",
                "Clearly distinguish known facts from missing "
                "information.",
                "Ask only discriminating questions that could "
                "change the diagnosis or next action.",
            ],
            "must_ask": build_must_ask(issue),
            "must_not_claim": build_must_not_claim(issue),
        }

        turns.append(turn_2)

    answer_points = [
        f"Explain the reported behavior described in issue "
        f"{get_issue_number(issue)}.",
        "Identify which claims are directly supported by "
        "the selected evidence.",
        "Identify important information that remains unknown.",
    ]

    if selected_comments:
        answer_points.append(
            "Incorporate the linked technical comment when it "
            "provides evidence relevant to the issue."
        )

    if related_docs:
        answer_points.append(
            "Use the official documentation as documentation "
            "evidence when it directly applies."
        )

    maintainer_summary = [
        "Problem summary based on the original issue.",
        "Evidence-backed findings.",
        "Known versus unknown technical details.",
        "Recommended next diagnostic information or action.",
    ]

    return {
        "case_id": f"{split}-{case_number:03d}",
        "family_id": candidate.get(
            "family_id",
            f"family-{get_issue_number(issue)}",
        ),
        "split": split,
        "multi_turn": multi_turn,
        "tags": sorted(set(topics)),
        "turns": turns,
        "final_expected": {
            "answer_points": answer_points,
            "evidence_requirements": [
                "Use at least one relevant issue record.",
                "Do not use evidence that is not present in corpus.",
                "Do not treat issue state as proof of resolution.",
                "Do not guess missing technical details.",
            ],
            "maintainer_summary_points": maintainer_summary,
        },
    }


# =========================================================
# Selection
# =========================================================

def candidate_quality(candidate, issue, comments):
    score = candidate.get(
        "score",
        0,
    )

    title = normalize_text(
        issue.get("title", "")
    )

    body = normalize_text(
        issue.get("text", "")
    )

    text = title + " " + body

    # Prefer real troubleshooting reports.
    positive_terms = [
        "bug",
        "error",
        "fails",
        "failure",
        "broken",
        "unexpected",
        "regression",
        "disappear",
        "lost",
        "drop",
        "wrong",
        "crash",
        "silently",
        "does not work",
        "not working",
        "issue",
    ]

    for term in positive_terms:
        if term in text:
            score += 5

    # Prefer issues with technical comments.
    score += min(
        len(comments) * 8,
        24,
    )

    # Prefer evidence-rich issues.
    if len(
        issue.get("text", "")
    ) > 300:
        score += 4

    return score


def deduplicate_by_family(candidates):
    result = {}

    for item in candidates:
        family_id = item.get(
            "family_id"
        )

        if not family_id:
            continue

        previous = result.get(
            family_id
        )

        if previous is None:
            result[family_id] = item
            continue

        if item["_quality"] > previous["_quality"]:
            result[family_id] = item

    return list(result.values())


def select_30_candidates(
    candidates,
    issues,
    comments_by_issue,
):
    enriched = []

    for candidate in candidates:
        issue_id = candidate.get(
            "primary_issue_id"
        )

        issue = None

        for possible_issue in issues.values():
            if possible_issue.get("id") == issue_id:
                issue = possible_issue
                break

        if issue is None:
            continue

        issue_number = get_issue_number(
            issue
        )

        comments = comments_by_issue.get(
            issue_number,
            [],
        )

        candidate["_quality"] = candidate_quality(
            candidate,
            issue,
            comments,
        )

        candidate["_issue"] = issue

        enriched.append(candidate)

    enriched = deduplicate_by_family(
        enriched
    )

    enriched.sort(
        key=lambda x: x["_quality"],
        reverse=True,
    )

    # We need at least 30 distinct families.
    if len(enriched) < TOTAL_CASES:
        raise RuntimeError(
            f"Only {len(enriched)} unique families are available. "
            f"{TOTAL_CASES} are required."
        )

    # Take a larger pool first so topic diversity can be applied.
    pool = enriched[:50]

    selected = []

    topic_counts = Counter()

    for candidate in pool:
        topics = candidate.get(
            "topics",
            [],
        )

        if not topics:
            continue

        rare_topic = min(
            topics,
            key=lambda t: topic_counts[t],
        )

        if topic_counts[rare_topic] < 7:
            selected.append(candidate)

            for topic in topics:
                topic_counts[topic] += 1

        if len(selected) >= TOTAL_CASES:
            break

    # Fill remaining places by quality.
    selected_ids = {
        item["family_id"]
        for item in selected
    }

    for candidate in enriched:
        if len(selected) >= TOTAL_CASES:
            break

        if candidate["family_id"] in selected_ids:
            continue

        selected.append(candidate)
        selected_ids.add(
            candidate["family_id"]
        )

    return selected[:TOTAL_CASES]


# =========================================================
# Split
# =========================================================

def split_candidates(candidates):
    rng = random.Random(
        RANDOM_SEED
    )

    shuffled = list(candidates)

    rng.shuffle(shuffled)

    # First make sure we have enough multi-turn cases.
    multi_turn_candidates = [
        candidate
        for candidate in shuffled
        if candidate.get(
            "multi_turn_recommended",
            False,
        )
    ]

    single_turn_candidates = [
        candidate
        for candidate in shuffled
        if not candidate.get(
            "multi_turn_recommended",
            False,
        )
    ]

    if len(multi_turn_candidates) < MIN_MULTI_TURN:
        raise RuntimeError(
            f"Only {len(multi_turn_candidates)} multi-turn candidates "
            f"are available. At least {MIN_MULTI_TURN} are required."
        )

    # We deliberately create 12 multi-turn cases:
    # 6 in DEV and 6 in TEST.
    multi_turn_candidates = multi_turn_candidates[:12]

    remaining = [
        candidate
        for candidate in shuffled
        if candidate not in multi_turn_candidates
    ]

    rng.shuffle(remaining)

    dev_multi = multi_turn_candidates[:6]
    test_multi = multi_turn_candidates[6:12]

    dev_needed = DEV_CASES - len(
        dev_multi
    )

    test_needed = TEST_CASES - len(
        test_multi
    )

    dev_single = remaining[:dev_needed]

    remaining = remaining[dev_needed:]

    test_single = remaining[:test_needed]

    dev = dev_multi + dev_single
    test = test_multi + test_single

    rng.shuffle(dev)
    rng.shuffle(test)

    if len(dev) != DEV_CASES:
        raise RuntimeError(
            f"DEV split has {len(dev)} cases."
        )

    if len(test) != TEST_CASES:
        raise RuntimeError(
            f"TEST split has {len(test)} cases."
        )

    return dev, test


# =========================================================
# Final validation
# =========================================================

def validate_cases(
    dev,
    test,
    corpus_by_id,
):
    all_cases = dev + test

    if len(all_cases) != TOTAL_CASES:
        raise RuntimeError(
            "Total case count is not 30."
        )

    if len(dev) != DEV_CASES:
        raise RuntimeError(
            "DEV must contain exactly 15 cases."
        )

    if len(test) != TEST_CASES:
        raise RuntimeError(
            "TEST must contain exactly 15 cases."
        )

    case_ids = [
        case["case_id"]
        for case in all_cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(
            "Duplicate case_id detected."
        )

    families = {}

    for case in dev:
        family_id = case["family_id"]

        if family_id in families:
            raise RuntimeError(
                f"Family {family_id} appears multiple times in DEV."
            )

        families[family_id] = "dev"

    for case in test:
        family_id = case["family_id"]

        if family_id in families:
            raise RuntimeError(
                f"Family {family_id} crosses DEV/TEST."
            )

        families[family_id] = "test"

    multi_turn_total = sum(
        1
        for case in all_cases
        if case["multi_turn"]
    )

    multi_turn_test = sum(
        1
        for case in test
        if case["multi_turn"]
    )

    if multi_turn_total < MIN_MULTI_TURN:
        raise RuntimeError(
            f"Only {multi_turn_total} multi-turn cases. "
            f"At least {MIN_MULTI_TURN} required."
        )

    if multi_turn_test < MIN_TEST_MULTI_TURN:
        raise RuntimeError(
            f"Only {multi_turn_test} multi-turn TEST cases. "
            f"At least {MIN_TEST_MULTI_TURN} required."
        )

    for case in all_cases:
        for turn in case["turns"]:
            evidence_ids = turn.get(
                "gold_evidence_ids",
                [],
            )

            if not evidence_ids:
                raise RuntimeError(
                    f"{case['case_id']} has no evidence."
                )

            for evidence_id in evidence_ids:
                if evidence_id not in corpus_by_id:
                    raise RuntimeError(
                        f"{case['case_id']} references "
                        f"unknown evidence ID: {evidence_id}"
                    )

    return {
        "total": len(all_cases),
        "dev": len(dev),
        "test": len(test),
        "multi_turn_total": multi_turn_total,
        "multi_turn_test": multi_turn_test,
        "families": len(families),
    }


# =========================================================
# Main
# =========================================================

def main():
    print()
    print("=" * 60)
    print("BUILDING FINAL EVALUATION DATASET")
    print("=" * 60)

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found:\n{CORPUS_PATH}"
        )

    if not SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Selected candidates not found:\n{SELECTED_PATH}"
        )

    corpus_data = load_json(
        CORPUS_PATH
    )

    selected_data = load_json(
        SELECTED_PATH
    )

    records = get_records(
        corpus_data
    )

    (
        corpus_by_id,
        issues,
        comments_by_issue,
        documents,
        release_notes,
    ) = build_corpus_index(
        records
    )

    candidates = get_selected_candidates(
        selected_data
    )

    print(
        f"Corpus records: {len(records)}"
    )

    print(
        f"Selected candidates: {len(candidates)}"
    )

    # -----------------------------------------------------
    # Select 30 distinct issue families.
    # -----------------------------------------------------

    selected = select_30_candidates(
        candidates,
        issues,
        comments_by_issue,
    )

    print(
        f"Final candidate families: {len(selected)}"
    )

    # -----------------------------------------------------
    # Split DEV / TEST.
    # -----------------------------------------------------

    dev_candidates, test_candidates = split_candidates(
        selected
    )

    print(
        f"DEV candidates: {len(dev_candidates)}"
    )

    print(
        f"TEST candidates: {len(test_candidates)}"
    )

    # -----------------------------------------------------
    # Build actual evaluation cases.
    # -----------------------------------------------------

    dev_cases = []
    test_cases = []

    for index, candidate in enumerate(
        dev_candidates,
        start=1,
    ):
        issue = candidate["_issue"]

        issue_number = get_issue_number(
            issue
        )

        comments = comments_by_issue.get(
            issue_number,
            [],
        )

        related_docs = find_related_documents(
            issue,
            documents,
            limit=2,
        )

        related_release_notes = (
            find_related_release_notes(
                issue,
                release_notes,
                limit=1,
            )
        )

        multi_turn = (
            candidate.get(
                "multi_turn_recommended",
                False,
            )
        )

        case = build_case(
            candidate=candidate,
            issue=issue,
            comments=comments,
            related_docs=related_docs,
            related_release_notes=related_release_notes,
            split="dev",
            multi_turn=multi_turn,
            case_number=index,
        )

        dev_cases.append(case)

    for index, candidate in enumerate(
        test_candidates,
        start=1,
    ):
        issue = candidate["_issue"]

        issue_number = get_issue_number(
            issue
        )

        comments = comments_by_issue.get(
            issue_number,
            [],
        )

        related_docs = find_related_documents(
            issue,
            documents,
            limit=2,
        )

        related_release_notes = (
            find_related_release_notes(
                issue,
                release_notes,
                limit=1,
            )
        )

        multi_turn = (
            candidate.get(
                "multi_turn_recommended",
                False,
            )
        )

        case = build_case(
            candidate=candidate,
            issue=issue,
            comments=comments,
            related_docs=related_docs,
            related_release_notes=related_release_notes,
            split="test",
            multi_turn=multi_turn,
            case_number=index,
        )

        test_cases.append(case)

    # -----------------------------------------------------
    # Remove internal fields before saving.
    # -----------------------------------------------------

    for candidate in selected:
        candidate.pop(
            "_issue",
            None,
        )

        candidate.pop(
            "_quality",
            None,
        )

    # -----------------------------------------------------
    # Validate.
    # -----------------------------------------------------

    stats = validate_cases(
        dev_cases,
        test_cases,
        corpus_by_id,
    )

    # -----------------------------------------------------
    # Save.
    # -----------------------------------------------------

    save_json(
        DEV_PATH,
        dev_cases,
    )

    save_json(
        TEST_PATH,
        test_cases,
    )

    # -----------------------------------------------------
    # Statistics.
    # -----------------------------------------------------

    topic_counts = Counter()

    for case in dev_cases + test_cases:
        for tag in case.get(
            "tags",
            [],
        ):
            topic_counts[tag] += 1

    dev_multi = sum(
        1
        for case in dev_cases
        if case["multi_turn"]
    )

    test_multi = sum(
        1
        for case in test_cases
        if case["multi_turn"]
    )

    print()
    print("=" * 60)
    print("FINAL EVALUATION DATASET CREATED")
    print("=" * 60)

    print(
        f"Total cases: {stats['total']}"
    )

    print(
        f"DEV cases: {stats['dev']}"
    )

    print(
        f"TEST cases: {stats['test']}"
    )

    print(
        f"Multi-turn total: {stats['multi_turn_total']}"
    )

    print(
        f"Multi-turn DEV: {dev_multi}"
    )

    print(
        f"Multi-turn TEST: {test_multi}"
    )

    print()
    print("Topic counts:")

    for topic, count in sorted(
        topic_counts.items()
    ):
        print(
            f"  {topic}: {count}"
        )

    print()
    print("Files:")

    print(
        f"  DEV : {DEV_PATH}"
    )

    print(
        f"  TEST: {TEST_PATH}"
    )

    print()
    print("Validation: PASSED")

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
