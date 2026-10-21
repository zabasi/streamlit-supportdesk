from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

ISSUES_FILE = RAW_DIR / "issues.json"
COMMENTS_FILE = RAW_DIR / "comments.json"
DOCUMENTS_FILE = RAW_DIR / "documents.json"
RELEASE_NOTES_FILE = RAW_DIR / "release_notes.json"

CORPUS_FILE = CORPUS_DIR / "corpus.json"
MANIFEST_FILE = CORPUS_DIR / "manifest.json"

DATASET_VERSION = "runtime_state_widgets-v01"

DOCS_COMMIT = "695e6ce2fb1b23afc6df66b40420ab0013ad72fd"


# =========================================================
# Basic helpers
# =========================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_list(
    data: Any,
    name: str,
) -> list[dict[str, Any]]:

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError(
            f"{name}.json must contain a JSON list or "
            f"a wrapper object, but got "
            f"{type(data).__name__}"
        )

    preferred_keys = [
        "records",
        "issues",
        "comments",
        "documents",
        "release_notes",
        "data",
        "items",
        "results",
    ]

    for key in preferred_keys:
        value = data.get(key)

        if isinstance(value, list):
            print(
                f"Detected wrapped JSON data "
                f"under key: '{key}'"
            )
            return value

    # Generic fallback
    for key, value in data.items():
        if isinstance(value, list):
            if all(
                isinstance(item, dict)
                for item in value
            ):
                print(
                    f"Detected wrapped JSON data "
                    f"under key: '{key}'"
                )
                return value

    raise ValueError(
        f"Could not find a list of records inside "
        f"{name}.json.\n"
        f"Available keys: {list(data.keys())}"
    )


def first_value(
    record: dict[str, Any],
    *keys: str,
) -> Any:

    for key in keys:
        if key in record:
            value = record[key]

            if value is not None:
                return value

    return None


def as_string(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


def normalize_text(value: Any) -> str:

    if value is None:
        return ""

    text = as_string(value)

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    result = []

    previous_blank = False

    for line in lines:

        blank = not line.strip()

        if blank and previous_blank:
            continue

        result.append(line)

        previous_blank = blank

    return "\n".join(result).strip()


def make_content_hash(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# Issue mapping
# =========================================================

def build_issue_mapping(
    issues: list[dict[str, Any]],
) -> dict[str, str]:

    """
    Build:

        issue_number -> normalized corpus ID

    Example:

        2911 -> issue-2911
    """

    mapping: dict[str, str] = {}

    for index, issue in enumerate(issues):

        issue_number = first_value(
            issue,
            "issue_number",
            "number",
        )

        if issue_number is None:
            print(
                f"WARNING: Issue at index {index} "
                f"has no issue_number."
            )
            continue

        issue_number_str = as_string(
            issue_number
        )

        issue_id = (
            f"issue-{issue_number_str}"
        )

        if issue_number_str in mapping:

            raise ValueError(
                "Duplicate issue_number found: "
                f"{issue_number_str}"
            )

        mapping[
            issue_number_str
        ] = issue_id

    print(
        f"\nIssue mapping created: "
        f"{len(mapping)} issues"
    )

    return mapping


# =========================================================
# Normalize records
# =========================================================

def normalize_record(
    record: dict[str, Any],
    source_type: str,
    index: int,
    issue_mapping: dict[str, str],
) -> dict[str, Any]:

    if not isinstance(record, dict):

        raise ValueError(
            f"{source_type}[{index}] "
            f"is not an object."
        )

    # -----------------------------------------------------
    # ID
    # -----------------------------------------------------

    if source_type == "issue":

        raw_id = first_value(
            record,
            "issue_number",
            "number",
            "id",
        )

        if raw_id is None:
            raw_id = index + 1

        record_id = (
            f"issue-{as_string(raw_id)}"
        )

    elif source_type == "comment":

        raw_id = first_value(
            record,
            "comment_id",
            "id",
        )

        if raw_id is None:
            raw_id = index + 1

        record_id = (
            f"comment-{as_string(raw_id)}"
        )

    elif source_type == "document":

        raw_id = first_value(
            record,
            "document_id",
            "id",
            "path",
        )

        if raw_id is None:
            raw_id = index + 1

        raw_id = as_string(raw_id)

        # Avoid problematic characters in IDs.
        raw_id = raw_id.replace(
            "\\",
            "/",
        )

        record_id = (
            f"document-{raw_id}"
            if not raw_id.startswith(
                "document-"
            )
            else raw_id
        )

    elif source_type == "release_note":

        raw_id = first_value(
            record,
            "release_note_id",
            "id",
            "path",
            "version",
            "year",
        )

        if raw_id is None:
            raw_id = index + 1

        raw_id = as_string(raw_id)

        record_id = (
            f"release-note-{raw_id}"
            if not raw_id.startswith(
                "release-note-"
            )
            else raw_id
        )

    else:

        raw_id = first_value(
            record,
            "id",
        )

        if raw_id is None:
            raw_id = index + 1

        record_id = as_string(raw_id)

    # -----------------------------------------------------
    # Parent relationship
    # -----------------------------------------------------

    parent_id = None

    if source_type == "comment":

        # This is the real relationship in the raw data:
        #
        # comments.issue_number
        #
        # ->
        #
        # issues.issue_number

        issue_number = record.get(
            "issue_number"
        )

        if issue_number is not None:

            issue_number_str = as_string(
                issue_number
            )

            parent_id = issue_mapping.get(
                issue_number_str
            )

    else:

        raw_parent = first_value(
            record,
            "parent_id",
            "issue_id",
            "parent",
        )

        if raw_parent is not None:

            raw_parent = as_string(
                raw_parent
            )

            parent_id = raw_parent

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    title = normalize_text(
        first_value(
            record,
            "title",
            "name",
            "heading",
        )
    )

    # -----------------------------------------------------
    # Text
    # -----------------------------------------------------

    text = normalize_text(
        first_value(
            record,
            "text",
            "body",
            "content",
            "description",
            "markdown",
        )
    )

    # -----------------------------------------------------
    # URL
    # -----------------------------------------------------

    url = as_string(
        first_value(
            record,
            "url",
            "html_url",
            "issue_url",
            "source_url",
        )
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata: dict[str, Any] = {}

    useful_keys = [
        "issue_number",
        "comment_id",
        "github_id",
        "state",
        "state_reason",
        "created_at",
        "updated_at",
        "closed_at",
        "author_association",
        "user",
        "author",
        "labels",
        "comments_count",
        "comments_url",
        "version",
        "date",
        "section",
        "path",
        "repository",
        "commit",
        "fixed_commit",
        "source_document_id",
        "domain",
        "domain_matches",
    ]

    for key in useful_keys:

        if key in record:

            metadata[key] = record[key]

    # -----------------------------------------------------
    # Nested metadata
    # -----------------------------------------------------

    nested_metadata = record.get(
        "metadata"
    )

    if isinstance(
        nested_metadata,
        dict,
    ):

        for key, value in nested_metadata.items():

            if key not in metadata:
                metadata[key] = value

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    if "labels" in record:

        labels = record["labels"]

        normalized_labels = []

        if isinstance(
            labels,
            list,
        ):

            for label in labels:

                if isinstance(
                    label,
                    dict,
                ):

                    name = label.get(
                        "name"
                    )

                    if name:
                        normalized_labels.append(
                            name
                        )

                elif isinstance(
                    label,
                    str,
                ):

                    normalized_labels.append(
                        label
                    )

        metadata["labels"] = (
            normalized_labels
        )

    # -----------------------------------------------------
    # Author
    # -----------------------------------------------------

    if isinstance(
        record.get("user"),
        dict,
    ):

        user = record["user"]

        if "login" in user:

            metadata["author_login"] = (
                user["login"]
            )

    if isinstance(
        record.get("author"),
        dict,
    ):

        author = record["author"]

        if "login" in author:

            metadata["author_login"] = (
                author["login"]
            )

    # -----------------------------------------------------
    # Source-specific metadata
    # -----------------------------------------------------

    if source_type == "document":

        metadata.setdefault(
            "repository",
            "streamlit/docs",
        )

        metadata.setdefault(
            "commit",
            DOCS_COMMIT,
        )

    # -----------------------------------------------------
    # Content type
    # -----------------------------------------------------

    content_type_map = {

        "issue": "issue",

        "comment": "issue_comment",

        "document": "documentation",

        "release_note": "release_note",
    }

    content_type = content_type_map[
        source_type
    ]

    # -----------------------------------------------------
    # Retrieval text
    # -----------------------------------------------------

    if title and text:

        combined_text = (
            f"{title}\n\n{text}"
        )

    elif title:

        combined_text = title

    else:

        combined_text = text

    combined_text = normalize_text(
        combined_text
    )

    # -----------------------------------------------------
    # Content hash
    # -----------------------------------------------------

    content_hash = make_content_hash(
        combined_text
    )

    return {

        "id": record_id,

        "source_type": source_type,

        "content_type": content_type,

        "parent_id": parent_id,

        "title": title,

        "text": text,

        "url": url,

        "metadata": metadata,

        "content_hash": content_hash,
    }


# =========================================================
# Source loader
# =========================================================

def load_source(
    path: Path,
    name: str,
    source_type: str,
    issue_mapping: dict[str, str],
) -> list[dict[str, Any]]:

    print(
        f"\nLoading {name}..."
    )

    raw = load_json(path)

    records = ensure_list(
        raw,
        path.stem,
    )

    normalized = []

    for index, record in enumerate(
        records
    ):

        normalized_record = (
            normalize_record(
                record,
                source_type,
                index,
                issue_mapping,
            )
        )

        normalized.append(
            normalized_record
        )

    print(
        f"{name}: "
        f"{len(normalized)} records loaded."
    )

    return normalized


# =========================================================
# Validation
# =========================================================

def validate_unique_ids(
    records: list[dict[str, Any]],
) -> None:

    counter = Counter(
        record["id"]
        for record in records
    )

    duplicates = {
        record_id: count
        for record_id, count
        in counter.items()
        if count > 1
    }

    if duplicates:

        print(
            "\nERROR: Duplicate IDs found:"
        )

        for record_id, count in (
            duplicates.items()
        ):

            print(
                f"  {record_id}: {count}"
            )

        raise ValueError(
            f"Found {len(duplicates)} "
            f"duplicate IDs."
        )

    print(
        "ID uniqueness: OK"
    )


def validate_required_fields(
    records: list[dict[str, Any]],
) -> None:

    required_fields = [
        "id",
        "source_type",
        "content_type",
        "title",
        "text",
        "url",
        "metadata",
        "content_hash",
    ]

    problems = []

    for index, record in enumerate(
        records
    ):

        for field in required_fields:

            if field not in record:

                problems.append(
                    f"Record {index} "
                    f"missing '{field}'"
                )

    if problems:

        for problem in problems[:20]:

            print(problem)

        raise ValueError(
            f"Found {len(problems)} "
            f"required-field problems."
        )

    print(
        "Required fields: OK"
    )


def validate_parent_ids(
    records: list[dict[str, Any]],
) -> None:

    ids = {
        record["id"]
        for record in records
    }

    comments = [
        record
        for record in records
        if record["source_type"]
        == "comment"
    ]

    comments_with_parent = 0

    missing = []

    for comment in comments:

        parent_id = comment.get(
            "parent_id"
        )

        if parent_id is None:
            continue

        comments_with_parent += 1

        if parent_id not in ids:

            missing.append(
                (
                    comment["id"],
                    parent_id,
                )
            )

    print(
        f"\nComment relationships:"
    )

    print(
        f"  Total comments: "
        f"{len(comments)}"
    )

    print(
        f"  Comments with parent: "
        f"{comments_with_parent}"
    )

    print(
        f"  Comments without parent: "
        f"{len(comments) - comments_with_parent}"
    )

    print(
        f"  Missing parent records: "
        f"{len(missing)}"
    )

    if missing:

        print(
            "\nWARNING: Some comment parents "
            "are not present in the selected "
            "issue dataset."
        )

        for comment_id, parent_id in (
            missing[:20]
        ):

            print(
                f"  {comment_id} "
                f"-> {parent_id}"
            )

        if len(missing) > 20:

            print(
                f"  ... and "
                f"{len(missing) - 20} more"
            )

    else:

        print(
            "Parent relationships: OK"
        )


# =========================================================
# Statistics
# =========================================================

def print_statistics(
    records: list[dict[str, Any]],
) -> None:

    source_counter = Counter(
        record["source_type"]
        for record in records
    )

    content_counter = Counter(
        record["content_type"]
        for record in records
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CORPUS STATISTICS"
    )

    print(
        "=" * 60
    )

    print(
        "\nSource counts:"
    )

    source_order = [
        "issue",
        "comment",
        "document",
        "release_note",
    ]

    for source in source_order:

        print(
            f"  {source:15} "
            f"{source_counter.get(source, 0)}"
        )

    print(
        f"  {'TOTAL':15} "
        f"{len(records)}"
    )

    print(
        "\nContent types:"
    )

    for content_type, count in sorted(
        content_counter.items()
    ):

        print(
            f"  {content_type:20} "
            f"{count}"
        )

    parent_count = sum(
        1
        for record in records
        if record.get("parent_id")
        is not None
    )

    print(
        "\nRelationships:"
    )

    print(
        f"  Records with parent_id: "
        f"{parent_count}"
    )

    print(
        f"  Records without parent_id: "
        f"{len(records) - parent_count}"
    )

    text_count = sum(
        1
        for record in records
        if record.get("text")
    )

    title_count = sum(
        1
        for record in records
        if record.get("title")
    )

    print(
        "\nContent quality:"
    )

    print(
        f"  Records with title: "
        f"{title_count}"
    )

    print(
        f"  Records with text: "
        f"{text_count}"
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print(
        "=" * 60
    )

    print(
        "BUILD STREAMLIT SUPPORT CORPUS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDataset version: "
        f"{DATASET_VERSION}"
    )

    print(
        f"Documentation commit: "
        f"{DOCS_COMMIT}"
    )

    # -----------------------------------------------------
    # Load raw issues first
    # -----------------------------------------------------

    print(
        "\nPreparing issue mapping..."
    )

    raw_issues = load_json(
        ISSUES_FILE
    )

    issue_records = ensure_list(
        raw_issues,
        "issues",
    )

    issue_mapping = build_issue_mapping(
        issue_records
    )

    # -----------------------------------------------------
    # Load and normalize sources
    # -----------------------------------------------------

    issues = load_source(
        ISSUES_FILE,
        "Issues",
        "issue",
        issue_mapping,
    )

    comments = load_source(
        COMMENTS_FILE,
        "Comments",
        "comment",
        issue_mapping,
    )

    documents = load_source(
        DOCUMENTS_FILE,
        "Documents",
        "document",
        issue_mapping,
    )

    release_notes = load_source(
        RELEASE_NOTES_FILE,
        "Release Notes",
        "release_note",
        issue_mapping,
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    records = (
        issues
        + comments
        + documents
        + release_notes
    )

    print(
        f"\nTotal normalized records: "
        f"{len(records)}"
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    print(
        "\nValidating corpus..."
    )

    validate_unique_ids(
        records
    )

    validate_required_fields(
        records
    )

    validate_parent_ids(
        records
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print_statistics(
        records
    )

    # -----------------------------------------------------
    # Corpus object
    # -----------------------------------------------------

    corpus = {

        "schema_version": "1.0",

        "corpus_version":
            DATASET_VERSION,

        "generated_at":
            now_iso(),

        "description": (
            "Streamlit technical support "
            "corpus for the "
            "runtime_state_widgets domain."
        ),

        "domain": {

            "name":
                "runtime_state_widgets",

            "terms": [

                "session_state",
                "session state",
                "st.session_state",
                "rerun",
                "st.rerun",
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
                "form",
                "forms",
                "callback",
                "on_click",
                "on_change",
                "cache_data",
                "cache_resource",
                "caching",
                "multipage",
                "pages",
                "navigation",
            ],
        },

        "sources": {

            "github_repository":
                "streamlit/streamlit",

            "documentation_repository":
                "streamlit/docs",

            "documentation_commit":
                DOCS_COMMIT,
        },

        "records":
            records,
    }

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    source_counter = Counter(
        record["source_type"]
        for record in records
    )

    content_counter = Counter(
        record["content_type"]
        for record in records
    )

    comments_with_parent = sum(
        1
        for record in records
        if (
            record["source_type"]
            == "comment"
            and record.get("parent_id")
            is not None
        )
    )

    manifest = {

        "schema_version": "1.0",

        "corpus_version":
            DATASET_VERSION,

        "generated_at":
            now_iso(),

        "reproducibility": {

            "github_repository":
                "streamlit/streamlit",

            "documentation_repository":
                "streamlit/docs",

            "documentation_commit":
                DOCS_COMMIT,

            "dataset_version":
                DATASET_VERSION,
        },

        "source_counts": {

            "issues":
                source_counter.get(
                    "issue",
                    0,
                ),

            "comments":
                source_counter.get(
                    "comment",
                    0,
                ),

            "documents":
                source_counter.get(
                    "document",
                    0,
                ),

            "release_notes":
                source_counter.get(
                    "release_note",
                    0,
                ),

            "total":
                len(records),
        },

        "content_type_counts":
            dict(content_counter),

        "relationship_statistics": {

            "comments_with_parent":
                comments_with_parent,

            "comments_without_parent":
                source_counter.get(
                    "comment",
                    0,
                ) - comments_with_parent,

            "records_with_parent_id":
                sum(
                    1
                    for record in records
                    if record.get(
                        "parent_id"
                    ) is not None
                ),

            "records_without_parent_id":
                sum(
                    1
                    for record in records
                    if record.get(
                        "parent_id"
                    ) is None
                ),
        },

        "files": {

            "issues":
                str(
                    ISSUES_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "comments":
                str(
                    COMMENTS_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "documents":
                str(
                    DOCUMENTS_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "release_notes":
                str(
                    RELEASE_NOTES_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
        },
    }

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    CORPUS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Write corpus
    # -----------------------------------------------------

    print(
        f"\nWriting corpus:\n"
        f"  {CORPUS_FILE}"
    )

    with CORPUS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            corpus,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Write manifest
    # -----------------------------------------------------

    print(
        f"Writing manifest:\n"
        f"  {MANIFEST_FILE}"
    )

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "CORPUS BUILD COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"\nIssues:        "
        f"{source_counter.get('issue', 0)}"
    )

    print(
        f"Comments:      "
        f"{source_counter.get('comment', 0)}"
    )

    print(
        f"Documents:     "
        f"{source_counter.get('document', 0)}"
    )

    print(
        f"Release notes: "
        f"{source_counter.get('release_note', 0)}"
    )

    print(
        f"Total:         "
        f"{len(records)}"
    )

    print(
        f"\nComments with parent_id: "
        f"{comments_with_parent}"
    )

    print(
        "\nOutput files:"
    )

    print(
        f"  {CORPUS_FILE}"
    )

    print(
        f"  {MANIFEST_FILE}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()