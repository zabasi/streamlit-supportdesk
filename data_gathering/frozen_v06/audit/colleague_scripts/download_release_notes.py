from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

DOCS_OWNER = "streamlit"
DOCS_REPO = "docs"

COMMIT = os.getenv(
    "DOCS_COMMIT",
    "695e6ce2fb1b23afc6df66b40420ab0013ad72fd",
).strip()

OUTPUT = ROOT / "data" / "raw" / "release_notes.json"

API = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "streamlit-support-dataset-builder",
}

if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


# ============================================================
# Domain terms
# ============================================================

DOMAIN_TERMS = [
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
]


# ============================================================
# HTTP
# ============================================================

MAX_RETRIES = 5
TIMEOUT = 60


def github_get(
    url: str,
    params: dict[str, Any] | None = None,
) -> requests.Response:

    for attempt in range(1, MAX_RETRIES + 1):

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=TIMEOUT,
        )

        # Rate limit
        if response.status_code in (403, 429):

            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")

            if remaining == "0" and reset:

                wait_seconds = max(
                    1,
                    int(reset) - int(time.time()) + 1,
                )

            else:

                wait_seconds = min(
                    2 ** attempt,
                    30,
                )

            print(
                f"GitHub rate limit ({response.status_code}). "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        return response

    raise RuntimeError(
        f"GitHub request failed after {MAX_RETRIES} retries:\n{url}"
    )


# ============================================================
# Git Tree
# ============================================================

def get_git_tree() -> list[dict[str, Any]]:

    print()
    print("Reading Git tree...")
    print(f"Repository: {DOCS_OWNER}/{DOCS_REPO}")
    print(f"Commit: {COMMIT}")

    url = (
        f"{API}/repos/"
        f"{DOCS_OWNER}/{DOCS_REPO}/git/trees/{COMMIT}"
    )

    response = github_get(
        url,
        {
            "recursive": "1",
        },
    )

    data = response.json()

    if data.get("truncated"):
        raise RuntimeError(
            "GitHub returned a truncated Git tree. "
            "The repository tree is too large for this request."
        )

    tree = data.get("tree", [])

    print(f"Tree entries found: {len(tree)}")

    return tree


# ============================================================
# Find release-note files
# ============================================================

def find_release_note_files(
    tree: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    candidates = []

    for item in tree:

        if item.get("type") != "blob":
            continue

        path = item.get("path", "")

        lower = path.lower()

        if not lower.endswith((".md", ".mdx")):
            continue

        # We are interested in Streamlit's release-note/changelog
        # documentation, not arbitrary files mentioning "release".
        is_release_path = (
            "release-notes" in lower
            or "release_notes" in lower
            or "changelog" in lower
            or "/releases/" in lower
        )

        if is_release_path:
            candidates.append(item)

    return candidates


# ============================================================
# Download blob
# ============================================================

def get_blob_text(sha: str) -> str:

    url = (
        f"{API}/repos/"
        f"{DOCS_OWNER}/{DOCS_REPO}/git/blobs/{sha}"
    )

    response = github_get(url)

    data = response.json()

    encoding = data.get("encoding")
    content = data.get("content", "")

    if encoding == "base64":

        content = content.replace("\n", "")

        return base64.b64decode(content).decode(
            "utf-8",
            errors="replace",
        )

    raise RuntimeError(
        f"Unsupported Git blob encoding: {encoding}"
    )


# ============================================================
# Front matter
# ============================================================

def parse_front_matter(
    text: str,
) -> tuple[dict[str, str], str]:

    if not text.startswith("---"):

        return {}, text

    parts = text.split("---", 2)

    if len(parts) != 3:

        return {}, text

    front_text = parts[1]
    body = parts[2].strip()

    metadata: dict[str, str] = {}

    for line in front_text.splitlines():

        match = re.match(
            r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*$",
            line,
        )

        if not match:
            continue

        key, value = match.groups()

        metadata[key] = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

    return metadata, body


# ============================================================
# Title
# ============================================================

def extract_title(
    path: str,
    front_matter: dict[str, str],
    body: str,
) -> str:

    title = (
        front_matter.get("title")
        or front_matter.get("name")
    )

    if title:

        return title

    for line in body.splitlines():

        stripped = line.strip()

        if stripped.startswith("#"):

            return stripped.lstrip("#").strip()

    filename = Path(path).stem

    return (
        filename
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )


# ============================================================
# Version / date
# ============================================================

def extract_version_and_date(
    path: str,
    front_matter: dict[str, str],
    body: str,
) -> tuple[str | None, str | None]:

    version = (
        front_matter.get("version")
        or front_matter.get("release")
        or front_matter.get("release_version")
    )

    release_date = (
        front_matter.get("date")
        or front_matter.get("release_date")
        or front_matter.get("published")
    )

    # Example:
    # 2025.md
    # 2024.md
    year_match = re.search(
        r"(20\d{2})",
        Path(path).name,
    )

    if not release_date and year_match:

        release_date = year_match.group(1)

    # Try semantic version numbers.
    if not version:

        version_match = re.search(
            r"\b(?:v)?(\d+\.\d+\.\d+)\b",
            body,
        )

        if version_match:

            version = version_match.group(1)

    return version, release_date


# ============================================================
# Domain relevance
# ============================================================

def is_domain_relevant(text: str) -> bool:

    lower = text.lower()

    return any(
        term.lower() in lower
        for term in DOMAIN_TERMS
    )


# ============================================================
# URL
# ============================================================

def build_docs_url(path: str) -> str:

    # Repository paths usually look like:
    #
    # content/develop/quick-references/release-notes/2025.md
    #
    # Convert to:
    #
    # https://docs.streamlit.io/develop/quick-references/...

    if path.startswith("content/"):

        path = path[len("content/"):]

    if path.endswith(".md"):

        path = path[:-3]

    if path.endswith("/_index"):

        path = path[:-6]

    return f"https://docs.streamlit.io/{path}"


# ============================================================
# Build record
# ============================================================

def build_record(
    item: dict[str, Any],
) -> dict[str, Any] | None:

    path = item["path"]
    sha = item["sha"]

    body_raw = get_blob_text(sha)

    front_matter, body = parse_front_matter(body_raw)

    if not is_domain_relevant(body):

        return None

    title = extract_title(
        path,
        front_matter,
        body,
    )

    version, release_date = extract_version_and_date(
        path,
        front_matter,
        body,
    )

    return {
        "id": f"release-note-{sha}",

        "source_type": "streamlit_release_note",

        "content_type": "release_note",

        "parent_id": None,

        "title": title,

        "text": body,

        "url": build_docs_url(path),

        "metadata": {
            "repository": f"{DOCS_OWNER}/{DOCS_REPO}",

            "path": path,

            "git_blob_sha": sha,

            "section": "release-notes",

            "release_version": version,

            "release_date": release_date,

            "source_version": COMMIT,
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("Downloading Streamlit release notes...")
    print(
        f"Repository: {DOCS_OWNER}/{DOCS_REPO}"
    )
    print(
        f"Fixed commit: {COMMIT}"
    )
    print(
        "Source: GitHub Git Tree + Blob API"
    )

    if not TOKEN:

        print()
        print(
            "WARNING: GITHUB_TOKEN is not set."
        )

        print(
            "GitHub public API rate limits may apply."
        )

    # --------------------------------------------------------
    # 1. Get complete Git tree
    # --------------------------------------------------------

    tree = get_git_tree()

    # --------------------------------------------------------
    # 2. Find release-note/changelog files
    # --------------------------------------------------------

    files = find_release_note_files(tree)

    print()
    print(
        f"Release-note/changelog candidates: {len(files)}"
    )

    if not files:

        print()
        print(
            "No release-note/changelog files were found "
            f"in commit {COMMIT}."
        )

        print()
        print(
            "This means the fixed commit does not contain "
            "Streamlit release-note files under a path containing "
            "'release-notes', 'release_notes', or 'changelog'."
        )

        print()
        print(
            "No fallback to main/master was performed."
        )

        raise RuntimeError(
            "No release-note/changelog files found "
            "in the fixed commit."
        )

    # --------------------------------------------------------
    # 3. Download and filter
    # --------------------------------------------------------

    records: list[dict[str, Any]] = []

    for index, item in enumerate(files, start=1):

        path = item["path"]

        print(
            f"[{index}/{len(files)}] {path}"
        )

        try:

            record = build_record(item)

            if record is not None:

                records.append(record)

                print(
                    "    -> domain relevant"
                )

            else:

                print(
                    "    -> skipped"
                )

        except Exception as exc:

            print(
                f"    -> WARNING: {exc}"
            )

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)

    print(
        f"Release-note files found: {len(files)}"
    )

    print(
        f"Domain-relevant records saved: {len(records)}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    print(
        f"Fixed commit: {COMMIT}"
    )

    print()
    print(
        "Reproducibility: OK"
    )

    print(
        "The downloader did not use main/master or the current docs snapshot."
    )


if __name__ == "__main__":
    main()
