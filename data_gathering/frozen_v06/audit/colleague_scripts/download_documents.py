"""
Download a reproducible, domain-focused snapshot of Streamlit documentation.

This script:
- Uses the fixed DOCS_COMMIT from .env.
- Reads the Streamlit docs repository through GitHub Contents API.
- Accepts both directories and files.
- Searches the selected documentation areas for Markdown files relevant
  to runtime state, widgets, reruns, callbacks, forms, caching and multipage apps.
- Preserves title, text, URL, section and source_version.
- Does not use current/main documentation as the canonical source.

It intentionally does not require a GitHub token for public repository content,
but uses GITHUB_TOKEN when available to improve API limits.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

OWNER = os.getenv("GITHUB_OWNER", "streamlit")
DOCS_REPO = "docs"
COMMIT = os.getenv(
    "DOCS_COMMIT",
    "695e6ce2fb1b23afc6df66b40420ab0013ad72fd",
)
TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "streamlit-support-dataset-builder/2.0",
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

# These are intentionally file/directory candidates, not assumptions that
# every entry is a directory at every historical commit.
ROOT_CANDIDATES = [
    "content/develop/concepts/architecture",
    "content/develop/concepts/multipage-apps",
    "content/develop/api-reference",
    "content/get-started/fundamentals",
]

KEYWORDS = {
    "session_state",
    "session-state",
    "caching",
    "cache",
    "widget",
    "widgets",
    "rerun",
    "callback",
    "callbacks",
    "form",
    "forms",
    "multipage",
    "state",
    "execution-flow",
}

MAX_FILES = 60


def github_get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    for attempt in range(5):
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=60,
        )

        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if remaining == "0" and reset:
                wait = max(1, int(reset) - int(time.time()) + 1)
                print(f"Rate limit reached. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if attempt < 4:
                wait = 2 ** attempt
                print(
                    f"HTTP {response.status_code}. Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"GitHub request failed after retries: {url}")


def contents(path: str) -> list[dict[str, Any]] | dict[str, Any]:
    url = f"{API}/repos/{OWNER}/{DOCS_REPO}/contents/{path}"
    response = github_get(url, {"ref": COMMIT})
    return response.json()


def raw_file(path: str) -> str:
    url = f"{API}/repos/{OWNER}/{DOCS_REPO}/contents/{path}"
    response = github_get(url, {"ref": COMMIT})
    item = response.json()

    if item.get("type") != "file":
        raise RuntimeError(f"Expected file, got {item.get('type')}: {path}")

    data = base64.b64decode(item["content"])
    return data.decode("utf-8", errors="replace")


def walk(path: str, found: list[str]) -> None:
    if len(found) >= MAX_FILES:
        return

    try:
        item = contents(path)
    except requests.HTTPError as exc:
        print(f"  Missing/unavailable: {path} ({exc.response.status_code})")
        return

    if isinstance(item, dict):
        if item.get("type") == "file" and path.endswith(".md"):
            found.append(path)
        return

    for entry in item:
        entry_path = entry.get("path", "")
        entry_type = entry.get("type")

        if entry_type == "file" and entry_path.endswith(".md"):
            lower = entry_path.lower()
            if any(k in lower for k in KEYWORDS):
                found.append(entry_path)

        elif entry_type == "dir":
            # Limit recursion to directories whose names can plausibly
            # contain our domain.
            lower = entry_path.lower()
            if (
                any(k in lower for k in KEYWORDS)
                or entry_path.count("/") <= 5
            ):
                walk(entry_path, found)

        if len(found) >= MAX_FILES:
            break


def title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else fallback


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_document(path: str, text: str) -> dict[str, Any]:
    cleaned = clean_markdown(text)
    name = Path(path).stem.replace("-", " ").replace("_", " ")
    title = title_from_markdown(cleaned, name.title())

    return {
        "id": f"doc-{COMMIT[:12]}-{abs(hash(path))}",
        "source_type": "streamlit_docs",
        "content_type": "document",
        "parent_id": None,
        "title": title,
        "text": cleaned,
        "url": f"https://github.com/{OWNER}/{DOCS_REPO}/blob/{COMMIT}/{path}",
        "metadata": {
            "repository": f"{OWNER}/{DOCS_REPO}",
            "path": path,
            "section": path,
            "source_version": COMMIT,
        },
    }


def main() -> None:
    print("Downloading Streamlit documentation...")
    print(f"Repository: {OWNER}/{DOCS_REPO}")
    print(f"Fixed commit: {COMMIT}")
    print("Searching domain-relevant Markdown files...\n")

    candidates: list[str] = []

    for index, root in enumerate(ROOT_CANDIDATES, start=1):
        print(
            f"[{index}/{len(ROOT_CANDIDATES)}] "
            f"Scanning: {root}"
        )
        walk(root, candidates)

    # Deduplicate while preserving order.
    candidates = list(dict.fromkeys(candidates))

    # Explicitly prioritize the most important domain files if they exist.
    priority = [
        "session_state.md",
        "widget-behavior.md",
        "caching.md",
        "execution-flow.md",
        "forms.md",
        "multipage",
    ]

    candidates.sort(
        key=lambda p: (
            0 if any(k in p.lower() for k in priority) else 1,
            p.lower(),
        )
    )

    print(f"\nCandidate Markdown files: {len(candidates)}")

    documents: list[dict[str, Any]] = []
    missing_read: list[str] = []

    for index, path in enumerate(candidates, start=1):
        try:
            text = raw_file(path)
            documents.append(build_document(path, text))
            print(
                f"[{index}/{len(candidates)}] "
                f"saved: {path}"
            )
        except Exception as exc:
            missing_read.append(f"{path}: {exc}")
            print(f"[{index}/{len(candidates)}] failed: {path} -> {exc}")

    output = RAW_DIR / "documents.json"
    output.write_text(
        __import__("json").dumps(
            {
                "dataset_version": os.getenv("DATASET_VERSION", ""),
                "source_version": COMMIT,
                "repository": f"{OWNER}/{DOCS_REPO}",
                "documents": documents,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Saved {len(documents)} documents.")
    print(f"Output: {output}")
    print(f"Fixed commit: {COMMIT}")

    if missing_read:
        print("\nFiles that could not be read:")
        for item in missing_read:
            print(f"- {item}")


if __name__ == "__main__":
    main()
