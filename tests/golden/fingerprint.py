"""Everything that defines the baseline, as short hashes: the code it runs, its profile and prompt
files, and the values of the settings it uses. tests/golden/baseline_fingerprint.json records them;
the golden test fails when any of them changes, until the golden files are regenerated on purpose.

Shared files are hashed per function or class, so adding something new to them (another retriever,
another setting) does not count as a change to the baseline; changing what the baseline runs does.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

from supportdesk.config import Settings

ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = Path(__file__).with_name("baseline_fingerprint.json")

#: Files the baseline runs as a whole.
WHOLE_FILES = (
    "configs/pipelines/baseline.toml",
    "configs/prompts/answer_system.txt",
    "configs/prompts/answer_output_format.txt",
    "src/supportdesk/pipelines/baseline.py",
    "src/supportdesk/pipelines/staged.py",
    "src/supportdesk/pipelines/state.py",
    "src/supportdesk/pipelines/packet.py",
    "src/supportdesk/pipelines/parsing.py",
    "src/supportdesk/pipelines/prompts.py",
    "src/supportdesk/pipelines/summary.py",
    "src/supportdesk/pipelines/stages/defaults.py",
    "src/supportdesk/pipelines/stages/protocols.py",
    "src/supportdesk/tokenizer.py",
    "src/supportdesk/data/corpus.py",
    "src/supportdesk/data/environment.py",
    "src/supportdesk/indexing.py",
)

#: Shared modules: only the parts the baseline uses.
PARTS = {
    "supportdesk.retrieval.retriever": ("DenseRetriever", "Hit", "_top", "RANK_DECIMALS"),
    "supportdesk.retrieval.index": ("VectorIndex", "index_from_vectors", "_fill", "_payload"),
    "supportdesk.retrieval.filters": ("parse_ts", "visible_mask"),
    "supportdesk.data.chunking": ("chunk_text", "FixedTokenChunker", "build_chunks", "Exclusions"),
    "supportdesk.data.clean": ("clean_text",),
}

#: Settings whose values change what the baseline does.
SETTINGS = (
    "models.chat", "models.embedding", "prompts.system", "prompts.output_format",
    "chunking.method", "chunking.max_tokens", "chunking.overlap_tokens",
    "retrieval.top_k", "retrieval.candidate_k", "retrieval.historical_filter", "pipeline.retriever",
    "context.max_chunks", "context.max_tokens", "context.max_query_chars", "context.max_body_chars",
    "generation.max_output_tokens", "generation.temperature", "generation.max_retries",
)


def _sha(text: str) -> str:
    """Line endings are normalised, so a Windows checkout (CRLF) hashes the same as any other."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def _shown(value) -> str:
    if isinstance(value, Path):  # record paths relative to the project, the same on every machine
        try:
            return value.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(value)
    return str(value)


def baseline_fingerprint() -> dict:
    files = {rel: _sha((ROOT / rel).read_text(encoding="utf-8")) for rel in WHOLE_FILES}
    parts = {}
    for module_name, names in PARTS.items():
        module = importlib.import_module(module_name)
        for name in names:
            obj = getattr(module, name)
            source = inspect.getsource(obj) if inspect.isclass(obj) or inspect.isfunction(obj) else repr(obj)
            parts[f"{module_name}.{name}"] = _sha(source)
    settings = Settings.load(env={}, overrides=["pipeline.name=baseline"])
    values = {dotted: _shown(getattr(getattr(settings, dotted.split(".")[0]), dotted.split(".")[1]))
              for dotted in SETTINGS}
    return {"files": files, "parts": parts, "settings": values}


def changes(recorded: dict, current: dict) -> list[str]:
    """What differs, as readable lines (for example ``settings: models.chat = gpt-4o``)."""
    found = []
    for group in ("files", "parts", "settings"):
        old, new = recorded.get(group, {}), current.get(group, {})
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                shown = f" = {new.get(key)}" if group == "settings" else ""
                found.append(f"{group}: {key}{shown}")
    return found


def write() -> None:
    FINGERPRINT.write_text(json.dumps(baseline_fingerprint(), indent=1, sort_keys=True) + "\n")
