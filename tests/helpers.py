"""Test helpers: build validated settings with every path under a temp directory."""
from __future__ import annotations

from pathlib import Path

from supportdesk.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"


def override(settings: Settings, **sections) -> Settings:
    """A copy of ``settings`` with some values changed, e.g. ``override(s, retrieval={"top_k": 3})``."""
    data = settings.to_dict()
    for name, body in sections.items():
        data[name] = {**data[name], **body}
    return Settings.from_dict(data)


def make_settings(base: Path, **sections) -> Settings:
    """Offline settings on the sample data. Pass sections as dicts, e.g.
    ``make_settings(tmp, retrieval={"top_k": 2}, run={"mode": "replay"})``."""
    data = {
        "run": {"mode": "offline"},
        "prompts": {"system": str(ROOT / "configs" / "prompts" / "answer_system.txt"),
                    "output_format": str(ROOT / "configs" / "prompts" / "answer_output_format.txt")},
        "paths": {"data_dir": str(SAMPLE), "index_dir": str(base / "index"),
                  "artifacts_dir": str(base / "artifacts"), "cache_path": str(base / "cache.sqlite"),
                  "ledger_path": str(base / "ledger.jsonl"),
                  "pricing_path": str(ROOT / "configs" / "pricing.json")},
    }
    for name, body in sections.items():
        data.setdefault(name, {}).update(body)
    return Settings.from_dict(data)
