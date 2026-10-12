#!/usr/bin/env python3
"""Rewrite the Reference section of docs/configuration.md from the settings dataclasses.

    python scripts/gen_config_docs.py
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from supportdesk.config import SECTIONS, Settings

DOC = ROOT / "docs" / "configuration.md"
MARKER = "## Reference"


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell: a bare | would start a new column."""
    return str(text).replace("|", "\\|")


def reference() -> str:
    defaults = Settings.from_dict({})
    rows = []
    for section, cls in SECTIONS.items():
        rows.append(f"\n### `[{section}]`\n\n| Key | Default | What it does |\n|---|---|---|")
        for f in fields(cls):
            value = getattr(getattr(defaults, section), f.name)
            if value is None:
                shown = "*unset*"
            else:
                shown = f"`{list(value) if isinstance(value, tuple) else value}`"
            rows.append(f"| `{f.name}` | {_cell(shown)} | {_cell(f.metadata.get('help', ''))} |")
    return "\n".join(rows)


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    head = text.split(MARKER)[0]
    intro = ("Defaults shown are the built-in ones. `*unset*` means the key is optional and normally "
             "left out.\n")
    DOC.write_text(f"{head}{MARKER}\n\n{intro}{reference()}\n", encoding="utf-8")
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
