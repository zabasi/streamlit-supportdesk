"""Write a short summary of every documentation section, once, for the section-summaries experiment.

Sections are the top-level sections of each documentation page (a ## section with its ### subsections),
as in the sections experiments. Each summary is written by one model call through the usual gateway
(cache, budget limit, cost ledger). Only documentation is summarized; evaluation cases are never seen.

    python scripts/summarize_sections.py --mode live            # writes data/summaries/sections.jsonl
    python scripts/summarize_sections.py --mode live --set paths.summaries=data/summaries/other.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from supportdesk.config import Settings
from supportdesk.data.corpus import load_corpus
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.stages.sections import section_spans

MAX_SECTION_CHARS = 4000


def summarize(settings: Settings, *, limit: int | None = None) -> Path:
    clients = build_clients(settings)
    prompt_path = Path(settings.prompts.summarize_section)
    prompt = prompt_path.read_text(encoding="utf-8")
    records = [r for r in load_corpus(settings.paths.data_dir).records if r.source_type == "documentation"]
    rows = []
    for record in records:
        for start, end in section_spans(record.text):
            text = record.text[start:end].strip()
            heading = text.splitlines()[0].lstrip("# ").strip() if text else ""
            if not text:
                continue
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": f"Page: {record.title}\n\n{text[:MAX_SECTION_CHARS]}"}]
            summary, _ = clients.chat.complete(messages, stage="summarize", json_mode=False)
            rows.append({"source_id": record.source_id, "start_char": start, "end_char": end,
                         "title": record.title, "heading": heading, "summary": " ".join(summary.split())})
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    out = Path(settings.paths.summaries)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {"sections": len(rows), "pages": len({r["source_id"] for r in rows}), "model": settings.models.chat,
                "mode": settings.run.mode, "prompt": str(prompt_path),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "data_dir": str(settings.paths.data_dir),
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    parser.add_argument("--limit", type=int, default=None, help="summarize at most this many sections (to test)")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    out = summarize(settings, limit=args.limit)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    print(f"wrote {manifest['sections']} section summaries of {manifest['pages']} pages to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
