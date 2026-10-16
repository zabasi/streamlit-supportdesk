"""Saved conversations for the web UI: one SQLite file, a JSON document per conversation.

A conversation keeps its report (title, body, environment, filing time), the earlier messages it started
with, every message since (the assistant's replies and the user's answers), and a record of each pipeline
turn (outcome, evidence, cost). It is local to this computer, like the cache and the cost ledger.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path

MAX_FOLLOWUPS = 5  # clarification questions per conversation (the controlled-routing v2 limit)
MAX_TURNS = 6  # pipeline turns per conversation
OPEN = ("new", "awaiting_reply", "error")  # statuses in which another turn may run


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations "
                       "(id TEXT PRIMARY KEY, updated TEXT NOT NULL, data TEXT NOT NULL)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def find(self, **source) -> dict | None:
        """The first conversation whose source has all these fields (an imported run case, say)."""
        for item in self.listing():
            if all(item["source"].get(k) == v for k, v in source.items()):
                return self.get(item["id"])
        return None

    def create(self, *, pipeline: str, source: dict, report: dict, history: list[dict],
               case_id: str | None = None) -> dict:
        conv = {"id": uuid.uuid4().hex[:12], "created_at": now(), "updated_at": now(), "pipeline": pipeline,
                "case_id": case_id or content_case_id(report, history),
                "source": source, "report": report, "initial_history": history, "messages": [],
                "questions_used": 0, "status": "new", "turns": []}
        self.save(conv)
        return conv

    def save(self, conv: dict) -> None:
        conv["updated_at"] = now()
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO conversations (id, updated, data) VALUES (?, ?, ?)",
                       (conv["id"], conv["updated_at"], json.dumps(conv, ensure_ascii=False)))

    def get(self, conv_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT data FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, conv_id: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)).rowcount > 0

    def listing(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT data FROM conversations ORDER BY updated DESC").fetchall()
        out = []
        for (data,) in rows:
            c = json.loads(data)
            out.append({"id": c["id"], "title": c["report"]["title"], "status": c["status"],
                        "pipeline": c["pipeline"], "questions_used": c["questions_used"],
                        "turns": len(c["turns"]), "source": c["source"], "updated_at": c["updated_at"]})
        return out

    def all(self) -> list[dict]:
        return [self.get(item["id"]) for item in self.listing()]


def content_case_id(report: dict, history: list[dict]) -> str:
    """A case ID from the report itself: the same report and earlier messages give the same ID, so the
    same requests (some answer steps include the case ID) replay from the cache instead of being paid
    again; a different report gives a different ID."""
    key = json.dumps({"title": report.get("title"), "body": report.get("body"), "created_at": report.get("created_at"),
                      "environment": report.get("environment") or {},
                      "history": [{"role": m.get("role"), "body": m.get("body")} for m in history]},
                     sort_keys=True, ensure_ascii=False)
    return "web-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def summary_markdown(conv: dict) -> str:
    """The conversation and the latest turn's maintainer summary, as Markdown."""
    r = conv["report"]
    lines = [f"# {r['title']}", "",
             (f"Conversation `{conv['id']}` · pipeline `{conv['pipeline']}` · status **{conv['status']}** · "
             f"{conv['questions_used']} clarification question(s) · {len(conv['turns'])} turn(s)"), "",
             "## Report", "", r.get("body") or "_(no description)_", ""]
    if r.get("environment"):
        lines += ["Environment: " + ", ".join(f"{k}={v}" for k, v in r["environment"].items()), ""]
    if conv["initial_history"] or conv["messages"]:
        lines += ["## Conversation", ""]
        for m in conv["initial_history"] + conv["messages"]:
            lines += [f"**{'Assistant' if m['role'] == 'assistant' else 'User'}:** {m['body']}", ""]
    last = conv["turns"][-1]["result"] if conv["turns"] else {}
    s = last.get("maintainer_summary") or {}
    lines += ["## Maintainer summary", ""]
    if not s:
        lines += ["_No summary yet._", ""]
    else:
        known = (s.get("environment") or {}).get("known") or {}
        unknown = (s.get("environment") or {}).get("unknown") or []
        lines += [f"- **Outcome:** {last.get('outcome')}", f"- **Problem:** {s.get('problem') or '—'}",
                  f"- **Environment:** {', '.join(f'{k}: {v}' for k, v in known.items()) or 'nothing stated'}"
                  + (f" (not stated: {', '.join(unknown)})" if unknown else ""),
                  f"- **Next step:** {s.get('next_step') or '—'}", ""]
        if s.get("known_facts"):
            lines += ["### Facts", ""] + [f"- {f['statement']}" + (f" [{', '.join(f['evidence'])}]" if f.get("evidence") else "")
                                          for f in s["known_facts"]] + [""]
        if s.get("open_questions"):
            lines += ["### Open questions", ""] + [f"- {q}" for q in s["open_questions"]] + [""]
        if s.get("sources"):
            lines += ["### Sources", ""] + [f"- {src.get('chunk_id')} ({src.get('type')}) {src.get('url') or ''}".rstrip()
                                            for src in s["sources"]] + [""]
    return "\n".join(lines)


def report_html(conversations: list[dict]) -> str:
    """Every saved conversation on one self-contained page (all text escaped)."""
    css = ("body{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1000px;padding:0 16px;color:#1c2330}"
           "table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #dfe3ea;padding:6px 9px;"
           "text-align:left;vertical-align:top}th{background:#f1f3f7}section{border:1px solid #dfe3ea;border-radius:8px;"
           "padding:4px 16px;margin:18px 0}.msg{white-space:pre-wrap;margin:6px 0;padding:8px 10px;border-radius:6px;"
           "background:#f6f7f9}.assistant{background:#e8eefc}.muted{color:#5d6879}")
    rows = "".join(f"<tr><td><a href='#c-{escape(c['id'])}'>{escape(c['report']['title'])}</a></td>"
                   f"<td>{escape(c['status'])}</td><td>{escape(c['pipeline'])}</td><td>{len(c['turns'])}</td>"
                   f"<td>{c['questions_used']}</td><td>{escape(c['updated_at'])}</td></tr>" for c in conversations)
    parts = [f"<!doctype html><meta charset=utf-8><title>supportdesk conversations</title><style>{css}</style>",
             (f"<h1>supportdesk conversations</h1><p class=muted>{len(conversations)} saved conversation(s), "
             f"generated {escape(now())}.</p>"),
             ("<table><tr><th>Report</th><th>Status</th><th>Pipeline</th><th>Turns</th><th>Questions</th>"
             f"<th>Updated</th></tr>{rows}</table>")]
    for c in conversations:
        msgs = "".join(f"<div class='msg {escape(m['role'])}'><b>{'Assistant' if m['role'] == 'assistant' else 'User'}"
                       f"</b>\n{escape(m['body'])}</div>" for m in c["initial_history"] + c["messages"])
        cost = sum((t.get("usage") or {}).get("cost_usd", 0) or 0 for t in c["turns"])
        parts.append(f"<section id='c-{escape(c['id'])}'><h2>{escape(c['report']['title'])}</h2>"
                     f"<p class=muted>{escape(c['pipeline'])} · {escape(c['status'])} · {len(c['turns'])} turn(s) · "
                     f"cost ${cost:.6f}</p><div class=msg>{escape(c['report'].get('body') or '')}</div>{msgs}"
                     f"<pre class=muted>{escape(summary_markdown(c).split('## Maintainer summary', 1)[-1])}</pre></section>")
    return "".join(parts)


__all__ = ["MAX_FOLLOWUPS", "MAX_TURNS", "OPEN", "ConversationStore", "content_case_id", "report_html",
           "summary_markdown"]
