"""A small SQLite request cache. It is what makes re-runs free and replay mode possible."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def request_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)  # callers serialise access
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict | None:
        row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self._db.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                         (key, json.dumps(value, ensure_ascii=False)))
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self) -> None:
        self._db.close()
