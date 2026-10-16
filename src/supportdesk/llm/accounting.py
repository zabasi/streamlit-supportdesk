"""The cost ledger: every priced request, including failures and cache hits. In live mode it is
appended to disk per call, so spend survives a crash and accumulates across runs."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = (
    "seq", "timestamp", "stage", "request_type", "model", "mode", "request_hash", "items",
    "new_items", "cache_hits", "input_tokens", "cached_input_tokens", "output_tokens",
    "latency_seconds", "status", "cost_usd", "original_cost_usd",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BudgetLedger:
    ceiling_usd: float
    path: Path | None = None
    rows: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path) if self.path else None
        self._read_to = 0  # bytes of the file already in rows
        self._mine: list[str] = []  # lines this ledger wrote that refresh() has not yet passed
        if self.path and self.path.exists():  # prior runs count against the same ceiling
            data = self.path.read_bytes()
            self.rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
            self._read_to = len(data)

    def refresh(self) -> None:
        """Catch up with lines other processes appended to the same file (an evaluation running while the
        web UI is open), so the budget readout and the guard count all spending, not only this
        process's. Reads only the new bytes; this ledger's own lines are not counted twice."""
        if not self.path or not self.path.exists():
            return
        size = self.path.stat().st_size
        if size <= self._read_to:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._read_to)
            chunk = handle.read(size - self._read_to)
        end = chunk.rfind(b"\n") + 1  # whole lines only; a line still being written waits for next time
        for line in chunk[:end].decode("utf-8").splitlines():
            if not line.strip():
                continue
            if line in self._mine:
                self._mine.remove(line)  # written by this ledger: already in rows
            else:
                self.rows.append(json.loads(line))
        self._read_to += end

    def record(self, **row) -> dict:
        self.refresh()
        entry = {"seq": len(self.rows), "status": "ok", "cost_usd": 0.0, **row}
        entry.setdefault("request_type", entry.get("stage", "unknown"))
        entry.setdefault("timestamp", _now() if entry.get("mode") == "live" else "")
        self.rows.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            self._mine.append(line)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return entry

    @property
    def total_cost_usd(self) -> float:
        self.refresh()
        return round(sum(r.get("cost_usd", 0.0) for r in self.rows), 12)

    def remaining_usd(self) -> float:
        return round(self.ceiling_usd - self.total_cost_usd, 12)

    def would_exceed(self, estimated_cost: float, reserve: float = 0.0) -> bool:
        return self.total_cost_usd + estimated_cost + reserve > self.ceiling_usd

    def mark(self) -> int:
        """A position to diff against later (per-case usage = rows since the mark)."""
        return len(self.rows)

    def summarize(self, since: int = 0) -> dict:
        rows = self.rows[since:]
        live = [r for r in rows if r.get("mode") == "live"]
        return {
            "calls": len(live),
            "cache_hits": sum(1 for r in rows if r.get("mode") == "cache"),
            "failed_calls": sum(1 for r in rows if r.get("status") != "ok"),
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 10),
            "replayed_cost_usd": round(sum(r.get("original_cost_usd", 0.0) for r in rows), 10),
            "latency_seconds": round(sum(r.get("latency_seconds") or 0.0 for r in rows), 4),
        }

    def by_stage(self) -> dict:
        out: dict = {}
        for r in self.rows:
            b = out.setdefault(r.get("stage", "unknown"),
                               {"rows": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            b["rows"] += 1
            b["input_tokens"] += r.get("input_tokens", 0)
            b["output_tokens"] += r.get("output_tokens", 0)
            b["cost_usd"] = round(b["cost_usd"] + r.get("cost_usd", 0.0), 10)
        return out

    def write_csv(self, path: str | Path, since: int = 0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows[since:]:
                writer.writerow({k: row.get(k, "") for k in COLUMNS})


def compute_cost(usage: dict, price: dict) -> float:
    """USD for one call. ``price`` is per-million-token {input, cached_input, output}."""
    cached = usage.get("cached_input_tokens", 0)
    fresh = max(usage.get("input_tokens", 0) - cached, 0)
    return round(
        (fresh * price["input"] + cached * price.get("cached_input", price["input"])
         + usage.get("output_tokens", 0) * price["output"]) / 1_000_000, 12)
