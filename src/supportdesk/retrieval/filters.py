"""Visibility filters. ``as_of`` makes an evaluation *historical*: a case only sees what existed
when it was filed. Undated chunks (documentation at a pinned commit) are always visible."""
from __future__ import annotations

from datetime import datetime, timezone

from ..types import Chunk


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def visible_mask(chunks: list[Chunk], as_of: str | None) -> list[bool]:
    cutoff = parse_ts(as_of)
    if cutoff is None:
        return [True] * len(chunks)
    out = []
    for chunk in chunks:
        stamp = parse_ts(chunk.as_of)
        out.append(stamp is None or stamp < cutoff)
    return out
