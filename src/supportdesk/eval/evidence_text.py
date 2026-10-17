"""The text of an evidence piece as the model saw it: an index chunk, or a whole section or merged
neighbours (``source_id#start-end``) rebuilt from the documentation page with the helper the pipeline uses."""
from __future__ import annotations

import dataclasses
import re

from ..pipelines.stages.sections import page_texts
from ..types import Chunk


class EvidenceTexts:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        self._all = list(chunks)
        self._pages = None

    def get(self, chunk_id: str) -> Chunk | None:
        """The piece, or None when it cannot be rebuilt."""
        chunk = self.chunks.get(chunk_id)
        if chunk is not None:
            return chunk
        match = re.fullmatch(r"(.+)#(\d+)-(\d+)", chunk_id)
        if not match:
            return None
        source, a, b = match.group(1), int(match.group(2)), int(match.group(3))
        if self._pages is None:
            self._pages = page_texts(self._all)
        page = self._pages.get(source)
        base = next((c for c in self._all if c.source_id == source), None)
        if page is None or base is None or not 0 <= a < b <= len(page):
            return None
        return dataclasses.replace(base, chunk_id=chunk_id, text=page[a:b], start_char=a, end_char=b)


__all__ = ["EvidenceTexts"]
