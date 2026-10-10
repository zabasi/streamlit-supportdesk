"""Keyword search (BM25). It finds exact identifiers and error messages that embedding search can miss."""
from __future__ import annotations

import math
from collections import Counter

from ..tokenizer import lexical_tokens
from ..types import Chunk


class BM25:
    def __init__(self, chunks: list[Chunk], *, k1: float, b: float):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(lexical_tokens(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = lexical_tokens(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1.0))
                    s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            out.append(s)
        return out
