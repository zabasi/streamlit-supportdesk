"""Free offline stand-ins for the models. The embedder gives each word a fixed vector, so texts that
share words come out similar; the chat model copies a sentence from the best evidence into a fixed
template. They give the same output on every run: enough to test the system, not to judge answer
quality."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from ..tokenizer import count_tokens, lexical_tokens
from .accounting import BudgetLedger
from .errors import ProviderError

DIM = 256


def embed_text(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    for token in lexical_tokens(text):
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class FakeEmbeddingClient:
    ledger: BudgetLedger | None = None
    model: str = "fake-embedding"
    script: dict = field(default_factory=dict)  # maps a call number to "error" or "short_batch"
    calls: int = 0

    def embed(self, texts, *, stage="embed"):
        self.calls += 1
        behaviour = self.script.get(self.calls)
        if behaviour == "error":
            raise ProviderError("fake embedding provider unavailable")
        vectors = [embed_text(t) for t in texts]
        if behaviour == "short_batch":
            vectors = vectors[:-1]
        usage = {"model": self.model, "mode": "offline", "input_tokens": sum(map(count_tokens, texts)),
                 "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "status": "ok"}
        if self.ledger is not None:
            self.ledger.record(stage=stage, request_type="embedding", items=len(texts), **usage)
        return vectors, usage


_EVIDENCE = re.compile(r'<evidence id="(S\d+)"[^>]*>\n?(.*?)\n?</evidence>', re.DOTALL)


@dataclass
class FakeChatClient:
    """Extractive answerer. ``script`` maps call number -> "error" | "bad_json" | raw text."""

    ledger: BudgetLedger | None = None
    model: str = "fake-chat"
    script: dict = field(default_factory=dict)
    calls: int = 0
    seen: list = field(default_factory=list)

    def complete(self, messages, *, stage="answer", json_mode=True):
        self.calls += 1
        self.seen.append(messages)
        behaviour = self.script.get(self.calls)
        if behaviour == "error":
            raise ProviderError("fake chat provider unavailable")
        user = messages[-1]["content"]
        if behaviour == "bad_json":
            text = "Sorry, here is some prose instead of JSON."
        elif behaviour:
            text = behaviour
        else:
            text = json.dumps(self._answer(user))
        usage = {"model": self.model, "mode": "offline", "input_tokens": count_tokens(user),
                 "cached_input_tokens": 0, "output_tokens": count_tokens(text), "cost_usd": 0.0,
                 "status": "ok"}
        if self.ledger is not None:
            self.ledger.record(stage=stage, request_type="chat", items=1, **usage)
        return text, usage

    @staticmethod
    def _answer(user: str) -> dict:
        blocks = _EVIDENCE.findall(user)
        if not blocks:
            return {"outcome": "request_info", "user_reply": "I could not find related material. "
                    "Could you share the Streamlit version and a minimal code example?",
                    "known_facts": [], "open_questions": ["Streamlit version?", "Minimal example?"],
                    "next_step": "Ask the reporter for a minimal reproducible example.",
                    "citations": []}
        label, text = blocks[0]
        sentence = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))[0][:300]
        return {"outcome": "answer", "user_reply": f"Related material suggests: {sentence} [{label}]",
                "known_facts": [{"statement": sentence, "evidence": [label]}],
                "open_questions": [], "next_step": "Check whether this matches your setup.",
                "citations": [label]}
