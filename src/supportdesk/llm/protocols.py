"""The only way the rest of the system reaches a model, so the same code runs offline, on replay
and live without a branch."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], dict]:
        """Vectors in input order, and a usage record."""


@runtime_checkable
class ChatClient(Protocol):
    def complete(self, messages: list[dict], *, stage: str = "answer",
                 json_mode: bool = True) -> tuple[str, dict]:
        """The raw model text and a usage record. ``json_mode`` asks the provider for a JSON object."""
