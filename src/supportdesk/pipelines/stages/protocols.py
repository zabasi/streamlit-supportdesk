"""The stage contracts. Each stage is one replaceable step of a pipeline.

A stage is any callable with the signature below; it needs no base class. Stages that need a model
take the client bundle in their constructor (``Clients``), so every call they make lands in the
same cost ledger and is counted in the per-case usage automatically.

Failure rules (the pipeline maps these for you, so stages should not catch them):
  * let ``GatewayError`` subclasses (provider error, budget exhausted, call cap) propagate;
  * raise ``GenerationFailed`` (generators only) when no valid output was obtained;
  * raise ``PayloadError`` for model output that parses but has the wrong shape;
  * any other exception is a bug and is allowed to crash the run, so it is never mistaken for a
    model failure in an evaluation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...retrieval.retriever import Hit
from ...types import CaseInput
from ..packet import Packet
from ..parsing import ParsedAnswer


class GenerationFailed(RuntimeError):
    """The generator could not obtain a valid answer. Carries the attempt log for the trace."""

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


@runtime_checkable
class QueryBuilder(Protocol):
    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        """One or more search queries (never empty). Several queries are fused with RRF."""


@runtime_checkable
class Reranker(Protocol):
    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        """Reorder (and optionally drop) candidates. The pipeline trims the result to top_k."""


@runtime_checkable
class PacketBuilder(Protocol):
    def __call__(self, hits: list[Hit]) -> Packet:
        """Choose and label what the model may see, within the chunk and token limits."""


@runtime_checkable
class Generator(Protocol):
    def __call__(self, case: CaseInput, env: dict, packet: Packet) -> ParsedAnswer:
        """Ask the model and return a parsed, label-resolved answer (with ``attempts`` filled)."""


@runtime_checkable
class Validator(Protocol):
    def __call__(self, case: CaseInput, packet: Packet, answer: ParsedAnswer) -> ParsedAnswer:
        """Check the answer. May add violations or downgrade the outcome; must return an answer."""


@runtime_checkable
class Summarizer(Protocol):
    def __call__(self, case: CaseInput, env: dict, packet: Packet, answer: ParsedAnswer) -> dict:
        """The maintainer summary, built from validated fields."""
