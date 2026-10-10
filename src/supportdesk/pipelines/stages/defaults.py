"""The baseline's stage implementations. Deliberately plain; every advanced stage is measured
against these."""
from __future__ import annotations

from ...config import Settings, redact
from ...data.clean import clean_text
from ...llm.errors import ProviderError
from ...llm.factory import Clients
from ...retrieval.retriever import Hit
from ...types import CaseInput
from ..packet import Packet, assemble_packet
from ..parsing import NotJson, ParsedAnswer, parse_answer
from ..prompts import build_messages, load_prompts
from ..summary import build_summary
from .protocols import GenerationFailed


def build_query(case: CaseInput, max_chars: int) -> str:
    """Title, body and the reporter's own earlier messages (never the maintainer's)."""
    parts = [case.title, case.body] + [h.get("body", "") for h in case.history
                                       if h.get("role", "user") == "user"]
    return clean_text("\n\n".join(p for p in parts if p))[:max_chars]


class PlainQuery:
    """One query: the report as written, cut to ``context.max_query_chars``."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def __call__(self, case: CaseInput, env: dict) -> list[str]:
        return [build_query(case, self.max_chars)]


class NoRerank:
    """Keep the retriever's order."""

    def __call__(self, case: CaseInput, query: str, hits: list[Hit]) -> list[Hit]:
        return hits


class DefaultPacketBuilder:
    """Keep chunks in rank order until the chunk or token limit is reached."""

    def __init__(self, max_chunks: int, max_tokens: int):
        self.max_chunks, self.max_tokens = max_chunks, max_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> DefaultPacketBuilder:
        return cls(settings.prompt_chunks, settings.context.max_tokens)

    def __call__(self, hits: list[Hit]) -> Packet:
        return assemble_packet(hits, max_chunks=self.max_chunks, max_tokens=self.max_tokens)


def call_and_parse(chat, messages: list[dict], packet: Packet, *, retries: int,
                   stage: str = "answer") -> ParsedAnswer:
    """One model call with the shared retry policy; reuse it in any generator.

    Retries only provider errors and non-JSON output. A wrong-shape payload or a budget error
    propagates at once: asking again until the output fits is a slow way of inventing a citation.
    """
    attempts: list[dict] = []
    for attempt in range(1 + retries):
        try:
            text, _ = chat.complete(messages, stage=stage, json_mode=True)
            parsed = parse_answer(text, packet)
        except (ProviderError, NotJson) as exc:
            attempts.append({"attempt": attempt, "status": type(exc).__name__,
                             "detail": redact(exc)})
            continue
        attempts.append({"attempt": attempt, "status": "ok"})
        parsed.attempts, parsed.raw_output = attempts, text
        return parsed
    raise GenerationFailed(f"no valid model output after {len(attempts)} attempts", attempts)


class SingleCallGenerator:
    """One prompt with the evidence, one model call (retried only for provider errors or non-JSON)."""

    def __init__(self, clients: Clients, settings: Settings):
        self.clients, self.settings = clients, settings
        self.prompts = load_prompts(settings)

    def __call__(self, case: CaseInput, env: dict, packet: Packet) -> ParsedAnswer:
        messages = build_messages(case, env, packet, self.prompts,
                                  max_body_chars=self.settings.context.max_body_chars)
        return call_and_parse(self.clients.chat, messages, packet,
                              retries=self.settings.generation.max_retries)


class NoValidator:
    """No checks beyond the citation-membership flags the parser already sets."""

    def __call__(self, case: CaseInput, packet: Packet, answer: ParsedAnswer) -> ParsedAnswer:
        return answer


def default_stages(retriever, clients: Clients, settings: Settings) -> dict:
    """The seven baseline stages, as keyword arguments for ``StagedPipeline``."""
    return {
        "query": PlainQuery(settings.context.max_query_chars), "retriever": retriever,
        "reranker": NoRerank(), "packet": DefaultPacketBuilder.from_settings(settings),
        "generator": SingleCallGenerator(clients, settings), "validator": NoValidator(),
        "summarizer": build_summary,
    }
