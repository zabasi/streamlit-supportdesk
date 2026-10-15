"""The clients that call a real model (live mode) or play back recorded calls (replay mode). Every call
goes through the cache, then the safety checks (request size, call limit, budget), then the
provider, and is written to the cost ledger. Tests pass in a fake provider."""
from __future__ import annotations

import json
import os
import time
from typing import Protocol

from .. import config
from ..config import ConfigError, redact
from .accounting import BudgetLedger, compute_cost
from .cache import DiskCache, request_hash
from .errors import BudgetExhausted, CallLimitReached, InputTooLarge, ProviderError, ReplayMiss

QUOTA_MARKERS = (
    "insufficient_quota", "insufficient quota", "quota exceeded", "quota_exceeded",
    "billing_hard_limit", "exceeded your current quota", "budget exceeded", "budget_exceeded",
    "credit balance is too low", "insufficient balance", "insufficient credit",
)


def is_quota_error(error: object) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


class Transport(Protocol):
    def chat(self, *, model: str, messages: list[dict], max_output_tokens: int,
             temperature: float, json_mode: bool) -> dict:
        """-> {"text": str, "usage": {"input_tokens", "cached_input_tokens", "output_tokens"}}"""

    def embed(self, *, model: str, texts: list[str]) -> dict:
        """-> {"vectors": list[list[float]], "usage": {...}}"""


class OpenAITransport:
    """Thin wrapper over the ``openai`` SDK. The key is read from the environment variable it is given."""

    def __init__(self, *, api_key_env: str, base_url: str | None, timeout_seconds: float) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("live mode needs the 'live' extra: pip install -e '.[live]'") from exc
        key = os.environ.get(api_key_env)
        if not key:
            raise ConfigError(f"environment variable {api_key_env} is not set (models.api_key_env)")
        self._client = OpenAI(api_key=key, base_url=base_url,
                              timeout=timeout_seconds, max_retries=0)

    def chat(self, *, model, messages, max_output_tokens, temperature, json_mode):
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        # GPT-5 chat models reject max_tokens and only allow the default temperature.
        newer = str(model).startswith("gpt-5")
        extras = {("max_completion_tokens" if newer else "max_tokens"): max_output_tokens}
        if not newer:
            extras["temperature"] = temperature
        try:
            resp = self._client.chat.completions.create(
                model=model, messages=messages, **extras, **options)
        except Exception as exc:  # noqa: BLE001 - SDK exception types vary across gateways
            raise self._map(exc) from None
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        return {"text": resp.choices[0].message.content or "",
                "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                          "cached_input_tokens": getattr(details, "cached_tokens", 0) or 0,
                          "output_tokens": getattr(usage, "completion_tokens", 0) or 0}}

    def embed(self, *, model, texts):
        try:
            resp = self._client.embeddings.create(model=model, input=texts)
        except Exception as exc:  # noqa: BLE001
            raise self._map(exc) from None
        usage = getattr(resp, "usage", None)
        return {"vectors": [d.embedding for d in resp.data],
                "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                          "cached_input_tokens": 0, "output_tokens": 0}}

    @staticmethod
    def _map(exc: Exception) -> Exception:
        message = redact(exc)
        if is_quota_error(message):
            return BudgetExhausted(message)
        return ProviderError(message)


class Gateway:
    """Chat + embeddings in one of: ``replay`` (cache only) or ``live`` (cache, then network)."""

    def __init__(self, *, mode: str, settings: config.Settings, ledger: BudgetLedger,
                 cache: DiskCache, pricing: dict, transport: Transport | None = None):
        if mode not in ("replay", "live"):
            raise ConfigError(f"Gateway handles replay/live, not {mode!r}")
        if mode == "live":
            if not pricing.get("verified"):
                raise ConfigError("configs/pricing.json is not verified; set real prices and "
                                  "verified=true before spending money")
            for model in (settings.models.chat, settings.models.embedding):
                if model not in pricing.get("models", {}):
                    raise ConfigError(f"no price for model {model!r} in pricing.json")
            transport = transport or OpenAITransport(
                api_key_env=settings.models.api_key_env, base_url=settings.models.base_url,
                timeout_seconds=settings.generation.timeout_seconds)
        self.mode, self.settings, self.ledger = mode, settings, ledger
        self.cache, self.pricing, self.transport = cache, pricing, transport
        self.max_output_tokens = settings.generation.max_output_tokens
        self.temperature = settings.generation.temperature
        self.live_calls = 0

    def _price(self, model: str) -> dict:
        return self.pricing["models"][model]

    def _embedding_batches(self, texts: list[str], indexes: list[int]) -> list[list[int]]:
        """Split the texts to embed into requests of at most retrieval.embed_batch_size texts, each
        also small enough to pass limits.max_input_chars, so long chunks never make a request too
        big. A single text over the limit still gets its own request, which the size check refuses."""
        limit, size = self.settings.limits.max_input_chars, self.settings.retrieval.embed_batch_size
        batches, current, chars = [], [], 32  # 32: room for the request's own JSON wrapping
        for i in indexes:
            cost = len(json.dumps(texts[i], ensure_ascii=False)) + 2
            if current and (len(current) == size or chars + cost > limit):
                batches.append(current)
                current, chars = [], 32
            current.append(i)
            chars += cost
        if current:
            batches.append(current)
        return batches

    def _guard(self, payload: dict, model: str, *, est_output_tokens: int) -> None:
        size = len(json.dumps(payload, ensure_ascii=False))
        limit = self.settings.limits.max_input_chars
        if size > limit:
            raise InputTooLarge(f"request is {size} chars; limit is {limit}")
        if self.live_calls >= self.settings.budget.max_live_calls:
            raise CallLimitReached(f"{self.live_calls} upstream calls made; cap reached")
        price = self._price(model)
        worst = (size / 3 * price["input"] + est_output_tokens * price["output"]) / 1_000_000
        if self.ledger.would_exceed(worst, self.settings.budget.reserve_usd):
            raise BudgetExhausted(
                f"next call could cost ${worst:.4f}; remaining ${self.ledger.remaining_usd():.4f} "
                f"with ${self.settings.budget.reserve_usd:.2f} reserved")

    def _priced_call(self, *, kind: str, stage: str, model: str, payload: dict, items: int,
                     est_output_tokens: int, send):
        key = request_hash({"kind": kind, "model": model, **payload})
        hit = self.cache.get(key)
        if hit is not None:
            original = compute_cost(hit["usage"], self._price(model)) if model in self.pricing.get(
                "models", {}) else 0.0
            self.ledger.record(stage=stage, request_type=kind, model=model, mode="cache",
                               request_hash=key[:32], items=items, cache_hits=items,
                               cost_usd=0.0, original_cost_usd=original, **hit["usage"])
            return hit["result"], {**hit["usage"], "model": model, "mode": "cache",
                                   "cost_usd": 0.0, "status": "ok"}
        if self.mode == "replay":
            raise ReplayMiss(f"no recorded response for {kind} request {key[:12]}")
        self._guard(payload, model, est_output_tokens=est_output_tokens)
        started = time.perf_counter()
        self.live_calls += 1
        try:
            raw = send()
        except BudgetExhausted:
            self._record_failure(stage, kind, model, key, items, started, "budget_exhausted")
            raise
        except ProviderError:
            self._record_failure(stage, kind, model, key, items, started, "provider_error")
            raise
        latency = round(time.perf_counter() - started, 4)
        cost = compute_cost(raw["usage"], self._price(model))
        self.ledger.record(stage=stage, request_type=kind, model=model, mode="live",
                           request_hash=key[:32], items=items, new_items=items,
                           latency_seconds=latency, cost_usd=cost, **raw["usage"])
        result = raw["text"] if kind == "chat" else raw["vectors"]
        self.cache.put(key, {"result": result, "usage": raw["usage"]})
        return result, {**raw["usage"], "model": model, "mode": "live", "cost_usd": cost,
                        "latency_seconds": latency, "status": "ok"}

    def _record_failure(self, stage, kind, model, key, items, started, status):
        self.ledger.record(stage=stage, request_type=kind, model=model, mode="live",
                           request_hash=key[:32], items=items, status=status, cost_usd=0.0,
                           latency_seconds=round(time.perf_counter() - started, 4))

    def complete(self, messages, *, stage="answer", json_mode=True):
        model = self.settings.models.chat
        payload = {"messages": messages, "max_output_tokens": self.max_output_tokens,
                   "temperature": self.temperature, "json_mode": json_mode}
        return self._priced_call(
            kind="chat", stage=stage, model=model, payload=payload, items=1,
            est_output_tokens=self.max_output_tokens,
            send=lambda: self.transport.chat(model=model, messages=messages,
                                             max_output_tokens=self.max_output_tokens,
                                             temperature=self.temperature, json_mode=json_mode))

    def embed(self, texts, *, stage="embed"):
        """Per-text caching: a re-run, or a larger corpus, only pays for texts never embedded."""
        model = self.settings.models.embedding
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for i, text in enumerate(texts):
            hit = self.cache.get(request_hash({"kind": "embedding", "model": model, "text": text}))
            if hit is not None:
                vectors[i] = hit["result"]
            else:
                missing.append(i)
        usage = {"model": model, "mode": "cache" if not missing else self.mode, "input_tokens": 0,
                 "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "status": "ok"}
        if len(texts) - len(missing):
            self.ledger.record(stage=stage, request_type="embedding", model=model, mode="cache",
                               items=len(texts), cache_hits=len(texts) - len(missing), cost_usd=0.0)
        for batch in self._embedding_batches(texts, missing):
            batch_texts = [texts[i] for i in batch]
            result, u = self._priced_call(
                kind="embedding", stage=stage, model=model, payload={"texts": batch_texts},
                items=len(batch), est_output_tokens=0,
                send=lambda bt=batch_texts: self.transport.embed(model=model, texts=bt))
            if len(result) != len(batch):
                raise ProviderError(f"asked for {len(batch)} vectors, got {len(result)}")
            for i, vec in zip(batch, result):
                vectors[i] = vec
                self.cache.put(request_hash({"kind": "embedding", "model": model, "text": texts[i]}),
                               {"result": vec, "usage": {"input_tokens": 0, "cached_input_tokens": 0,
                                                         "output_tokens": 0}})
            for k in ("input_tokens", "cached_input_tokens", "cost_usd"):
                usage[k] += u.get(k, 0)
        return [vectors[i] for i in range(len(texts))], usage
