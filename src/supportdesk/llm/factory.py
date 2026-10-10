"""Build the client bundle for a mode. The single place where mode decides behaviour."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import ConfigError, Settings
from .accounting import BudgetLedger
from .cache import DiskCache
from .fake import FakeChatClient, FakeEmbeddingClient
from .gateway import Gateway, Transport
from .protocols import ChatClient, EmbeddingClient


@dataclass
class Clients:
    mode: str
    chat: ChatClient
    embedding: EmbeddingClient
    ledger: BudgetLedger
    embedding_model: str


def load_pricing(settings: Settings) -> dict:
    if not settings.paths.pricing_path.exists():
        raise ConfigError(f"pricing file not found: {settings.paths.pricing_path}")
    return json.loads(settings.paths.pricing_path.read_text(encoding="utf-8"))


def build_clients(settings: Settings, *, transport: Transport | None = None) -> Clients:
    """Fakes offline; otherwise the gateway (replay never calls out, live spends money)."""
    mode = settings.mode
    if mode == "offline":
        ledger = BudgetLedger(ceiling_usd=settings.budget.ceiling_usd)
        chat, emb = FakeChatClient(ledger=ledger), FakeEmbeddingClient(ledger=ledger)
        return Clients(mode, chat, emb, ledger, emb.model)
    # replay keeps an in-memory ledger (nothing is spent); live persists spend across runs
    ledger = BudgetLedger(ceiling_usd=settings.budget.ceiling_usd,
                          path=settings.paths.ledger_path if mode == "live" else None)
    gateway = Gateway(mode=mode, settings=settings, ledger=ledger,
                      cache=DiskCache(settings.paths.cache_path), pricing=load_pricing(settings),
                      transport=transport)
    return Clients(mode, gateway, gateway, ledger, settings.models.embedding)
