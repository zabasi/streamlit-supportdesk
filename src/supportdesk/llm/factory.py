"""Build the client bundle for a mode. The single place where mode decides behaviour."""
from __future__ import annotations

import dataclasses
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


def chat_client_for(clients: Clients, settings: Settings, model: str):
    """A chat client for another model that shares ``clients``' ledger (spend counted and capped with
    everything else) and, beyond offline, the same cache; the model must have a price in pricing.json."""
    if not model or model == settings.models.chat:
        return clients.chat
    if clients.mode == "offline":
        return FakeChatClient(ledger=clients.ledger)
    other = dataclasses.replace(settings, models=dataclasses.replace(settings.models, chat=model))
    # the same provider connection as the main client (the model is chosen per call)
    return Gateway(mode=clients.mode, settings=other, ledger=clients.ledger,
                   cache=DiskCache(settings.paths.cache_path), pricing=load_pricing(other),
                   transport=getattr(clients.chat, "transport", None))


def second_embedding_client(clients: Clients, settings: Settings, model: str):
    """An embedding client for another model that shares ``clients``' ledger (so its spending is counted and
    capped with everything else) and, beyond offline, the same cache."""
    if clients.mode == "offline":
        return FakeEmbeddingClient(ledger=clients.ledger)
    other = dataclasses.replace(settings, models=dataclasses.replace(settings.models, embedding=model))
    return Gateway(mode=clients.mode, settings=other, ledger=clients.ledger,
                   cache=DiskCache(settings.paths.cache_path), pricing=load_pricing(other),
                   transport=getattr(clients.embedding, "transport", None))
