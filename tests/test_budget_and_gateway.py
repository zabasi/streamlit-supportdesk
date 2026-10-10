import json

import pytest

from supportdesk.config import ConfigError, Settings, redact
from supportdesk.llm.accounting import BudgetLedger, compute_cost
from supportdesk.llm.cache import DiskCache
from supportdesk.llm.errors import (
    BudgetExhausted,
    CallLimitReached,
    InputTooLarge,
    ProviderError,
    ReplayMiss,
)
from supportdesk.llm.factory import build_clients
from supportdesk.llm.gateway import Gateway

PRICING = {"verified": True, "models": {
    "chat-x": {"input": 1.0, "cached_input": 0.5, "output": 2.0},
    "emb-x": {"input": 0.1, "cached_input": 0.1, "output": 0.0}}}


class StubTransport:
    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    def chat(self, *, model, messages, max_output_tokens, temperature, json_mode):
        self.calls += 1
        if self.fail:
            raise self.fail
        return {"text": '{"ok": true}', "usage": {"input_tokens": 1000, "cached_input_tokens": 0,
                                                    "output_tokens": 500}}

    def embed(self, *, model, texts):
        self.calls += 1
        return {"vectors": [[1.0, 0.0] for _ in texts],
                "usage": {"input_tokens": 10 * len(texts), "cached_input_tokens": 0, "output_tokens": 0}}


def make_gateway(tmp_path, *, mode="live", ceiling=1.0, transport=None, pricing=PRICING, **kw):
    settings = Settings.from_dict({
        "run": {"mode": mode}, "models": {"chat": "chat-x", "embedding": "emb-x"},
        "budget": {"ceiling_usd": ceiling, "reserve_usd": 0.0,
                   "max_live_calls": kw.pop("max_live_calls", 100)},
        "paths": {"cache_path": str(tmp_path / "c.sqlite")}})
    ledger = BudgetLedger(ceiling_usd=ceiling, path=tmp_path / "l.jsonl" if mode == "live" else None)
    gw = Gateway(mode=mode, settings=settings, ledger=ledger, cache=DiskCache(tmp_path / "c.sqlite"),
                 pricing=pricing, transport=transport or StubTransport())
    return gw, ledger


def test_cost_formula_uses_cached_discount():
    price = PRICING["models"]["chat-x"]
    cost = compute_cost({"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 500}, price)
    assert cost == pytest.approx((600 * 1.0 + 400 * 0.5 + 500 * 2.0) / 1e6)


def test_live_call_is_priced_logged_and_cached(tmp_path):
    gw, ledger = make_gateway(tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    text, usage = gw.complete(msgs)
    assert json.loads(text) == {"ok": True} and usage["mode"] == "live"
    assert ledger.total_cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    again, usage2 = gw.complete(msgs)  # identical request: free
    assert again == text and usage2["mode"] == "cache" and usage2["cost_usd"] == 0.0
    assert gw.transport.calls == 1
    assert ledger.rows[-1]["mode"] == "cache" and ledger.rows[-1]["original_cost_usd"] > 0
    summary = ledger.summarize()
    assert summary["calls"] == 1 and summary["cache_hits"] == 1 and summary["replayed_cost_usd"] > 0


def test_replay_never_calls_out(tmp_path):
    gw, _ = make_gateway(tmp_path, mode="replay")
    with pytest.raises(ReplayMiss):
        gw.complete([{"role": "user", "content": "never recorded"}])
    assert gw.transport.calls == 0


def test_budget_guard_refuses_before_spending(tmp_path):
    gw, ledger = make_gateway(tmp_path, ceiling=0.0005)  # worst case for one call is ~0.0018
    with pytest.raises(BudgetExhausted):
        gw.complete([{"role": "user", "content": "x" * 100}])
    assert gw.transport.calls == 0 and ledger.total_cost_usd == 0.0


def test_quota_error_is_budget_exhausted_and_recorded(tmp_path):
    gw, ledger = make_gateway(tmp_path)
    from supportdesk.llm import gateway as g
    assert g.is_quota_error("Error: insufficient_quota")
    gw.transport = StubTransport(fail=BudgetExhausted("insufficient_quota"))
    with pytest.raises(BudgetExhausted):
        gw.complete([{"role": "user", "content": "q"}])
    assert ledger.rows[-1]["status"] == "budget_exhausted"


def test_provider_error_is_recorded_not_dropped(tmp_path):
    gw, ledger = make_gateway(tmp_path, transport=StubTransport(fail=ProviderError("boom")))
    with pytest.raises(ProviderError):
        gw.complete([{"role": "user", "content": "q"}])
    assert ledger.rows[-1]["status"] == "provider_error"


def test_call_cap_and_input_size(tmp_path):
    gw, _ = make_gateway(tmp_path, max_live_calls=1)
    gw.complete([{"role": "user", "content": "one"}])
    with pytest.raises(CallLimitReached):
        gw.complete([{"role": "user", "content": "two"}])
    sub = tmp_path / "b"
    sub.mkdir()
    gw2, _ = make_gateway(sub)
    with pytest.raises(InputTooLarge):
        gw2.complete([{"role": "user", "content": "y" * 70_000}])


def test_unverified_or_missing_pricing_blocks_live(tmp_path):
    with pytest.raises(ConfigError, match="not verified"):
        make_gateway(tmp_path, pricing={**PRICING, "verified": False})
    with pytest.raises(ConfigError, match="no price"):
        make_gateway(tmp_path, pricing={"verified": True, "models": {}})


def test_embeddings_are_cached_per_text(tmp_path):
    gw, _ = make_gateway(tmp_path)
    gw.embed(["a", "b"])
    calls = gw.transport.calls
    vecs, _ = gw.embed(["a", "b", "c"])  # only "c" is new
    assert gw.transport.calls == calls + 1 and len(vecs) == 3


def test_ledger_persists_across_runs_and_counts_toward_the_ceiling(tmp_path):
    gw, ledger = make_gateway(tmp_path, ceiling=0.01)
    gw.complete([{"role": "user", "content": "persist"}])
    reopened = BudgetLedger(ceiling_usd=0.01, path=tmp_path / "l.jsonl")
    assert reopened.total_cost_usd == ledger.total_cost_usd > 0
    assert BudgetLedger(ceiling_usd=0.0001, path=tmp_path / "l.jsonl").remaining_usd() < 0


def test_ledger_csv_has_stable_columns(tmp_path):
    _, ledger = make_gateway(tmp_path)
    ledger.record(stage="s", model="m", mode="live", cost_usd=0.1)
    ledger.write_csv(tmp_path / "out.csv")
    header = (tmp_path / "out.csv").read_text().splitlines()[0]
    assert header.startswith("seq,timestamp,stage,request_type,model,mode") and "cost_usd" in header


def test_secrets_never_reach_public_settings_or_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretkey1234567890")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    public = Settings().public()
    assert "sk-" not in json.dumps(public) and public["api_key_present"] is True
    assert public["gateway_host"] == "gateway.example.com"
    assert "sk-supersecretkey" not in redact("401 for key sk-supersecretkey1234567890")


def test_offline_factory_needs_nothing(settings):
    clients = build_clients(settings)
    assert clients.mode == "offline" and clients.ledger.total_cost_usd == 0.0


def _transport_with(monkeypatch, client):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttest1234")
    from supportdesk.llm.gateway import OpenAITransport
    transport = OpenAITransport(api_key_env="OPENAI_API_KEY", base_url=None, timeout_seconds=5.0)
    transport._client = client
    return transport


def test_openai_transport_parses_sdk_shapes(monkeypatch):
    from types import SimpleNamespace as NS
    chat_resp = NS(choices=[NS(message=NS(content='{"a": 1}'))],
                   usage=NS(prompt_tokens=120, completion_tokens=30,
                            prompt_tokens_details=NS(cached_tokens=100)))
    emb_resp = NS(data=[NS(embedding=[0.1, 0.2]), NS(embedding=[0.3, 0.4])], usage=NS(prompt_tokens=7))
    client = NS(chat=NS(completions=NS(create=lambda **kw: chat_resp)),
                embeddings=NS(create=lambda **kw: emb_resp))
    transport = _transport_with(monkeypatch, client)
    out = transport.chat(model="m", messages=[], max_output_tokens=10, temperature=0.0, json_mode=True)
    assert out["text"] == '{"a": 1}' and out["usage"] == {
        "input_tokens": 120, "cached_input_tokens": 100, "output_tokens": 30}
    emb = transport.embed(model="m", texts=["a", "b"])
    assert emb["vectors"] == [[0.1, 0.2], [0.3, 0.4]] and emb["usage"]["input_tokens"] == 7


def test_openai_transport_maps_errors_and_redacts(monkeypatch):
    from types import SimpleNamespace as NS

    def quota(**kw):
        raise RuntimeError("Error 429: insufficient_quota for key sk-abcdefghijklmnop1234")

    def flaky(**kw):
        raise RuntimeError("502 bad gateway")
    for fn, expected in ((quota, BudgetExhausted), (flaky, ProviderError)):
        t = _transport_with(monkeypatch, NS(chat=NS(completions=NS(create=fn)), embeddings=None))
        with pytest.raises(expected) as info:
            t.chat(model="m", messages=[], max_output_tokens=10, temperature=0.0, json_mode=True)
        assert "sk-abcdefghijkl" not in str(info.value)


def test_openai_transport_requires_a_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from supportdesk.llm.gateway import OpenAITransport
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAITransport(api_key_env="OPENAI_API_KEY", base_url=None, timeout_seconds=5.0)


def _priced_settings(tmp_path, mode, verified=True):
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps({**PRICING, "verified": verified}))
    return Settings.from_dict({"run": {"mode": mode}, "models": {"chat": "chat-x", "embedding": "emb-x"},
                               "paths": {"pricing_path": str(pricing), "cache_path": str(tmp_path / "c.sqlite"),
                                         "ledger_path": str(tmp_path / "l.jsonl")}})


def test_factory_builds_replay_clients_that_never_call_out(tmp_path):
    clients = build_clients(_priced_settings(tmp_path, "replay"), transport=StubTransport())
    with pytest.raises(ReplayMiss):
        clients.chat.complete([{"role": "user", "content": "x"}])
    assert clients.ledger.path is None  # replay spends nothing, so nothing is persisted


def test_factory_builds_live_clients_with_a_persistent_ledger(tmp_path):
    clients = build_clients(_priced_settings(tmp_path, "live"), transport=StubTransport())
    clients.chat.complete([{"role": "user", "content": "x"}])
    assert (tmp_path / "l.jsonl").exists() and clients.ledger.total_cost_usd > 0


def test_factory_refuses_live_mode_with_unverified_prices_or_a_missing_price_file(tmp_path):
    with pytest.raises(ConfigError, match="not verified"):
        build_clients(_priced_settings(tmp_path, "live", verified=False), transport=StubTransport())
    s = Settings.from_dict({"run": {"mode": "live"}, "paths": {"pricing_path": str(tmp_path / "none.json")}})
    with pytest.raises(ConfigError, match="pricing file not found"):
        build_clients(s, transport=StubTransport())


def test_embedding_requests_are_split_to_stay_under_the_size_limit(tmp_path):
    """Long chunks used to make one batch of 64 exceed limits.max_input_chars, so indexing was
    refused. Requests are now split by size as well as count, and none is refused."""
    import json

    class Recording(StubTransport):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def embed(self, *, model, texts):
            self.sizes.append(len(json.dumps({"texts": texts}, ensure_ascii=False)))
            return super().embed(model=model, texts=texts)

    transport = Recording()
    gw, _ = make_gateway(tmp_path, ceiling=100.0, transport=transport)
    texts = [f"chunk {i} " + "x" * 3000 for i in range(64)]  # about 194,000 characters in one batch
    vectors, _ = gw.embed(texts, stage="index")
    assert len(vectors) == 64
    limit = gw.settings.limits.max_input_chars
    assert len(transport.sizes) > 1 and max(transport.sizes) <= limit  # split, and every request fits
    assert transport.calls == len(transport.sizes)
