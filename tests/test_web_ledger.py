"""The web UI's spending in live mode lands in the shared cost ledger, like every evaluation's."""
import dataclasses
import json

from supportdesk.llm.accounting import BudgetLedger
from supportdesk.llm.factory import build_clients
from supportdesk.web.server import App


class LiveStub:
    """A provider stand-in: real-looking usage, no network."""

    def __init__(self, dimension):
        self.dimension, self.calls = dimension, 0

    def chat(self, *, model, messages, max_output_tokens, temperature, json_mode, extra=None):
        self.calls += 1
        return {"text": '{"ok": true}', "finish": {"finish_reason": "stop", "refusal": None},
                "usage": {"input_tokens": 1200, "cached_input_tokens": 0, "output_tokens": 300}}

    def embed(self, *, model, texts):
        self.calls += 1
        return {"vectors": [[1.0] + [0.0] * (self.dimension - 1) for _ in texts],
                "usage": {"input_tokens": 20 * len(texts), "cached_input_tokens": 0, "output_tokens": 0}}


def test_a_web_turn_in_live_mode_is_recorded_in_the_shared_ledger(settings, index, tmp_path):
    live = dataclasses.replace(
        settings, run=dataclasses.replace(settings.run, mode="live"),
        paths=dataclasses.replace(settings.paths, ledger_path=tmp_path / "cost_ledger.jsonl",
                                  cache_path=tmp_path / "cache.sqlite", artifacts_dir=tmp_path / "artifacts"),
        budget=dataclasses.replace(settings.budget, ceiling_usd=5.0, reserve_usd=0.0))
    stub = LiveStub(index.dimension)
    clients = build_clients(live, transport=stub)
    app = App(live, clients=clients, index=index, store_path=tmp_path / "c.sqlite")
    before = app.status()["budget"]["spent_usd"]
    conv = app.conversation_new({"title": "Value lost", "body": "st.text_input loses its value after a rerun.",
                                 "pipeline": "baseline"})
    conv = app.conversation_turn({"id": conv["id"]})
    assert stub.calls >= 2  # the query embedding and the answer call went to the provider
    rows = [json.loads(line) for line in (tmp_path / "cost_ledger.jsonl").read_text().splitlines() if line.strip()]
    live_rows = [r for r in rows if r.get("mode") == "live"]
    assert live_rows and sum(r["cost_usd"] for r in live_rows) > 0  # on disk, priced
    spent = app.status()["budget"]["spent_usd"]
    assert spent > before  # the page's budget readout moves
    elsewhere = BudgetLedger(ceiling_usd=5.0, path=tmp_path / "cost_ledger.jsonl")  # an evaluation's ledger
    assert abs(elsewhere.total_cost_usd - spent) < 1e-9
    assert conv["turns"][0]["usage"].get("cost_usd", 0) > 0  # and the turn shows what it cost


def test_a_case_run_by_an_evaluation_replays_in_the_web_page(settings, index, tmp_path):
    from supportdesk.eval.cases import load_cases
    from supportdesk.registry import make_pipeline

    cases = load_cases(*settings.eval.case_files, split="dev")
    if not cases:
        import pytest

        pytest.skip("no development cases in this checkout")
    ev = next(c for c in cases if "\r\n" in c.input.body)  # one a browser would change
    live = dataclasses.replace(
        settings, run=dataclasses.replace(settings.run, mode="live"),
        paths=dataclasses.replace(settings.paths, ledger_path=tmp_path / "cost_ledger.jsonl",
                                  cache_path=tmp_path / "cache.sqlite", artifacts_dir=tmp_path / "artifacts"),
        budget=dataclasses.replace(settings.budget, ceiling_usd=5.0, reserve_usd=0.0))
    stub = LiveStub(index.dimension)
    clients = build_clients(live, transport=stub)
    make_pipeline("baseline", "dense", index, clients, live).run(ev.input)  # the evaluation: pays, fills the cache
    paid_calls = stub.calls
    app = App(live, clients=clients, index=index, store_path=tmp_path / "c.sqlite")
    dev = app.dev_case(ev.case_id)
    browser = lambda text: (text or "").replace("\r\n", "\n")
    conv = app.conversation_new({"title": dev["title"], "body": browser(dev["body"]), "created_at": dev["created_at"],
                                 "history": [{**m, "body": browser(m["body"])} for m in dev["history"]],
                                 "environment": dev["environment"], "pipeline": "baseline",
                                 "source": {"kind": "dev", "case_id": ev.case_id}})
    conv = app.conversation_turn({"id": conv["id"]})
    usage = conv["turns"][0]["usage"]
    assert stub.calls == paid_calls  # no provider call: everything came from the cache
    assert usage.get("cost_usd", 0) == 0 and usage.get("cache_hits", 0) > 0
    assert usage.get("replayed_cost_usd", 0) > 0  # and the page says what was replayed
