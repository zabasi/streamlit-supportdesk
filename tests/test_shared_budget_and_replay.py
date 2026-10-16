"""The ledger counts spending by other processes, and the web UI's requests replay from the cache."""
import dataclasses

import pytest

from supportdesk.llm.accounting import BudgetLedger
from supportdesk.web.conversations import content_case_id
from supportdesk.web.server import App


def test_a_ledger_sees_what_another_process_spent(tmp_path):
    path = tmp_path / "ledger.jsonl"
    web, cli = BudgetLedger(ceiling_usd=1.0, path=path), BudgetLedger(ceiling_usd=1.0, path=path)
    cli.record(stage="answer", mode="live", cost_usd=0.30)
    assert web.total_cost_usd == pytest.approx(0.30) and web.remaining_usd() == pytest.approx(0.70)
    assert web.would_exceed(0.80)  # the guard counts the other process's spending
    web.record(stage="answer", mode="live", cost_usd=0.10)
    assert web.total_cost_usd == pytest.approx(0.40) and len(web.rows) == 2  # its own line not counted twice
    assert cli.total_cost_usd == pytest.approx(0.40)


def test_a_half_written_line_waits_until_it_is_complete(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(ceiling_usd=1.0, path=path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"cost_usd": 0.2, "stage": "x"')  # another process is still writing this line
    assert ledger.total_cost_usd == 0
    with path.open("a", encoding="utf-8") as f:
        f.write("}\n")
    assert ledger.total_cost_usd == pytest.approx(0.2)


def test_case_ids_are_stable_for_the_same_report_and_differ_otherwise():
    r = {"title": "Value lost", "body": "after a rerun", "created_at": None, "environment": {}}
    assert content_case_id(r, []) == content_case_id(dict(r), [])
    assert content_case_id(r, []) != content_case_id({**r, "body": "after a click"}, [])
    assert content_case_id(r, []) != content_case_id(r, [{"role": "user", "body": "hi"}])


@pytest.fixture()
def app(settings, index, clients, tmp_path):
    controlled = dataclasses.replace(settings, pipeline=dataclasses.replace(
        settings.pipeline, name="advanced", scope_gate="off", generator="controlled_v2"))
    return App(settings, clients=clients, index=index, store_path=tmp_path / "c.sqlite",
               settings_for=lambda name: controlled if name == "advanced" else settings)


def test_the_same_report_sends_the_same_request_so_it_can_replay(app):
    report = {"title": "Value lost", "body": "st.text_input loses its value after a rerun.", "pipeline": "advanced"}
    requests = []
    for _ in range(2):
        conv = app.conversation_turn({"id": app.conversation_new(report)["id"]})
        generator = next(iter(app._pipelines.values())).generator
        requests.append(generator.chat.seen[-1])
    assert requests[0] == requests[1]  # byte-identical: the cache can replay the second
    assert conv["case_id"].startswith("web-")


def test_an_unchanged_development_case_keeps_its_own_id(app):
    cases = app.dev_cases()
    if not cases:
        pytest.skip("no development cases in this checkout")
    dev = app.dev_case(cases[0]["id"])
    payload = {"title": dev["title"], "body": dev["body"], "created_at": dev["created_at"], "history": dev["history"],
               "environment": dev["environment"], "pipeline": "advanced", "source": {"kind": "dev", "case_id": dev["id"]}}
    assert app.conversation_new(payload)["case_id"] == dev["id"]
    edited = app.conversation_new({**payload, "body": dev["body"] + "\nOne more detail."})
    assert edited["case_id"].startswith("web-") and edited["source"]["case_id"] == dev["id"]
