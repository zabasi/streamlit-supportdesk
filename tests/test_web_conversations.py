"""Saved, multi-turn conversations in the web UI: store, turns, limits, downloads and routes."""
import dataclasses
import json
import threading

import pytest

from supportdesk.pipelines.stages.conversation import QUESTIONS_USED
from supportdesk.web.conversations import MAX_FOLLOWUPS, MAX_TURNS, report_html, summary_markdown
from supportdesk.web.server import App, BadRequest, create_server
from tests.test_web import call

NEW = {"title": "Value lost <script>alert(1)</script>", "body": "st.text_input loses its value after a rerun.",
       "pipeline": "baseline"}


@pytest.fixture()
def app(settings, index, clients, tmp_path):
    return App(settings, clients=clients, index=index, store_path=tmp_path / "conv.sqlite")


def scripted(app, outcomes, seen=None):
    """Replace the pipeline run with scripted outcomes; record what each turn was given."""
    queue = list(outcomes)

    def run(case, pipeline, retriever, *, questions_used=None, keep_result=False):
        if seen is not None:
            seen.append({"history": [dict(m) for m in case.history], "questions_used": questions_used})
        outcome = queue.pop(0)
        reply = {"answer": "Here is the fix.", "request_info": "Which Streamlit version?",
                 "escalate": "This needs a maintainer."}.get(outcome)
        result = {"outcome": outcome, "user_reply": reply, "citations": [], "violations": [],
                  "usage": {"cost_usd": 0.001}, "error": "boom" if outcome == "error" else None,
                  "maintainer_summary": {"problem": "value lost", "next_step": "check the key"}}
        return {"result": result, "evidence": [], "retriever": retriever, "applied": None}
    app._run = run


def test_creating_a_conversation_makes_no_model_call(app, clients):
    before = clients.chat.calls
    conv = app.conversation_new(NEW)
    assert clients.chat.calls == before and conv["status"] == "new" and conv["turns"] == []
    assert [c["id"] for c in app.store.listing()] == [conv["id"]]


def test_a_clarification_waits_for_the_users_reply_and_passes_history_and_count(app):
    seen = []
    scripted(app, ["request_info", "answer"], seen)
    conv = app.conversation_new(NEW)
    conv = app.conversation_turn({"id": conv["id"]})
    assert conv["status"] == "awaiting_reply" and conv["questions_used"] == 1
    with pytest.raises(BadRequest, match="reply"):
        app.conversation_turn({"id": conv["id"]})  # a reply is required now
    conv = app.conversation_turn({"id": conv["id"], "reply": "1.31.1"})
    assert conv["status"] == "answer" and len(conv["turns"]) == 2
    assert [m["role"] for m in conv["messages"]] == ["assistant", "user", "assistant"]
    assert seen[0] == {"history": [], "questions_used": 0}
    assert seen[1]["questions_used"] == 1 and [m["role"] for m in seen[1]["history"]] == ["assistant", "user"]
    with pytest.raises(BadRequest, match="finished"):
        app.conversation_turn({"id": conv["id"], "reply": "more"})


def test_the_clarification_limit_and_the_turn_limit(app):
    scripted(app, ["request_info"] * (MAX_FOLLOWUPS + 1))
    conv = app.conversation_new(NEW)
    conv = app.conversation_turn({"id": conv["id"]})
    for n in range(MAX_FOLLOWUPS - 1):
        conv = app.conversation_turn({"id": conv["id"], "reply": f"reply {n}"})
    assert conv["questions_used"] == MAX_FOLLOWUPS and conv["status"] == "limit_reached"
    assert len(conv["turns"]) <= MAX_TURNS


def test_a_failed_turn_can_be_retried_by_hand(app):
    scripted(app, ["error", "answer"])
    conv = app.conversation_turn({"id": app.conversation_new(NEW)["id"]})
    assert conv["status"] == "error"
    assert app.conversation_turn({"id": conv["id"]})["status"] == "answer"


def test_the_real_pipeline_runs_a_turn_offline(app):
    conv = app.conversation_turn({"id": app.conversation_new(NEW)["id"]})
    assert len(conv["turns"]) == 1 and conv["turns"][0]["outcome"] in ("answer", "request_info", "escalate")
    assert conv["status"] != "new"


def test_downloads_escape_the_report_and_carry_the_summary(app):
    scripted(app, ["answer"])
    conv = app.conversation_turn({"id": app.conversation_new(NEW)["id"]})
    md = summary_markdown(conv)
    assert "## Maintainer summary" in md and "check the key" in md and "Here is the fix." in md
    page = report_html(app.store.all())
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_development_cases_expose_inputs_only(app):
    cases = app.dev_cases()
    if not cases:
        pytest.skip("no development cases in this checkout")
    case = app.dev_case(cases[0]["id"])
    assert set(case) == {"id", "title", "body", "created_at", "environment", "history"}
    assert "rubric" not in json.dumps(case) and "gold" not in json.dumps(case)


def test_the_question_count_reaches_the_controlled_answer_step(clients, settings, monkeypatch):
    from supportdesk.pipelines.packet import Packet
    from supportdesk.pipelines.stages import controlled
    from supportdesk.pipelines.stages.protocols import GenerationFailed
    from supportdesk.types import CaseInput

    seen = []

    def fake(case, packet, complete, questions_used=0):
        seen.append(questions_used)
        raise ValueError("stop here")
    monkeypatch.setattr(controlled, "generate_from_packet", fake)
    gen = controlled.ControlledGenerator(clients, settings)
    case = CaseInput(case_id="c", title="t", body="b")
    with pytest.raises(GenerationFailed):
        gen(case, {}, Packet())  # an evaluation: unset, so 0
    token = QUESTIONS_USED.set(3)
    try:
        with pytest.raises(GenerationFailed):
            gen(case, {}, Packet())
    finally:
        QUESTIONS_USED.reset(token)
    assert seen == [0, 3]


def test_the_routes_over_http(app):
    scripted(app, ["request_info", "answer"])
    httpd = create_server(app, port=0)
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True).start()
    port = httpd.server_address[1]
    try:
        status, _, data = call(port, "POST", "/api/conversations/new", NEW)
        conv = json.loads(data)
        assert status == 200 and conv["status"] == "new"
        status, _, data = call(port, "POST", "/api/conversations/turn", {"id": conv["id"]})
        assert status == 200 and json.loads(data)["status"] == "awaiting_reply"
        status, _, data = call(port, "POST", "/api/conversations/turn", {"id": conv["id"], "reply": "1.31"})
        assert json.loads(data)["status"] == "answer"
        status, _, data = call(port, "GET", "/api/conversations")
        assert [c["id"] for c in json.loads(data)["conversations"]] == [conv["id"]]
        status, headers, data = call(port, "GET", f"/api/conversation/summary?id={conv['id']}")
        assert status == 200 and headers["Content-Type"].startswith("text/markdown") and b"Maintainer summary" in data
        status, headers, data = call(port, "GET", "/api/report")
        assert status == 200 and headers["Content-Type"].startswith("text/html") and "Content-Security-Policy" in headers
        assert call(port, "GET", "/api/conversation?id=nope")[0] == 400
        assert call(port, "POST", "/api/conversations/delete", {"id": conv["id"]})[0] == 200
        assert json.loads(call(port, "GET", "/api/conversations")[2])["conversations"] == []
        status, _, _ = call(port, "POST", "/api/conversations/new", NEW, headers={"Origin": "http://evil.example"})
        assert status == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_earlier_messages_keep_their_roles(app):
    conv = app.conversation_new({**NEW, "history": [{"role": "assistant", "body": "Which version?"},
                                                    {"role": "user", "body": "1.31"}, "plain line"]})
    assert [m["role"] for m in conv["initial_history"]] == ["assistant", "user", "user"]
    with pytest.raises(BadRequest):
        app.conversation_new({**NEW, "history": [{"role": "system", "body": "x"}]})


def test_settings_unchanged_for_evaluation(settings):
    assert QUESTIONS_USED.get() is None  # nothing sets it outside a web conversation
    assert dataclasses.is_dataclass(settings)
