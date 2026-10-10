import http.client
import json
import threading
from pathlib import Path

import pytest

from supportdesk.web.server import App, create_server
from tests.helpers import override

STATIC = Path(__file__).resolve().parent.parent / "src" / "supportdesk" / "web" / "static"
GOOD = {"title": "Session state empty after refresh",
        "body": "st.session_state value is gone after F5. Streamlit 1.31.1, Chrome."}


@pytest.fixture()
def server(settings, index, clients):
    app = App(settings, clients=clients, index=index)
    httpd = create_server(app, port=0)
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    yield httpd.server_address[1], app
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    if body is not None and raw is None:
        raw = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=raw, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def test_static_files_are_served_with_security_headers(server):
    port, _ = server
    for path, marker in (("/", b"supportdesk"), ("/app.js", b"renderResult"), ("/style.css", b"--accent")):
        status, headers, data = call(port, "GET", path)
        assert status == 200 and marker in data
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"


def test_status_reports_mode_budget_and_registry(server):
    port, _ = server
    status, _, data = call(port, "GET", "/api/status")
    info = json.loads(data)
    assert status == 200 and info["settings"]["mode"] == "offline"
    assert "baseline" in info["pipelines"] and {"dense", "bm25", "hybrid"} <= set(info["retrievers"])
    assert info["index"]["chunk_count"] > 0 and info["budget"]["spent_usd"] == 0
    assert "sk-" not in json.dumps(info)


def test_ask_returns_result_with_evidence_and_usage(server):
    port, _ = server
    status, _, data = call(port, "POST", "/api/ask", GOOD)
    out = json.loads(data)
    assert status == 200 and out["result"]["outcome"] == "answer"
    labels = {e["label"] for e in out["evidence"]}
    assert labels and out["evidence"][0]["text"] and out["evidence"][0]["url"]
    assert any(e["cited"] for e in out["evidence"])
    assert out["retriever"] == "dense" and out["budget"]["spent_usd"] == 0  # baseline's own
    assert out["result"]["usage"]["wall_seconds"] >= 0


@pytest.mark.parametrize("payload, fragment", [
    ({}, "title is required"),
    ({"title": "t", "body": 5}, "body must be a string"),
    ({"title": "x" * 301}, "longer than"),
    ({"title": "t", "created_at": "yesterday"}, "ISO timestamp"),
    ({"title": "t", "history": "not a list"}, "history must be"),
    ({"title": "t", "pipeline": "magic"}, "unknown pipeline"),
    ({"title": "t", "retriever": "magic"}, "its own retriever"),
    ({"title": "t", "retriever": "bm25"}, "use --retriever on the command line"),
])
def test_bad_input_is_a_400_with_a_reason(server, payload, fragment):
    port, _ = server
    status, _, data = call(port, "POST", "/api/ask", payload)
    assert status == 400 and fragment in json.loads(data)["error"]


def test_non_json_and_wrong_content_type_and_size(server):
    port, app = server
    assert call(port, "POST", "/api/ask", raw=b"{nope", headers={"Content-Type": "application/json"})[0] == 400
    assert call(port, "POST", "/api/ask", raw=b"title=x",
                headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    big = json.dumps({"title": "t", "body": "x" * (app.settings.web.max_request_bytes + 1)}).encode()
    assert call(port, "POST", "/api/ask", raw=big, headers={"Content-Type": "application/json"})[0] == 413
    assert call(port, "GET", "/nope")[0] == 404 and call(port, "POST", "/api/other", GOOD)[0] == 404


def test_other_websites_and_dns_rebinding_are_refused(server):
    port, app = server
    evil_origin = {"Origin": "http://evil.example.com"}
    assert call(port, "POST", "/api/ask", GOOD, headers=evil_origin)[0] == 403
    assert call(port, "GET", "/api/status", headers=evil_origin)[0] == 403
    assert call(port, "POST", "/api/ask", GOOD, headers={"Origin": "null"})[0] == 403
    assert call(port, "POST", "/api/ask", GOOD, headers={"Host": "evil.example.com"})[0] == 403
    same = {"Origin": f"http://127.0.0.1:{port}"}
    assert call(port, "POST", "/api/ask", GOOD, headers=same)[0] == 200
    assert app.clients.ledger.total_cost_usd == 0


def test_server_only_binds_to_loopback(index, settings, clients):
    from supportdesk.config import ConfigError
    app = App(settings, clients=clients, index=index)
    with pytest.raises(ConfigError, match="loopback"):
        create_server(app, host="0.0.0.0")


def test_internal_errors_do_not_leak(server, monkeypatch, capsys):
    port, app = server

    def boom(payload):
        raise RuntimeError("db exploded with key sk-abcdefghijklmnop1234")
    monkeypatch.setattr(app, "ask", boom)
    status, _, data = call(port, "POST", "/api/ask", GOOD)
    assert status == 500 and "sk-" not in data.decode() and "exploded" not in data.decode()
    assert "sk-abcdef" not in capsys.readouterr().err


def test_concurrent_requests_are_serialised(server):
    port, app = server
    before, results = len(app.clients.ledger.rows), []

    def worker():
        results.append(call(port, "POST", "/api/ask", GOOD)[0])
    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200] * 6
    assert len(app.clients.ledger.rows) - before == 12  # 6 x (query embed + chat), none lost


def test_page_has_no_external_resources_and_no_innerhtml_writes():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()
    for text in (html, js, css):
        assert "http://" not in text.replace("http://127", "") or "https?://" in text
        assert "cdn." not in text and "googleapis" not in text and "@import" not in text
    assert ".innerHTML" not in js and "insertAdjacentHTML" not in js and "eval(" not in js
    assert "<script src=\"/app.js\">" in html and "onclick" not in html


def test_status_exposes_the_configured_defaults_not_alphabetical_ones(server):
    port, _ = server
    info = json.loads(call(port, "GET", "/api/status")[2])
    assert info["defaults"] == {"pipeline": "baseline", "retriever": "dense"}
    from supportdesk.registry import PIPELINES
    assert info["pipelines"] == sorted(PIPELINES)
    assert {k: info["retriever_by_pipeline"][k] for k in ("advanced", "baseline")} == {"advanced": "dense", "baseline": "dense"}
    assert info["retrievers"][0] == "bm25"  # alphabetical first: the UI must not fall back to this


def test_the_page_runs_each_pipeline_with_its_own_retriever_only(index, clients, settings):
    from supportdesk.web.server import BadRequest
    app = App(override(settings, pipeline={"retriever": "bm25"}), clients=clients, index=index)
    assert app.ask({"title": "session state refresh", "body": "st.session_state lost"})["retriever"] == "bm25"
    assert app.ask({"title": "session state refresh", "retriever": "bm25"})["retriever"] == "bm25"
    with pytest.raises(BadRequest, match="use --retriever on the command line"):
        app.ask({"title": "session state refresh", "retriever": "hybrid"})


def test_each_pipeline_is_served_with_its_own_profile(index, clients, settings, monkeypatch):
    from supportdesk.pipelines.baseline import build_baseline
    from supportdesk.registry import PIPELINES
    monkeypatch.setitem(PIPELINES, "wide", build_baseline)

    def settings_for(name):
        if name == "wide":
            return override(settings, retrieval={"top_k": 3}, pipeline={"retriever": "hybrid"})
        return settings
    app = App(settings, clients=clients, index=index, settings_for=settings_for)
    base = app.ask({"title": "session state refresh", "body": "F5 lost", "pipeline": "baseline"})
    wide = app.ask({"title": "session state refresh", "body": "F5 lost", "pipeline": "wide"})
    assert base["applied"]["top_k"] == 6 and len(base["evidence"]) == 6
    assert wide["applied"]["top_k"] == 3 and len(wide["evidence"]) == 3
    assert wide["retriever"] == "hybrid"  # that pipeline's own default retriever
    assert app.status()["retriever_by_pipeline"]["wide"] == "hybrid"


def test_a_pipeline_that_needs_a_different_index_is_refused_with_instructions(
        index, clients, settings, monkeypatch):
    from supportdesk.pipelines.baseline import build_baseline
    from supportdesk.registry import PIPELINES
    from supportdesk.web.server import BadRequest
    monkeypatch.setitem(PIPELINES, "small", build_baseline)
    app = App(settings, clients=clients, index=index, settings_for=lambda name: (
        override(settings, chunking={"max_tokens": 100, "overlap_tokens": 10}) if name == "small" else settings))
    with pytest.raises(BadRequest, match=r"supportdesk index --pipeline small"):
        app.ask({"title": "t", "pipeline": "small"})
    assert app.ask({"title": "t", "pipeline": "baseline"})["result"]["outcome"] == "answer"
