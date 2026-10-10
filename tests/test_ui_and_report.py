import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from supportdesk.eval.report import render_report
from supportdesk.eval.runner import run_eval
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.retrieval.retriever import DenseRetriever
from supportdesk.web.server import App, create_server
from tests.test_web import GOOD, call

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "ui" / "render_harness.js"
APP_JS = ROOT / "src" / "supportdesk" / "web" / "static" / "app.js"


def _render(tmp_path, response: dict) -> dict:
    path = tmp_path / "response.json"
    path.write_text(json.dumps(response))
    out = subprocess.run(["node", str(HARNESS), str(APP_JS), str(path)], capture_output=True, check=False,
                         text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_renders_a_real_server_response_as_text_only(tmp_path, settings, index, clients):
    httpd = create_server(App(settings, clients=clients, index=index), port=0)
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True).start()
    try:
        _, _, data = call(httpd.server_address[1], "POST", "/api/ask", GOOD)
    finally:
        httpd.shutdown()
        httpd.server_close()
    response = json.loads(data)
    page = _render(tmp_path, response)
    assert page["hidden"] is False and page["emptyHidden"] is True
    visible = page["visible"]
    assert "null" not in visible and "undefined" not in visible
    assert "[object" not in visible  # no stringified nodes/arrays leaking into the page
    assert "top_k 6" in visible and "6 chunks in prompt" in visible  # what the pipeline actually applied
    assert "Answer" in page["text"] and "Maintainer summary" in page["text"]
    assert "Evidence shown to the model" in page["text"] and "operating_system" in page["text"]
    assert all(h.startswith("http") for h in page["hrefs"]) and page["hrefs"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_does_not_execute_hostile_content_or_unsafe_links(tmp_path):
    evil = "<img src=x onerror=alert(1)><script>alert(2)</script>"
    response = {"pipeline": "baseline", "retriever": "dense", "budget": {}, "evidence": [
        {"label": "S1", "chunk_id": "c", "source_id": "s", "type": "issue", "title": evil,
         "url": "javascript:alert(3)", "version": None, "state": None, "as_of": None,
         "score": 0.5, "text": evil, "cited": True}],
        "result": {"outcome": "answer", "mode": "offline", "user_reply": evil, "error": None,
                   "violations": [], "usage": {}, "citations": ["c"], "maintainer_summary": {
                       "problem": evil, "environment": {"known": {"k": evil}, "unknown": []},
                       "known_facts": [{"statement": evil, "evidence": ["c"]}],
                       "open_questions": [evil], "next_step": evil,
                       "sources": [{"chunk_id": "c", "type": "issue", "url": "javascript:alert(4)"}]}}}
    page = _render(tmp_path, response)  # the harness throws if innerHTML is ever assigned
    assert evil in page["text"]  # shown as visible text, not interpreted
    assert page["hrefs"] == []  # javascript: URLs never become links


def _make_runs(index, clients, settings, cases, tmp_path):
    summaries = []
    for name, retr in (("a", "dense"),):
        pipe = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)
        run_eval(pipe, cases, clients, settings, run_id=name, index_manifest=index.manifest,
                 out_root=tmp_path)
        summaries.append(tmp_path / name)
    return summaries


def test_report_compares_runs_and_escapes_everything(index, clients, settings, cases, tmp_path):
    runs = _make_runs(index, clients, settings, cases, tmp_path)
    # a second run with hostile text in its metadata
    second = tmp_path / "b"
    second.mkdir()
    summary = json.loads((runs[0] / "summary.json").read_text())
    summary.update(run_id="<script>alert(1)</script>", pipeline="advanced")
    (second / "summary.json").write_text(json.dumps(summary))
    (second / "results.jsonl").write_text((runs[0] / "results.jsonl").read_text())
    html = render_report([runs[0], second])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Not all runs are live" in html  # offline runs are labelled as such
    for needle in ("Recall@3", "by category", "answerable", "ambiguous", "c-001", "Cost (real spend)"):
        assert needle.lower() in html.lower()
    assert "http://" not in html and "<link" not in html  # self-contained


def test_report_cli_writes_a_file(index, clients, settings, cases, tmp_path):
    from supportdesk.cli import main
    _make_runs(index, clients, settings, cases, tmp_path / "runs")
    out = tmp_path / "r.html"
    code = main(["report", "--artifacts-dir", str(tmp_path), "a", "--output", str(out)])
    assert code == 0 and out.read_text().startswith("<!doctype html>")


def _status_page(tmp_path, status: dict, switch_to: str) -> dict:
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    out = subprocess.run(["node", str(HARNESS), str(APP_JS), str(path), "status", switch_to],
                         capture_output=True, text=True, timeout=20, check=False)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_preselects_the_configured_defaults_and_follows_the_pipelines_retriever(tmp_path):
    status = {"settings": {"mode": "offline"}, "index": {"chunk_count": 1, "embedding_model": "m"},
              "budget": {}, "pipelines": ["advanced", "baseline"],
              "retrievers": ["bm25", "dense", "hybrid"],  # alphabetical: bm25 would be first
              "defaults": {"pipeline": "baseline", "retriever": "dense"},
              "retriever_by_pipeline": {"baseline": "dense", "advanced": "hybrid"},
              "limits": {"title": 300, "body": 8000, "created_at": 40}}
    page = _status_page(tmp_path, status, switch_to="advanced")
    assert page["first"] == {"pipeline": "baseline", "retriever": "dense"}  # not "advanced"/"bm25"
    assert page["afterSwitch"] == "hybrid"  # switching pipelines brings that pipeline's retriever
