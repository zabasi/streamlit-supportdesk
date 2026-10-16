"""Finished evaluation runs, viewed and continued in the web UI."""
import json
import threading

import pytest

from supportdesk.web.server import App, BadRequest, create_server
from tests.test_web import call


@pytest.fixture()
def app(settings, index, clients, tmp_path):
    import dataclasses

    s = dataclasses.replace(settings, paths=dataclasses.replace(settings.paths, artifacts_dir=tmp_path / "artifacts"))
    a = App(s, clients=clients, index=index, store_path=tmp_path / "c.sqlite")
    if not a._case_inputs():
        pytest.skip("no evaluation cases in this checkout")
    return a


def write_run(app, run_id, outcomes, kind="full", profile="configs/pipelines/candidate-1-dense-rerank-section.toml"):
    d = app.settings.paths.artifacts_dir / "runs" / run_id
    d.mkdir(parents=True)
    ids = list(app._case_inputs())[: len(outcomes)]
    chunk = app.index.chunks[0].chunk_id
    rows = [{"case_id": cid, "outcome": o, "user_reply": "Which Streamlit version?" if o == "request_info" else "Fix it so.",
             "citations": [chunk], "violations": [], "error": None, "category": "ambiguous",
             "trace": {"packet": [{"label": "S1", "chunk_id": chunk}]}, "retrieved": [{"chunk_id": chunk, "score": 0.5}],
             "maintainer_summary": {"problem": "p", "next_step": "n"}, "usage": {"cost_usd": 0.02}} for cid, o in zip(ids, outcomes)]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps({"kind": kind, "pipeline": "advanced", "mode": "live", "run_id": run_id,
                                                "retriever": "dense_doc_slots", "created_at": "2026-10-16T10:00:00Z",
                                                "settings": {"config_profile": profile}, "metrics": {"n_cases": len(rows)}}))
    return ids


def test_runs_lists_finished_runs_with_their_pipeline(app):
    write_run(app, "e2e-candidate-1", ["answer"])
    write_run(app, "ret-only", ["answer"], kind="retrieval")
    runs = app.runs()
    assert [r["run_id"] for r in runs] == ["e2e-candidate-1"]
    assert runs[0]["pipeline"] == "candidate-1-dense-rerank-section"


def test_importing_a_run_makes_saved_conversations_without_a_model_call(app, clients):
    ids = write_run(app, "e2e-x", ["answer", "request_info", "escalate"])
    before = clients.chat.calls
    result = app.import_run({"run_id": "e2e-x"})
    assert clients.chat.calls == before and result["imported"] == 3
    assert app.import_run({"run_id": "e2e-x"})["already_imported"] == 3  # not imported twice
    convs = {c["source"]["case_id"]: app.store.get(c["id"]) for c in app.store.listing()}
    assert [convs[i]["status"] for i in ids] == ["answer", "awaiting_reply", "escalate"]
    first = convs[ids[0]]
    assert first["case_id"] == ids[0] and first["pipeline"] == "candidate-1-dense-rerank-section"
    assert first["turns"][0]["imported_from"] == "e2e-x" and first["turns"][0]["evidence"][0]["label"] == "S1"
    assert first["report"]["title"]  # the case's own input


def test_an_imported_question_can_be_continued(app):
    ids = write_run(app, "e2e-q", ["request_info"])
    app.import_run({"run_id": "e2e-q"})
    conv = app.store.find(kind="run", run_id="e2e-q", case_id=ids[0])
    seen = []

    def run(case, pipeline, retriever, *, questions_used=None, keep_result=False):
        seen.append((pipeline, [m["role"] for m in case.history][-2:], questions_used, case.case_id))
        return {"result": {"outcome": "answer", "user_reply": "Done.", "citations": [], "violations": [],
                           "usage": {}, "error": None}, "evidence": [], "retriever": retriever, "applied": None}
    app._run = run
    conv = app.conversation_turn({"id": conv["id"], "reply": "1.31.1"})
    assert conv["status"] == "answer" and len(conv["turns"]) == 2
    assert seen == [("candidate-1-dense-rerank-section", ["assistant", "user"], 1, ids[0])]


@pytest.mark.parametrize("bad", ["../secrets", "a/b", ".hidden", "", "x" * 200, "nope"])
def test_unsafe_or_unknown_run_names_are_refused(app, bad):
    with pytest.raises(BadRequest):
        app.import_run({"run_id": bad})


def test_the_routes_over_http(app):
    write_run(app, "e2e-http", ["answer"])
    httpd = create_server(app, port=0)
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True).start()
    port = httpd.server_address[1]
    try:
        assert [r["run_id"] for r in json.loads(call(port, "GET", "/api/runs")[2])["runs"]] == ["e2e-http"]
        status, _, data = call(port, "POST", "/api/runs/import", {"run_id": "e2e-http"})
        assert status == 200 and json.loads(data)["imported"] == 1
        assert call(port, "POST", "/api/runs/import", {"run_id": "../x"})[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_test_split_run_imports_from_the_case_files_it_used(app, tmp_path):
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "eval"
    test_files = tmp_path / "test-eval"
    test_files.mkdir()
    shutil.copy(root / "development_inputs.jsonl", test_files / "test_inputs.jsonl")
    shutil.copy(root / "development_rubrics.jsonl", test_files / "test_rubrics.jsonl")
    from supportdesk.eval.cases import load_cases

    assert {c.split for c in load_cases(str(test_files))} == {"test"}
    ids = write_run(app, "test-final", ["answer", "escalate"])
    summary_path = app.settings.paths.artifacts_dir / "runs" / "test-final" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["case_files"] = [str(test_files)]  # the run read the test split, not the default files
    summary_path.write_text(json.dumps(summary))
    result = app.import_run({"run_id": "test-final"})
    assert result["imported"] == 2 and result["without_case_input"] == 0
    assert {app.store.get(c["id"])["case_id"] for c in app.store.listing()} == set(ids)
