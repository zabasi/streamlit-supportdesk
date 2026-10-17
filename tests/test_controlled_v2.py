"""The controlled-routing v2 generator: the vendored package, the gateway transport and the pipeline stage."""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.packet import Packet, assemble_packet
from supportdesk.pipelines.stages.controlled import MODEL, ControlledGenerator
from supportdesk.pipelines.stages.defaults import SingleCallGenerator
from supportdesk.pipelines.stages.protocols import GenerationFailed
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk
from tests.test_budget_and_gateway import StubTransport, make_gateway

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/supportdesk/generation/controlled_v2"
HANDOFF = ROOT / "docs/generation/controlled-v2"
CASE = CaseInput(case_id="C-1", title="Value lost", body="The widget value is lost after a rerun.",
                 environment={"streamlit_version": "1.40.0"})


def _env():
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


# 1. the vendored package is exactly what was delivered

def test_the_runtime_package_is_the_delivered_one_byte_for_byte():
    delivered = sorted(p.relative_to(HANDOFF / "supportdesk_generation") for p in
                       (HANDOFF / "supportdesk_generation").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert delivered
    for rel in delivered:
        assert (RUNTIME / rel).read_bytes() == (HANDOFF / "supportdesk_generation" / rel).read_bytes(), rel


def test_the_frozen_files_match_the_packages_own_manifest():
    from supportdesk.generation.controlled_v2 import generation

    assert len(generation.rows()) == 15  # rows() checks every frozen file's sha256 before reading


def test_the_delivered_tests_and_handoff_verifier_pass():
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], check=False,
                           cwd=HANDOFF, env=_env(), capture_output=True, text=True, timeout=120)
    assert tests.returncode == 0, tests.stderr[-2000:]
    assert "OK" in tests.stderr
    verify = subprocess.run([sys.executable, "verify_handoff.py"], check=False, cwd=HANDOFF, env=_env(),
                            capture_output=True, text=True, timeout=120)
    assert verify.returncode == 0, verify.stderr[-2000:]
    assert json.loads(verify.stdout)["files_verified"] == 29


# 2. the gateway reports how an answer ended

class Finishing(StubTransport):
    def chat(self, **kw):
        out = super().chat(**kw)
        return {**out, "finish": {"finish_reason": "length", "refusal": None}}


def test_the_gateway_reports_finish_reason_live_and_from_the_cache(tmp_path):
    gw, ledger = make_gateway(tmp_path, transport=Finishing())
    messages = [{"role": "user", "content": "hello"}]
    _, live = gw.complete(messages)
    _, cached = gw.complete(messages)
    assert live["finish_reason"] == "length" and cached["finish_reason"] == "length"
    assert cached["mode"] == "cache"
    assert [r["mode"] for r in ledger.rows] == ["live", "cache"]  # recorded as before
    assert not any("finish_reason" in r or "refusal" in r for r in ledger.rows)  # the ledger's fields unchanged


def test_a_transport_without_finish_details_still_works(tmp_path):
    gw, _ = make_gateway(tmp_path)  # the stub returns text and usage only, as before
    text, usage = gw.complete([{"role": "user", "content": "hi"}])
    assert text == '{"ok": true}' and "finish_reason" not in usage


# 3. the stage

def packet():
    chunks = [Chunk(chunk_id=f"doc:{n}#0", source_id=f"doc:{n}", source_type="documentation", title=f"Page {n}",
                    text=f"Widget values are kept across reruns when a key is set ({n}).", start_char=0,
                    end_char=60) for n in range(2)]
    return assemble_packet([Hit(c, 1.0, i + 1, "dense") for i, c in enumerate(chunks)], max_chunks=4,
                           max_tokens=4000)


def assessment(case_id="C-1", evidence="S1"):
    return {"case_id": case_id, "main_need": "Keep the widget value", "reported_observation_refs": ["R1"],
            "required_human_review": None, "information_gaps": [],
            "answer_candidate": {"message": f"Set a key on the widget [{evidence}].", "evidence_refs": [evidence]},
            "internal_investigation": None, "workflow_findings": []}


def generator(clients, settings, script):
    gen = ControlledGenerator(clients, settings)
    gen.chat.script[1] = script
    return gen


def test_an_answer_with_citations_mapped_to_chunk_ids(clients, settings):
    gen = generator(clients, settings, json.dumps(assessment()))
    seen = {}
    original = gen.chat.complete

    def spy(messages, **kw):
        seen.update(kw)
        return original(messages, **kw)

    gen.chat.complete = spy
    answer = gen(CASE, {}, packet())
    assert answer.outcome == "answer" and answer.citations == ["doc:0#0"]
    assert seen["extra"]["response_format"]["json_schema"]["strict"] is True
    assert seen["extra"]["reasoning_effort"] == "medium"
    assert answer.attempts and "controlled_v2" in answer.attempts[0]
    saved = answer.attempts[0]["controlled_v2"]  # what results.jsonl keeps for the case
    assert saved["assessment"] == assessment()  # the model's structured assessment, as it wrote it
    assert "maintainer_summary" in saved


def test_a_cut_off_answer_is_an_error_not_an_escalation(clients, settings):
    gen = generator(clients, settings, "truncated:" + json.dumps(assessment()))
    with pytest.raises(GenerationFailed, match="No complete usable assessment") as failed:
        gen(CASE, {}, packet())
    assert failed.value.attempts[0]["usage"]["finish_reason"] == "length"  # the call's usage is kept


def test_an_invalid_assessment_is_an_error_for_that_case_only(clients, settings):
    gen = generator(clients, settings, json.dumps({"case_id": "C-1"}))  # missing fields
    with pytest.raises(GenerationFailed, match="missing or extra fields"):
        gen(CASE, {}, packet())


def test_another_model_is_refused(clients, settings):
    gen = ControlledGenerator(clients, settings)
    with pytest.raises(ValueError, match=MODEL):
        gen.complete({"model": "gpt-4o-mini", "messages": [], "response_format": {}, "reasoning_effort": "medium"})


# 4. the wiring

def test_advanced_uses_it_when_selected_and_the_baseline_never_does(index, clients, settings):
    on = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, name="advanced",
                                                                    scope_gate="off", generator="controlled_v2"))
    assert isinstance(make_pipeline("advanced", "dense_doc_slots", index, clients, on).generator, ControlledGenerator)
    base = dataclasses.replace(on, pipeline=dataclasses.replace(on.pipeline, name="baseline"))
    assert isinstance(make_pipeline("baseline", "dense", index, clients, base).generator, SingleCallGenerator)
    off = dataclasses.replace(on, pipeline=dataclasses.replace(on.pipeline, generator="single_call"))
    assert isinstance(make_pipeline("advanced", "dense_doc_slots", index, clients, off).generator, SingleCallGenerator)


def test_the_generator_setting_is_checked():
    assert Settings.load(env={}).pipeline.generator == "single_call"
    with pytest.raises(ConfigError, match="pipeline.generator"):
        Settings.load(env={}, overrides=["pipeline.generator=critic"])


def test_the_extracted_environment_is_used_only_when_the_case_has_none(clients, settings, monkeypatch):
    """An app case carries no reported environment, so the extracted one is passed; an evaluation case
    keeps its own, exactly as in the package's verified run."""
    from supportdesk.pipelines.stages import controlled

    seen = []
    monkeypatch.setattr(controlled, "generate_from_packet",
                        lambda case, packet, complete, **_: seen.append(case.environment) or (_ for _ in ()).throw(
                            ValueError("stop here")))
    gen = controlled.ControlledGenerator(clients, settings)
    app_case = CaseInput(case_id="a", title="t", body="b")
    eval_case = CaseInput(case_id="e", title="t", body="b", environment={"streamlit_version": "1.13.0"})
    for case in (app_case, eval_case):
        with pytest.raises(GenerationFailed):
            gen(case, {"streamlit_version": "1.40.0", "python_version": "3.12"}, Packet())
    assert seen == [{"streamlit_version": "1.40.0", "python_version": "3.12"}, {"streamlit_version": "1.13.0"}]
