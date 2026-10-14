import json

from supportdesk.pipelines.scope_gate import ScopeGatedPipeline
from supportdesk.pipelines.stages.scope_classifier import LLMScopeClassifier, RuleScopeClassifier
from supportdesk.types import CaseInput


def case(title, body=""):
    return CaseInput(case_id="scope", title=title, body=body)


def test_rule_classifier_keeps_streamlit_prose_and_code_and_rejects_greetings():
    clf = RuleScopeClassifier()
    assert clf(case("Session state resets", "My st.session_state value disappears")).in_scope
    assert clf(case("Why does this fail?", "```python\nimport streamlit as st\nst.button('Go')\n```" )).in_scope
    assert not clf(case("hello")).in_scope
    # Conservative fail-open: generic/ambiguous text is not silently discarded.
    assert clf(case("My app crashes after clicking submit")).in_scope


def test_llm_classifier_parses_binary_json_and_fails_open(clients, tmp_path):
    prompt = tmp_path / "scope.txt"
    prompt.write_text("classify", encoding="utf-8")
    clf = LLMScopeClassifier(clients, prompt)

    clients.chat.script[clients.chat.calls + 1] = json.dumps(
        {"label": "OUT_OF_SCOPE", "reason": "general Python question"})
    d = clf(case("How do I sort a Python list?"))
    assert (d.label, d.method) == ("OUT_OF_SCOPE", "llm")

    clients.chat.script[clients.chat.calls + 1] = "not json"
    d = clf(case("unclear report"))
    assert (d.label, d.method) == ("IN_SCOPE", "fail_open")


def _gated(index, clients, settings, classifier):
    from supportdesk.pipelines.advanced import build_advanced
    from supportdesk.retrieval.retriever import DenseRetriever

    advanced = build_advanced(DenseRetriever(index, clients.embedding), clients, settings)
    return ScopeGatedPipeline(
        name="scope_gate", clients=clients, settings=settings, classifier=classifier, extra_nodes=(),
        **{stage: getattr(advanced, stage) for stage in (
            "query", "retriever", "reranker", "packet", "generator", "validator", "summarizer"
        )},
    )


def test_scope_gate_does_not_retrieve_when_rejected(index, clients, settings):
    from supportdesk.pipelines.stages.scope_classifier import ScopeDecision

    class Reject:
        def __call__(self, case):
            return ScopeDecision("OUT_OF_SCOPE", "unrelated", "rule")

    gated = _gated(index, clients, settings, Reject())
    before_embed = clients.embedding.calls
    result = gated.run(case("What is the capital of France?"))
    assert result.outcome == "request_info" and result.retrieved == []
    assert result.trace["extra"]["scope"]["label"] == "OUT_OF_SCOPE"
    assert clients.embedding.calls == before_embed

    r = gated.retrieve(case("hello"))
    assert r.hits == [] and r.queries == [] and r.packet.chunks == []
    assert clients.embedding.calls == before_embed


def test_scope_gate_in_scope_delegates_to_advanced(index, clients, settings):
    from supportdesk.pipelines.stages.scope_classifier import ScopeDecision

    class Accept:
        def __call__(self, case):
            return ScopeDecision("IN_SCOPE", "streamlit", "rule")

    gated = _gated(index, clients, settings, Accept())
    result = gated.run(case("Session state lost", "st.session_state disappears after rerun"))
    assert result.pipeline == "scope_gate" and result.retrieved
    assert result.trace["extra"]["scope"]["label"] == "IN_SCOPE"
    assert result.trace["stages"]["classifier"] == "Accept"


def test_scope_gate_can_be_embedded_in_case_workflow(index, clients, settings, tmp_path):
    """Regression: registry pipelines are also used by CaseWorkflow, which requires add_to()."""
    from supportdesk.casework import CaseWorkflow, Tracker
    from supportdesk.pipelines.stages.scope_classifier import ScopeDecision

    class Reject:
        def __call__(self, case):
            return ScopeDecision("OUT_OF_SCOPE", "unrelated", "rule")

    gated = _gated(index, clients, settings, Reject())
    tracker = Tracker(tmp_path / "tracker.sqlite")
    CaseWorkflow(gated, tracker, tmp_path / "checkpoints.sqlite")


def test_scope_classifier_eval_matrix_has_balanced_boundary_cases():
    from pathlib import Path

    rows = [json.loads(line) for line in Path("data/eval/scope_classifier_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 30
    assert sum(r["label"] == "IN_SCOPE" for r in rows) == 15
    assert sum(r["label"] == "OUT_OF_SCOPE" for r in rows) == 15

    by_id = {r["case_id"]: r for r in rows}
    assert by_id["SCOPE-002"]["label"] == "OUT_OF_SCOPE"  # generic Python
    assert by_id["SCOPE-013"]["label"] == "IN_SCOPE"      # implicit Streamlit wording
    assert by_id["SCOPE-020"]["label"] == "IN_SCOPE"      # code-only Streamlit
    assert by_id["SCOPE-026"]["label"] == "OUT_OF_SCOPE"  # competing framework
    assert by_id["SCOPE-021"]["label"] == "IN_SCOPE"      # historical version still in scope


def test_scope_classifier_benchmark_has_120_balanced_cases_and_hard_categories():
    from pathlib import Path

    rows = [json.loads(line) for line in Path("data/eval/scope_classifier_benchmark.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 120
    assert sum(r["label"] == "IN_SCOPE" for r in rows) == 60
    assert sum(r["label"] == "OUT_OF_SCOPE" for r in rows) == 60
    assert len({r["case_id"] for r in rows}) == 120
    categories = {r["category"] for r in rows}
    assert {
        "explicit_api", "implicit_streamlit", "code_only", "historical_version",
        "generic_python", "other_framework", "ambiguous_non_streamlit",
    } <= categories
    assert sum(r["difficulty"] == "hard" for r in rows) >= 35


def test_registry_advanced_includes_llm_scope_gate(index, clients, settings):
    import dataclasses

    from supportdesk.registry import make_pipeline
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, name="advanced"))
    pipeline = make_pipeline("advanced", "dense_doc_slots", index, clients, s)
    assert isinstance(pipeline, ScopeGatedPipeline)
    assert type(pipeline.classifier).__name__ == "LLMScopeClassifier"
    assert pipeline.name == "advanced"

    # Regression: the gate wraps whatever advanced is today; it must not replace its retrieval design.
    from supportdesk.pipelines.advanced import build_advanced
    from supportdesk.retrieval.retriever import DocSlotsRetriever
    raw = build_advanced(DocSlotsRetriever(index, clients.embedding, doc_slots=s.retrieval.doc_slots), clients, s)
    for stage in ("query", "retriever", "reranker", "packet", "generator", "validator", "summarizer"):
        assert type(getattr(pipeline, stage)) is type(getattr(raw, stage))


def test_the_scope_prompt_is_a_setting(index, clients, settings, tmp_path):
    import dataclasses

    from supportdesk.pipelines.scope_gate import build_scope_gate
    from supportdesk.retrieval.retriever import DenseRetriever
    prompt = tmp_path / "scope.txt"
    prompt.write_text("custom scope prompt")
    s = dataclasses.replace(settings, prompts=dataclasses.replace(settings.prompts, scope=prompt))
    assert build_scope_gate(DenseRetriever(index, clients.embedding), clients, s).classifier.prompt == "custom scope prompt"
    assert str(settings.prompts.scope).endswith("configs/prompts/scope_system.txt")  # the default
