"""pipeline.scope_gate = combined: one model call decides the scope and writes the concept queries."""
import dataclasses
import json

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.scope_gate import ScopeGatedPipeline
from supportdesk.pipelines.stages.rewrite_query import QueryRewriter, with_report
from supportdesk.pipelines.stages.scope_rewrite import (
    CombinedQueryRewriter,
    CombinedScopeClassifier,
    ScopeRewrite,
    shared_scope_rewrite,
)
from supportdesk.registry import make_pipeline
from supportdesk.types import CaseInput

# A line from each system prompt, to tell the three kinds of request apart in what the fake chat saw.
COMBINED = "## Part 2: search queries"
REWRITE = "You help find the Streamlit documentation"
SCOPE = "Your ONLY task is to decide"

QUERIES = ["How widget values persist across reruns", "Session State and widget keys"]


def case(title, body="", case_id="c"):
    return CaseInput(case_id=case_id, title=title, body=body)


def advanced(index, clients, settings, gate):
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, name="advanced",
                                                                   scope_gate=gate))
    return make_pipeline("advanced", "dense_doc_slots", index, clients, s), s


def answer_next(clients, label, queries=()):
    """Script the fake chat's next answer as the combined call's JSON."""
    clients.chat.script[clients.chat.calls + 1] = json.dumps(
        {"label": label, "reason": "test", "queries": list(queries)})


def sent(clients, marker, since=0):
    """How many requests since ``since`` used the system prompt containing ``marker``."""
    return sum(1 for messages in clients.chat.seen[since:] if marker in messages[0]["content"])


def test_one_call_returns_the_scope_and_the_queries_and_is_remembered(clients, settings):
    shared = ScopeRewrite(clients, settings)
    report = case("Widget value lost", "st.text_input loses its value after st.button")
    answer_next(clients, "IN_SCOPE", QUERIES)
    decision, queries = shared(report)
    expected = QUERIES[: settings.retrieval.max_queries]  # the model's queries, up to retrieval.max_queries
    assert (decision.label, decision.method) == ("IN_SCOPE", "combined") and queries == expected
    calls = clients.chat.calls
    assert shared(report) == (decision, queries) and clients.chat.calls == calls  # read once, not twice
    cleaned = shared.clean(report, {})[0]
    rewriter = CombinedQueryRewriter(shared, settings)
    assert rewriter(report, {}) == with_report(cleaned, expected, settings.retrieval.keep_report_query)
    assert CombinedScopeClassifier(shared)(report) == decision and clients.chat.calls == calls


def test_unusable_output_fails_open_and_searches_with_the_report(clients, settings):
    shared = ScopeRewrite(clients, settings)
    report = case("unclear", "something is off")
    clients.chat.script[clients.chat.calls + 1] = "not json"
    decision, queries = shared(report)
    assert (decision.label, decision.method) == ("IN_SCOPE", "fail_open") and queries == []
    assert CombinedQueryRewriter(shared, settings)(report, {})[0] == shared.clean(report, {})[0]


def test_advanced_with_the_combined_gate_makes_one_call_before_search(index, clients, settings):
    pipeline, s = advanced(index, clients, settings, "combined")
    assert isinstance(pipeline, ScopeGatedPipeline) and isinstance(pipeline.classifier, CombinedScopeClassifier)
    assert isinstance(pipeline.query, CombinedQueryRewriter)
    report = case("Session state lost", "st.session_state disappears after rerun")
    answer_next(clients, "IN_SCOPE", QUERIES)
    before = len(clients.chat.seen)
    result = pipeline.run(report)
    # one combined request; neither the separate scope call nor the separate rewrite call, although the
    # reranker also reads the concept queries ("Mechanisms involved")
    assert (sent(clients, COMBINED, before), sent(clients, SCOPE, before), sent(clients, REWRITE, before)) == (1, 0, 0)
    assert result.retrieved and result.trace["extra"]["scope"]["method"] == "combined"
    searched = pipeline.retrieve(report).queries
    assert set(QUERIES[: s.retrieval.max_queries]) <= set(searched)  # the scripted queries were searched


def test_out_of_scope_stops_before_search(index, clients, settings):
    pipeline, _ = advanced(index, clients, settings, "combined")
    answer_next(clients, "OUT_OF_SCOPE")
    before_embed = clients.embedding.calls
    result = pipeline.run(case("What is the capital of France?"))
    assert result.outcome == "request_info" and result.retrieved == []
    assert result.trace["extra"]["scope"]["label"] == "OUT_OF_SCOPE"
    assert clients.embedding.calls == before_embed  # no search at all


def test_the_other_gates_keep_their_separate_rewrite_call(index, clients, settings):
    for gate in ("llm", "rules", "off"):
        pipeline, _ = advanced(index, clients, settings, gate)
        assert isinstance(pipeline.query, QueryRewriter), gate


def test_the_combined_call_is_shared_per_client(clients, settings, tmp_path):
    from supportdesk.llm.factory import build_clients

    assert shared_scope_rewrite(clients, settings) is shared_scope_rewrite(clients, settings)
    assert shared_scope_rewrite(build_clients(settings), settings) is not shared_scope_rewrite(clients, settings)


def test_combined_is_a_scope_gate_setting_with_its_own_prompt():
    s = Settings.load(env={}, overrides=["pipeline.name=advanced", "pipeline.scope_gate=combined"])
    assert s.pipeline.scope_gate == "combined" and s.prompts.scope_rewrite.exists()
    assert Settings.load(env={}, overrides=["pipeline.name=advanced"]).pipeline.scope_gate == "llm"  # unchanged
    with pytest.raises(ConfigError, match="combined"):
        Settings.load(env={}, overrides=["pipeline.scope_gate=maybe"])
