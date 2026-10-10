"""The stage architecture: swappable pieces, shared failure handling, honest accounting."""
from dataclasses import replace

import pytest

from supportdesk.llm.errors import BudgetExhausted, ProviderError
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.pipelines.staged import STAGE_NAMES, StagedPipeline
from supportdesk.pipelines.stages import (
    DefaultPacketBuilder,
    Generator,
    NoRerank,
    NoValidator,
    PacketBuilder,
    PlainQuery,
    QueryBuilder,
    Reranker,
    SingleCallGenerator,
    call_and_parse,
)
from supportdesk.pipelines.stages.protocols import GenerationFailed, Summarizer, Validator
from supportdesk.pipelines.summary import build_summary
from supportdesk.retrieval.retriever import DenseRetriever
from supportdesk.types import CaseInput
from tests.helpers import override

CASE = CaseInput(case_id="t", title="Session state is empty after browser refresh",
                 body="I store the user in st.session_state. After F5 the value is gone. "
                      "Streamlit 1.31.1, Chrome.", created_at="2024-04-20T10:00:00Z")
PROXY_QUERY = "page stays on loading spinner behind nginx reverse proxy WebSocket headers"


@pytest.fixture()
def base(index, clients, settings):
    return build_baseline(DenseRetriever(index, clients.embedding), clients, settings)


class Reverse:
    def __call__(self, case, query, hits):
        return list(reversed(hits))


def test_baseline_is_a_configuration_of_the_staged_pipeline(base):
    assert isinstance(base, StagedPipeline) and base.name == "baseline"
    assert base.describe() == {
        "query": "PlainQuery", "retriever": "DenseRetriever", "reranker": "NoRerank",
        "packet": "DefaultPacketBuilder", "generator": "SingleCallGenerator",
        "validator": "NoValidator", "summarizer": "build_summary"}
    assert tuple(base.describe()) == STAGE_NAMES


def test_default_stages_satisfy_their_contracts(clients, settings):
    assert isinstance(PlainQuery(4000), QueryBuilder) and isinstance(NoRerank(), Reranker)
    assert isinstance(DefaultPacketBuilder(6, 1800), PacketBuilder) and isinstance(NoValidator(), Validator)
    assert isinstance(SingleCallGenerator(clients, settings), Generator) and isinstance(build_summary, Summarizer)


def test_variant_replaces_one_stage_and_leaves_the_original_alone(base):
    swapped = base.variant(reranker=Reverse())
    assert swapped.name == "baseline+reranker" and swapped.describe()["reranker"] == "Reverse"
    assert base.describe()["reranker"] == "NoRerank"
    plain, flipped = base.run(CASE), swapped.run(CASE)
    assert [r["chunk_id"] for r in flipped.retrieved] == [
        r["chunk_id"] for r in reversed(plain.retrieved)]
    assert flipped.pipeline == "baseline+reranker" and plain.pipeline == "baseline"
    with pytest.raises(TypeError, match="unknown stage"):
        base.variant(magic=1)


def test_bad_stages_are_rejected_at_construction(base, clients, settings):
    kwargs = {s: getattr(base, s) for s in STAGE_NAMES}
    with pytest.raises(TypeError, match="reranker"):
        StagedPipeline(name="x", clients=clients, settings=settings, **{**kwargs, "reranker": 42})
    with pytest.raises(TypeError, match="retriever"):
        StagedPipeline(name="x", clients=clients, settings=settings, **{**kwargs, "retriever": print})


def test_candidate_pool_is_reranked_then_trimmed_to_top_k(index, clients, settings):
    seen = {}
    settings = override(settings, retrieval={"candidate_k": 12})

    class KeepAll:
        def __call__(self, case, query, hits):
            seen["candidates"] = len(hits)
            return hits  # returns more than top_k on purpose

    wide = build_baseline(DenseRetriever(index, clients.embedding), clients, settings).variant(
        reranker=KeepAll())
    result = wide.run(CASE)
    # the reranker saw the whole visible pool (11: the historical filter hides the 2024-10 release)
    assert seen["candidates"] == 11 > settings.retrieval.top_k
    assert len(result.retrieved) == settings.retrieval.top_k  # the pipeline trims what it returns


def test_several_queries_are_fused_with_rrf(base):
    class TwoQueries:
        def __call__(self, case, env):
            return ["session state lost after browser refresh F5", PROXY_QUERY]

    result = base.variant(query=TwoQueries()).run(CASE)
    top = [r["source_id"] for r in result.retrieved[:4]]
    assert "fixture-doc-session-state" in top and "fixture-doc-deploy-proxy" in top
    assert result.trace["n_queries"] == 2
    assert len({r["chunk_id"] for r in result.retrieved}) == len(result.retrieved)  # no duplicates


def test_empty_query_list_is_a_clear_error(base):
    class Nothing:
        def __call__(self, case, env):
            return []
    result = base.variant(query=Nothing()).run(CASE)
    assert result.outcome == "error" and "returned no queries" in result.error


def test_failures_in_any_stage_get_the_shared_mapping(base):
    class Broke:
        def __call__(self, case, env):
            raise BudgetExhausted("insufficient_quota key sk-abcdefghijklmnop1234")

    class Flaky:
        def __call__(self, case, query, hits):
            raise ProviderError("502 from gateway")

    class Buggy:
        def __call__(self, case, query, hits):
            raise RuntimeError("a bug in a stage")

    stopped = base.variant(query=Broke()).run(CASE)
    assert stopped.outcome == "budget_stopped" and "sk-abcdef" not in stopped.error
    assert base.variant(reranker=Flaky()).run(CASE).outcome == "error"
    with pytest.raises(RuntimeError, match="a bug"):  # bugs are not disguised as model failures
        base.variant(reranker=Buggy()).run(CASE)


def test_validator_can_downgrade_and_the_summary_follows(base):
    class Cautious:
        def __call__(self, case, packet, answer):
            return replace(answer, outcome="escalate", violations=[*answer.violations, "unsupported"])

    result = base.variant(validator=Cautious()).run(CASE)
    assert result.outcome == "escalate" and "unsupported" in result.violations
    assert result.maintainer_summary["outcome"] == "escalate"  # summarised after validation


def test_custom_generator_reuses_the_shared_retry_policy(base, clients, settings):
    from supportdesk.pipelines.prompts import build_messages, load_prompts

    prompts = load_prompts(settings)

    class NoRetryGenerator:
        def __call__(self, case, env, packet):
            return call_and_parse(clients.chat, build_messages(case, env, packet, prompts, max_body_chars=4000), packet,
                                  retries=0)

    clients.chat.script = {1: "bad_json"}
    result = base.variant(generator=NoRetryGenerator()).run(CASE)
    assert result.outcome == "error" and clients.chat.calls == 1
    assert [a["status"] for a in result.trace["attempts"]] == ["NotJson"]
    with pytest.raises(GenerationFailed):
        clients.chat.script = {2: "bad_json"}
        from supportdesk.pipelines.packet import Packet
        call_and_parse(clients.chat, [{"role": "user", "content": "x"}], Packet(), retries=0)


def test_model_calls_made_by_any_stage_are_counted_in_usage(base, clients):
    class Rewrite:
        def __call__(self, case, env):
            clients.chat.complete([{"role": "user", "content": "rewrite this report"}],
                                  stage="rewrite")
            return ["session state refresh"]

    mark = clients.ledger.mark()
    result = base.variant(query=Rewrite()).run(CASE)
    rows = clients.ledger.rows[mark:]
    rewrite_tokens = sum(r["input_tokens"] for r in rows if r["stage"] == "rewrite")
    assert rewrite_tokens > 0
    # per-case usage is exactly the ledger rows written during the run, rewrite call included
    assert result.usage["input_tokens"] == sum(r["input_tokens"] for r in rows)
    assert result.usage["input_tokens"] > rewrite_tokens  # plus the embedding and answer calls


def test_summarizer_is_swappable(base):
    result = base.variant(summarizer=lambda case, env, packet, answer: {"custom": True}).run(CASE)
    assert result.maintainer_summary == {"custom": True}


def test_trace_records_which_stages_produced_a_result(base):
    result = base.variant(reranker=Reverse()).run(CASE)
    assert result.trace["stages"]["reranker"] == "Reverse" and result.trace["n_queries"] == 1


def test_run_summary_and_report_show_the_stage_configuration(base, clients, settings, cases, tmp_path):
    import json

    from supportdesk.eval.report import render_report
    from supportdesk.eval.runner import run_eval
    run_eval(base, cases[:2], clients, settings, run_id="a", out_root=tmp_path)
    run_eval(base.variant(reranker=Reverse()), cases[:2], clients, settings, run_id="b", out_root=tmp_path)
    summary = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert summary["stages"]["reranker"] == "Reverse" and summary["pipeline"] == "baseline+reranker"
    html = render_report([tmp_path / "a", tmp_path / "b"])
    assert "Configuration" in html and "NoRerank" in html and "Reverse" in html and "ablation" in html
