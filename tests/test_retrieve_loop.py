"""The corrective loop searches the corpus at most three times. The baseline still searches once."""
import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.pipelines.retrieve_loop import build_retrieve_loop, next_step
from supportdesk.retrieval.retriever import DenseRetriever
from supportdesk.types import CaseInput

CASE = CaseInput(case_id="c", title="Session state resets", body="The value is gone after a rerun.")


def test_next_step_stops_when_the_chunks_are_enough_or_the_budget_is_spent():
    assert next_step("yes", 1, 3) == "generate"
    assert next_step("no", 1, 3) == "rewrite"
    assert next_step("no", 2, 3) == "rewrite"
    assert next_step("no", 3, 3) == "generate"


def test_max_loops_cannot_exceed_three():
    with pytest.raises(ConfigError, match="retrieval.max_loops"):
        Settings.load(env={}, overrides=["retrieval.max_loops=4"])
    with pytest.raises(ConfigError, match="retrieval.max_loops"):
        Settings.load(env={}, overrides=["retrieval.max_loops=0"])


class Counting:
    def __init__(self, inner):
        self.inner = inner
        self.queries: list[str] = []

    def search(self, query, *, k, as_of):
        self.queries.append(query)
        return self.inner.search(query, k=k, as_of=as_of)


def _loop(index, clients, settings):
    counting = Counting(DenseRetriever(index, clients.embedding))
    return build_retrieve_loop(counting, clients, settings), counting


def test_yes_on_the_first_judge_searches_once(index, clients, settings):
    pipeline, counting = _loop(index, clients, settings)
    clients.chat.script[1] = "yes"
    result = pipeline.run(CASE)
    assert len(counting.queries) == 1
    assert result.trace["loop"]["stop_reason"] == "passages were sufficient"
    assert result.trace["loop"]["attempt"] == 1
    assert len(result.retrieved) == 6 or result.retrieved
    assert clients.chat.calls == 2  # judge, then the generator


def test_no_then_yes_searches_twice(index, clients, settings):
    pipeline, counting = _loop(index, clients, settings)
    clients.chat.script.update({1: "no", 2: "expander session state documentation", 3: "yes"})
    result = pipeline.run(CASE)
    assert len(counting.queries) == 2
    assert counting.queries[1] == "expander session state documentation"
    assert result.trace["loop"]["sufficient"] == "yes"
    assert result.trace["n_queries"] == 2
    assert clients.chat.calls == 4  # judge, rewrite, judge, generator


def test_three_no_answers_stop_on_the_last_chunks(index, clients, settings):
    pipeline, counting = _loop(index, clients, settings)
    clients.chat.script.update({
        1: "no",
        2: "second query about session state",
        3: "no",
        4: "third query about widgets",
        5: "no",
    })
    result = pipeline.run(CASE)
    assert counting.queries == [
        counting.queries[0],
        "second query about session state",
        "third query about widgets",
    ]
    assert result.trace["loop"]["stop_reason"] == "reached maximum tries (3)"
    assert result.trace["loop"]["attempt"] == 3
    assert result.retrieved
    assert clients.chat.calls == 6  # three judges, two rewrites, one generator


def test_a_repeated_rewrite_is_changed_before_the_next_search(index, clients, settings):
    from supportdesk.pipelines.stages.defaults import build_query
    original = build_query(CASE, settings.context.max_query_chars)
    pipeline, counting = _loop(index, clients, settings)
    clients.chat.script = {1: "no", 2: original, 3: "yes"}
    pipeline.run(CASE)
    assert counting.queries[0] == original
    collapsed = " ".join(original.split())
    assert counting.queries[1] == f"{collapsed} streamlit session state evidence"


def test_the_baseline_still_searches_once(index, clients, settings):
    counting = Counting(DenseRetriever(index, clients.embedding))
    result = build_baseline(counting, clients, settings).run(CASE)
    assert len(counting.queries) == 1
    assert "loop" not in result.trace
    assert clients.chat.calls == 1
