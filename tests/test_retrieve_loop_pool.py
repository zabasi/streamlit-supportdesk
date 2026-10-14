"""The pooled corrective loops: keep every scored chunk, choose the final ones after the loop."""
import pytest

from supportdesk.pipelines.retrieve_loop_pool import (
    build_retrieve_loop_best,
    build_retrieve_loop_mixed,
    parse_scores,
    rank_pool,
    select_best,
    select_mixed,
    split_sizes,
)
from supportdesk.pipelines.stages.defaults import build_query
from supportdesk.retrieval.retriever import DenseRetriever
from supportdesk.types import CaseInput
from tests.helpers import override

CASE = CaseInput(case_id="c", title="Session state resets", body="The value is gone after a rerun.")


def entry(chunk_id, score, similarity=0.5, iteration=1):
    return {"row": {"chunk": {"chunk_id": chunk_id}, "score": similarity, "rank": 1, "source": "dense"},
            "score": score, "iteration": iteration, "score_source": "judge"}


def pool(*scores):
    return {f"c{i}": entry(f"c{i}", s) for i, s in enumerate(scores)}


def ids(chosen):
    return [(e["row"]["chunk"]["chunk_id"], group) for e, group in chosen]


def test_scores_are_read_with_or_without_brackets_and_clamped():
    known = {"doc:w:1", "issue:7:0", "doc:w:2"}
    text = "no\n[doc:w:1] 7\n[issue:7:0]: 12\nstray [other] 9\ndoc:w:2 = 3.5"
    assert parse_scores(text, known) == {"doc:w:1": 7.0, "issue:7:0": 10.0, "doc:w:2": 3.5}
    assert parse_scores("yes", known) == {}


def test_ranking_breaks_ties_by_similarity_then_earlier_try_then_id():
    p = {"a": entry("a", 5, similarity=0.2), "b": entry("b", 5, similarity=0.9),
         "c": entry("c", 5, similarity=0.9, iteration=0), "d": entry("d", 8, similarity=0.1)}
    assert [e["row"]["chunk"]["chunk_id"] for e in rank_pool(p)] == ["d", "c", "b", "a"]


def test_best_takes_the_highest_scores_of_the_whole_pool():
    ranked = rank_pool(pool(3, 9, 1, 7, 5, 8, 2, 6))
    assert ids(select_best(ranked, 6)) == [("c1", "best"), ("c5", "best"), ("c3", "best"),
                                           ("c7", "best"), ("c4", "best"), ("c0", "best")]


@pytest.mark.parametrize("n, sizes", [(6, (2, 2, 2)), (7, (3, 2, 2)), (8, (3, 3, 2)), (9, (3, 3, 3)),
                                      (3, (1, 1, 1)), (4, (2, 1, 1)), (2, (1, 1, 0)), (1, (1, 0, 0))])
def test_thirds_give_the_remainder_to_the_best_first(n, sizes):
    assert split_sizes(n) == sizes and sum(sizes) == n


def test_mixed_takes_best_middle_and_worst_thirds():
    ranked = rank_pool(pool(*range(12)))  # scores 11 ... 0
    chosen = select_mixed(ranked, 6)
    assert ids(chosen) == [("c11", "best"), ("c10", "best"), ("c6", "middle"), ("c5", "middle"),
                           ("c1", "worst"), ("c0", "worst")]


def test_mixed_with_a_small_pool_returns_all_of_it_in_thirds():
    chosen = select_mixed(rank_pool(pool(9, 5, 1, 0)), 6)
    assert [g for _, g in chosen] == ["best", "best", "middle", "worst"]
    assert len(select_mixed(rank_pool({}), 6)) == 0


def _two_tries(index, clients, settings, builder, first, second):
    """Run the loop with a judge that scores the first search's chunks with ``first`` and the second
    search's with ``second``; returns (result, ids of try 1, ids of try 2)."""
    settings = override(settings, retrieval={"max_loops": 2})
    dense = DenseRetriever(index, clients.embedding)
    query1 = build_query(CASE, settings.context.max_query_chars)
    query2 = "expander widget key rerun"
    hits1 = [h.chunk.chunk_id for h in dense.search(query1, k=6, as_of=None)]
    hits2 = [h.chunk.chunk_id for h in dense.search(query2, k=6, as_of=None)]
    clients.chat.script.update({
        1: "no\n" + "\n".join(f"[{c}] {s}" for c, s in zip(hits1, first)),
        2: query2,
        3: "no\n" + "\n".join(f"[{c}] {s}" for c, s in zip(hits2, second)),
    })
    return builder(dense, clients, settings).run(CASE), hits1, hits2


def test_best_keeps_the_highest_scoring_chunks_of_both_tries(index, clients, settings):
    result, hits1, hits2 = _two_tries(index, clients, settings, build_retrieve_loop_best,
                                      first=[9, 1, 1, 1, 1, 1], second=[8, 7, 6, 5, 4, 3])
    kept = [r["chunk_id"] for r in result.retrieved]
    new = [c for c in hits2 if c not in hits1]
    assert len(kept) == 6 and len(set(kept)) == 6
    assert hits1[0] == kept[0]  # the only high score of the first try survives the second search
    assert new and set(new) & set(kept)  # and the second search's chunks are there too
    loop = result.trace["loop"]
    assert loop["attempt"] == 2 and loop["stop_reason"] == "reached maximum tries (2)"
    assert loop["selection"]["mode"] == "best" and loop["selection"]["pool_size"] >= 6
    assert {c["group"] for c in loop["selection"]["chunks"]} == {"best"}
    assert [c["score"] for c in loop["selection"]["chunks"]] == sorted(
        (c["score"] for c in loop["selection"]["chunks"]), reverse=True)
    assert {r["found_by"] for r in result.retrieved} == {"loop_best"}
    assert clients.chat.calls == 4  # judge, rewrite, judge, generator: no extra model call


def test_mixed_returns_best_middle_and_worst(index, clients, settings):
    result, _first, _second = _two_tries(index, clients, settings, build_retrieve_loop_mixed,
                                      first=[9, 8, 7, 6, 5, 4], second=[3, 2, 1, 0, 0, 0])
    selection = result.trace["loop"]["selection"]
    groups = [c["group"] for c in selection["chunks"]]
    assert selection["mode"] == "mixed"
    assert len(groups) == 6
    if selection["pool_size"] >= 6:
        assert groups == ["best", "best", "middle", "middle", "worst", "worst"]
    scores = [c["score"] for c in selection["chunks"]]
    assert scores[0] == max(scores) and scores[-1] == min(scores)
    assert {r["found_by"] for r in result.retrieved} <= {"loop_best", "loop_middle", "loop_worst"}


def test_a_yes_on_the_first_try_stops_and_returns_that_tries_best(index, clients, settings):
    dense = DenseRetriever(index, clients.embedding)
    first = [h.chunk.chunk_id for h in dense.search(build_query(CASE, 4000), k=6, as_of=None)]
    clients.chat.script[1] = "yes\n" + "\n".join(f"[{c}] {10 - n}" for n, c in enumerate(first))
    result = build_retrieve_loop_best(dense, clients, override(settings, retrieval={"max_loops": 2})).run(CASE)
    assert [r["chunk_id"] for r in result.retrieved] == first
    assert result.trace["loop"]["stop_reason"] == "passages were sufficient"
    assert result.trace["loop"]["attempt"] == 1 and clients.chat.calls == 2


def test_chunks_the_judge_did_not_score_fall_back_to_their_similarity(index, clients, settings):
    clients.chat.script[1] = "yes"  # no scores at all
    dense = DenseRetriever(index, clients.embedding)
    result = build_retrieve_loop_best(dense, clients, settings).run(CASE)
    chunks = result.trace["loop"]["selection"]["chunks"]
    assert chunks and {c["score_source"] for c in chunks} == {"similarity"}
    assert [r["chunk_id"] for r in result.retrieved] == [c["chunk_id"] for c in chunks]


def test_the_shipped_profiles_allow_one_repeat():
    from supportdesk.config import Settings
    for name in ("retrieve_loop", "retrieve_loop_best", "retrieve_loop_mixed"):
        assert Settings.load(env={}, overrides=[f"pipeline.name={name}"]).retrieval.max_loops == 2
