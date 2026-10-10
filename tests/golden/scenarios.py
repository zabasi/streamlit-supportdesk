"""Fixed scenarios used to pin the baseline's behaviour across refactors.

``run_scenarios(make_pipeline)`` runs the baseline through the normal path and the failure paths
and returns plain, comparable data. The golden file was generated from the pre-refactor
``build_baseline``; ``test_golden_baseline.py`` asserts the current code still reproduces it.
Regenerate (``python -m tests.golden.generate``) only when the baseline is meant to change.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from supportdesk.eval.cases import exclusions_from_cases, load_cases
from supportdesk.indexing import run_indexing
from supportdesk.llm.errors import BudgetExhausted
from supportdesk.llm.factory import build_clients
from supportdesk.registry import make_retriever
from supportdesk.types import CaseInput
from tests.helpers import make_settings

ROOT = Path(__file__).resolve().parents[2]
SAMPLE, CASES = ROOT / "data" / "sample", ROOT / "data" / "eval" / "sample_cases.jsonl"
GOLDEN = Path(__file__).with_name("baseline_results.json")

SESSION = CaseInput(case_id="s1", title="Session state is empty after browser refresh",
                    body="I store the user in st.session_state. After F5 the value is gone. "
                         "Streamlit 1.31.1, Chrome.", created_at="2024-04-20T10:00:00Z")
HISTORY = CaseInput(case_id="s2", title="App stuck on loading on my server",
                    body="Works locally, server shows the loading screen.",
                    history=[{"role": "user", "body": "Reinstalling did not help"},
                             {"role": "maintainer", "body": "MAINTAINER ONLY TEXT"}])


class Broke:
    def complete(self, messages, *, stage="answer", json_mode=True):
        raise BudgetExhausted("insufficient_quota for key sk-abcdefghijklmnop12345")


def _world(**sections):
    tmp = Path(tempfile.mkdtemp(prefix="golden-"))
    settings = make_settings(tmp, **sections)
    clients = build_clients(settings)
    cases = load_cases(CASES)
    index = run_indexing(settings, clients, cases=cases, exclusions=exclusions_from_cases(cases))["index"]
    return settings, clients, index, cases


def normalize(result) -> dict:
    """Everything the baseline promises, minus wall-clock time and float noise in scores."""
    row = result.to_dict()
    usage = {k: v for k, v in row["usage"].items() if k != "wall_seconds"}
    trace = {k: row["trace"][k] for k in ("packet", "attempts", "query_sha256") if k in row["trace"]}
    return {
        "outcome": row["outcome"], "user_reply": row["user_reply"], "known_facts": row["known_facts"],
        "open_questions": row["open_questions"], "next_step": row["next_step"],
        "citations": row["citations"], "violations": row["violations"], "error": row["error"],
        "packet_hash": row["packet_hash"], "maintainer_summary": row["maintainer_summary"],
        "retrieved": [[r["chunk_id"], r["source_id"], r["parent_source_id"], r["rank"]]
                      for r in row["retrieved"]],
        "scores": [r["score"] for r in row["retrieved"]],
        "usage": usage, "trace": trace, "pipeline": row["pipeline"], "mode": row["mode"],
    }


def run_scenarios(make_pipeline) -> dict:
    """``make_pipeline(retriever, clients, settings)`` -> a pipeline. Returns {scenario: normalized}."""
    out: dict = {}

    def run(name, case, *, retriever="dense", setup=None, **sections):
        settings, clients, index, _ = _world(**sections)
        if setup:
            setup(clients, index)
        ret = make_retriever(retriever, index, clients, settings)
        out[name] = normalize(make_pipeline(ret, clients, settings).run(case))

    for case in load_cases(CASES):
        run(f"sample:{case.case_id}", case.input)
    run("history_user_turns_only", HISTORY)
    run("hybrid", load_cases(CASES)[1].input, retriever="hybrid")
    run("bm25", load_cases(CASES)[1].input, retriever="bm25")
    run("historical_filter_off", SESSION, retrieval={"historical_filter": False})
    run("top_k_2", SESSION, retrieval={"top_k": 2})
    run("bad_json_then_ok", SESSION, setup=lambda c, i: setattr(c.chat, "script", {1: "bad_json"}))
    run("errors_exhausted", SESSION,
        setup=lambda c, i: setattr(c.chat, "script", {1: "error", 2: "error", 3: "error"}))
    run("wrong_shape", SESSION, setup=lambda c, i: setattr(
        c.chat, "script", {1: json.dumps({"outcome": "solved", "user_reply": "x"})}))
    run("fabricated_citation", SESSION, setup=lambda c, i: setattr(c.chat, "script", {1: json.dumps(
        {"outcome": "answer", "user_reply": "see S9", "known_facts": [
            {"statement": "x", "evidence": ["S9"]}], "open_questions": [], "next_step": "n",
         "citations": ["S9"]})}))
    run("request_info_no_citations", SESSION, setup=lambda c, i: setattr(c.chat, "script", {1: json.dumps(
        {"outcome": "request_info", "user_reply": "Which version?", "open_questions": ["version?"]})}))
    run("budget_exhausted", SESSION, setup=lambda c, i: setattr(c, "chat", Broke()))
    run("retrieval_provider_error", SESSION, setup=lambda c, i: setattr(
        c.embedding, "script", {c.embedding.calls + 1: "error"}))
    return out
