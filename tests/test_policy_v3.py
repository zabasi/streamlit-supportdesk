"""Policy v3: schema, literal report hydration, packet adapter. No API calls."""
from __future__ import annotations

import json

from supportdesk.eval.cases import case_from_dict
from supportdesk.eval.grounded_metrics import evaluate_grounded
from supportdesk.generation.policy_context import blocks, context, reference_map
from supportdesk.generation.policy_v3 import (
    FROZEN_INPUTS,
    PROMPT_V3B_PATH,
    PROMPT_V3C_PATH,
    citations_from_message,
    issues,
    load_frozen_rows,
    load_prompt,
    rendered,
    request,
    resolve_report_observations,
    row_from_case,
    run_policy_turn,
    sources_from_packet,
    validate,
)
from supportdesk.pipelines.packet import Packet
from supportdesk.types import CaseInput, Chunk


def _row():
    return {
        "case_id": "DEV-1",
        "input": {
            "case_id": "DEV-1",
            "title": "widget value disappears",
            "body": "The expander hides the text_area.\n\nStreamlit 1.13.0",
            "history": [],
            "environment": {"streamlit_version": "1.13.0"},
        },
        "sources": [{
            "source_id": "doc:session_state",
            "source_type": "documentation",
            "title": "Session State",
            "passages": [{"span_id": "doc:session_state:000", "text":
                          "Every widget with a key is automatically added to Session State."}],
        }],
    }


def _payload(outcome="answer", message="Bind the widget with a key [S1].", **extra):
    action = {"outcome": outcome, "message": message}
    if outcome == "request_info":
        action["question"] = extra.get("question", "Which Streamlit version is installed?")
    return {
        "case_id": "DEV-1",
        "reported_observation_refs": extra.get("refs", ["R1"]),
        "workflow_findings": extra.get("findings", [{
            "report_ref": "R1",
            "problem": "text_area has no stable key across expander close",
            "correction": "give the widget a key that is not deleted",
            "evidence_refs": ["S1"],
        }]),
        "action": action,
        "maintainer_summary": {
            "problem": "widget value disappears",
            "reported_environment": "1.13.0",
            "uncertainties": [],
            "next_step": "confirm the key survives the expander rerun",
        },
    }


def test_report_blocks_keep_blank_line_splits():
    parts = blocks("The expander hides the text_area.\n\nStreamlit 1.13.0")
    assert [p["text"] for p in parts] == [
        "The expander hides the text_area.", "Streamlit 1.13.0"]


def test_schema_accepts_answer_and_request_info():
    row = _row()
    validate(_payload(), row)
    validate(_payload(outcome="request_info", message="Need the version.",
                      question="Which Streamlit version?", findings=[]), row)


def test_unknown_or_duplicate_observation_refs_fail():
    row = _row()
    bad = _payload(refs=["S1"])  # source ids are not report observations
    try:
        validate(bad, row)
        raise AssertionError("source id must not be a reported observation")
    except ValueError:
        pass
    try:
        validate(_payload(refs=["R1", "R1"]), row)
        raise AssertionError("duplicate refs must fail")
    except ValueError:
        pass
    inline = _payload(message="See [S99].")
    try:
        validate(inline, row)
        raise AssertionError("unknown inline reference must fail")
    except ValueError:
        pass


def test_controller_copies_original_report_text_not_model_quote():
    row = _row()
    record = rendered(_payload(), row)
    quotes = [item["quote"] for item in record["reported_observations"]]
    assert quotes == ["The expander hides the text_area."]
    asserted = resolve_report_observations(_payload(), row)
    original = reference_map(row)["R1"]["text"]
    assert asserted[0]["quote"] == original
    assert asserted[0]["origin"] == "original_report"
    # A capitalization change in a model-written quote cannot enter this field:
    # the model never supplied quote text, only the ID R1.


def test_request_info_appends_the_question():
    row = _row()
    record = rendered(_payload(outcome="request_info", message="Need one fact.",
                               question="Which Streamlit version?", findings=[]), row)
    assert record["outcome"] == "request_info"
    assert "Which Streamlit version?" in record["user_reply"]


def test_answer_without_inline_citation_is_a_warning_not_a_rewrite():
    row = _row()
    payload = _payload(message="Just add a key.")
    validate(payload, row)
    assert "Answer has no inline evidence citation" in issues(payload)


def test_packet_adapter_numbers_passages_as_s_labels():
    packet = Packet(chunks=[
        Chunk(chunk_id="c1", source_id="doc:a", source_type="documentation",
              title="A", text="Session state keeps widget values."),
        Chunk(chunk_id="c2", source_id="doc:a", source_type="documentation",
              title="A", text="Widget keys survive reruns."),
    ])
    case = CaseInput(case_id="DEV-1", title="t", body="b")
    row = row_from_case(case, sources_from_packet(packet))
    refs = reference_map(row)
    assert refs["S1"]["passage_id"] == "c1"
    assert refs["S2"]["passage_id"] == "c2"
    prompt = context(row)
    assert "[S1]" in prompt and "[S2]" in prompt
    assert citations_from_message("See [S2] and [S1].", row) == ["c2", "c1"]


def test_run_withholds_invalid_contract_and_scores_a_valid_turn():
    row = _row()

    class Bad:
        def complete(self, messages, **k):
            return json.dumps({"outcome": "answer", "message": "hi"}), {"cost_usd": 0.0}

    class Good:
        def complete(self, messages, **k):
            return json.dumps(_payload()), {"cost_usd": 0.0}

    case = case_from_dict({
        "case_id": "DEV-1", "family_id": "f", "split": "dev", "category": "answerable",
        "input": {"case_id": "DEV-1", "title": row["input"]["title"],
                  "body": row["input"]["body"],
                  "environment": row["input"]["environment"]},
        "acceptable_outcomes": ["answer"], "gold_evidence": ["doc:session_state"],
    })
    failed = run_policy_turn(case, row["sources"], Bad(), retries=0,
                             gold={"doc:session_state"}, evidence_kind="gold")
    assert failed.outcome == "validation_failed"
    assert failed.answer is None

    ok = run_policy_turn(case, row["sources"], Good(), retries=0,
                         gold={"doc:session_state"}, evidence_kind="gold")
    assert ok.outcome == "answer"
    assert ok.reported_observations[0]["quote"] == "The expander hides the text_area."
    assert ok.citations == ["doc:session_state:000"]
    h = evaluate_grounded([(case, ok.to_dict())])["headline"]
    assert h["outcome_accuracy"] == 1.0
    assert h["confusion_matrix"]["counts"]["answer"]["answer"] == 1


def test_frozen_fifteen_requests_build():
    assert FROZEN_INPUTS.is_file(), f"copy model_inputs.jsonl to {FROZEN_INPUTS}"
    rows = load_frozen_rows()
    assert len(rows) == 15
    built = request(rows[0], "gpt-4o-mini")
    assert built["messages"][0]["role"] == "system"
    assert "Streamlit support engineer" in built["messages"][0]["content"]
    assert rows[0]["case_id"] in built["messages"][1]["content"]
    gpt5 = request(rows[0], "gpt-5.4-mini")
    assert gpt5["reasoning_effort"] == "medium"
    assert "temperature" not in gpt5


def test_v3b_prompt_forces_request_info_first():
    text = load_prompt(PROMPT_V3B_PATH)
    assert "Choose the action FIRST" in text
    assert text.index("request_info") < text.index("escalate")
    built = request(load_frozen_rows()[0], "gpt-4o-mini", prompt_path=PROMPT_V3B_PATH)
    assert built["messages"][0]["content"] == text


def test_v3c_prompt_keeps_workaround_from_becoming_answer():
    text = load_prompt(PROMPT_V3C_PATH)
    assert "do not change the action to answer because a workaround exists" in text
    assert "one-rerun backend/frontend divergence" in text
    built = request(load_frozen_rows()[0], "gpt-4o-mini", prompt_path=PROMPT_V3C_PATH)
    assert built["messages"][0]["content"] == text
