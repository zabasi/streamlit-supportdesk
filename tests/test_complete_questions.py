"""pipeline.complete_questions: one question message for every material gap (invented examples only)."""
import copy
import dataclasses
import json

from supportdesk.pipelines.stages.complete_questions import askable, combine
from supportdesk.pipelines.stages.controlled import ControlledGenerator
from tests.test_controlled_v2 import CASE, assessment, packet


def gap(missing, question, *, material=True, user=True, supplied=False):
    return {"missing_information": missing, "material_to_main_need": material, "user_can_supply": user,
            "already_supplied_or_tested": supplied, "why_it_matters": f"Knowing the {missing} decides the fix.",
            "basis_refs": ["R1"], "question": question,
            "possible_results": [{"result": f"{missing} is A", "next_support_step": "Suggest the first fix."},
                                 {"result": f"{missing} is B", "next_support_step": "Suggest the second fix."}]}


def asking(*gaps):
    value = assessment()
    value["information_gaps"], value["answer_candidate"] = list(gaps), None
    return value


def result(value):
    first = askable(value)[0]
    return {"outcome": "request_info", "assessment": value,
            "action": {"outcome": "request_info", "message": first["why_it_matters"] + " [R1]", "question": first["question"]},
            "user_reply": first["why_it_matters"] + " [R1]\n\n" + first["question"],
            "routing": {"outcome": "request_info", "reason": "material_user_providable_gap", "gap_index": 0}}


def test_every_askable_gap_is_asked_in_one_message_and_the_decision_is_unchanged():
    original = result(asking(gap("library version", "Which version do you use?"), gap("full script", "Can you share the script?")))
    kept = copy.deepcopy(original)
    out = combine(original)
    assert out["outcome"] == "request_info" and out["routing"]["questions_combined"] == 2
    assert out["action"]["question"] == "1. Which version do you use?\n2. Can you share the script?"
    assert "Knowing the library version decides the fix. [R1]" in out["user_reply"]
    assert "Knowing the full script decides the fix. [R1]" in out["user_reply"]
    assert original == kept  # the package's result is not modified


def test_one_askable_gap_or_an_answer_leaves_the_result_as_it_was():
    value = asking(gap("library version", "Which version?"), gap("log", "The log?", supplied=True),
                   gap("intent", "Why?", material=False), gap("server setting", "Server?", user=False))
    single = result(value)
    assert askable(value) == [value["information_gaps"][0]] and combine(single) is single
    answered = {"outcome": "answer", "assessment": assessment(), "routing": {"reason": "supported_answer_candidate"}}
    assert combine(answered) is answered


def run(clients, settings, on):
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, complete_questions=on))
    gen = ControlledGenerator(clients, s)
    gen.chat.script[1] = json.dumps(asking(gap("library version", "Which version do you use?"),
                                           gap("full script", "Can you share the script?")))
    return gen(CASE, {}, packet())


def test_off_by_default_the_reply_asks_only_the_first_gap(clients, settings):
    answer = run(clients, settings, False)
    assert answer.outcome == "request_info" and answer.open_questions == ["Which version do you use?"]
    assert "Can you share the script?" not in answer.user_reply


def test_on_the_reply_and_open_questions_cover_every_gap(clients, settings):
    answer = run(clients, settings, True)
    assert answer.outcome == "request_info"
    assert answer.open_questions == ["Which version do you use?", "Can you share the script?"]
    assert "1. Which version do you use?\n2. Can you share the script?" in answer.user_reply
