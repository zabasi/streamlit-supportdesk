"""The case workflow: answers and questions are posted automatically; escalations wait for a
maintainer, and the rules on that path are enforced in code."""
import json

import pytest

from supportdesk.casework import CaseWorkflow, Tracker, TrackerError
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.retrieval.retriever import DenseRetriever

TITLE, BODY = "Session state lost after refresh", "st.session_state is empty after F5 on Streamlit 1.31.1"


@pytest.fixture()
def world(index, clients, settings, tmp_path):
    pipeline = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)
    tracker = Tracker(tmp_path / "tracker.sqlite")
    return CaseWorkflow(pipeline, tracker, tmp_path / "checkpoints.sqlite", review="all"), tracker, clients


def model_says(clients, outcome, reply, citations=()):
    """Script the (fake) model's next answer."""
    clients.chat.script[clients.chat.calls + 1] = json.dumps(
        {"outcome": outcome, "user_reply": reply, "known_facts": [], "open_questions": [],
         "next_step": "next", "citations": list(citations)})


def assistant_messages(tracker, case_id):
    return [m["body"] for m in tracker.show(case_id)["messages"] if m["role"] == "assistant"]


def test_every_proposal_waits_and_approval_posts_exactly_the_approved_payload(world):
    """MT-DEV-7044's tracker assertions: no write before approval; exactly one comment matching
    the approved payload."""
    flow, tracker, clients = world
    for case_id, outcome, reply, status, labels in (
            ("A", "answer", "Use separate keys for data and widget. [S1]", "answered", []),
            ("Q", "request_info", "Which Streamlit version are you on?", "waiting-for-user", ["needs-info"]),
            ("E", "escalate", "I've passed this to the maintainers.", "escalated", ["needs-triage"])):
        tracker.open_case(case_id, TITLE, BODY)
        model_says(clients, outcome, reply, ["S1"] if outcome == "answer" else [])
        out = flow.run_turn(case_id)
        assert out["status"] == "awaiting_review" and out["proposal"]["outcome"] == outcome
        assert assistant_messages(tracker, case_id) == [] and tracker.show(case_id)["notes"] == []
        assert tracker.show(case_id)["status"] == "awaiting-review"
        action = flow.review(case_id, "approve", expected_proposal=out["proposal"]["proposal_id"])
        assert action["posted"] is True and action["status"] == status
        case = tracker.show(case_id)
        assert assistant_messages(tracker, case_id) == [reply]  # exactly one, exactly the approved text
        assert case["status"] == status and case["labels"] == labels and len(case["notes"]) == 1
        with pytest.raises(TrackerError, match="nothing is waiting"):
            flow.review(case_id, "approve")  # a second approval does nothing


def test_edit_posts_the_edited_text_and_reject_posts_nothing(world):
    flow, tracker, clients = world
    for case_id in ("ED", "RJ"):
        tracker.open_case(case_id, TITLE, BODY)
        model_says(clients, "escalate", "Escalating.")
        flow.run_turn(case_id)
    assert flow.review("ED", "edit", text="A maintainer will look at this today.")["kind"] == "posted"
    assert assistant_messages(tracker, "ED") == ["A maintainer will look at this today."]
    assert flow.review("RJ", "reject")["kind"] == "rejected"
    assert assistant_messages(tracker, "RJ") == [] and tracker.show("RJ")["status"] == "open"
    with pytest.raises(TrackerError, match="edit plus the new text"):
        flow.review("RJ", "edit")


def test_a_proposal_goes_stale_when_the_user_writes_again(world):
    flow, tracker, clients = world
    tracker.open_case("S", TITLE, BODY)
    model_says(clients, "escalate", "Escalating.")
    flow.run_turn("S")
    tracker.add_user_message("S", "Actually it only happens on the other server.")
    action = flow.review("S", "approve")
    assert action == {"kind": "stale", "posted": False, "proposal_id": action["proposal_id"]}
    assert assistant_messages(tracker, "S") == []
    model_says(clients, "answer", "Then check that server's proxy. [S1]", ["S1"])
    out = flow.run_turn("S")  # the new turn sees the corrected facts and makes a new proposal
    assert flow.review("S", "approve", expected_proposal=out["proposal"]["proposal_id"])["posted"] is True
    assert tracker.case_input("S").history[0]["body"] == "Actually it only happens on the other server."


def test_a_waiting_review_survives_a_restart(world, tmp_path):
    flow, tracker, clients = world
    tracker.open_case("R", TITLE, BODY)
    model_says(clients, "escalate", "Escalating.")
    flow.run_turn("R")
    restarted = CaseWorkflow(flow.pipeline, Tracker(tmp_path / "tracker.sqlite"), tmp_path / "checkpoints.sqlite",
                             review="all")
    assert restarted.pending("R")["reply"] == "Escalating."
    assert restarted.review("R", "approve")["posted"] is True


def test_cases_never_mix_and_an_approval_names_its_proposal(world):
    flow, tracker, clients = world
    tracker.open_case("X", TITLE, BODY)
    tracker.open_case("Y", "Other", "other problem")
    model_says(clients, "escalate", "Escalating X.")
    flow.run_turn("X")
    with pytest.raises(TrackerError, match="nothing is waiting"):
        flow.review("Y", "approve")  # X's pending review cannot be approved through Y
    with pytest.raises(TrackerError, match="is not the one waiting"):
        flow.review("X", "approve", expected_proposal="0000000000000000")
    assert assistant_messages(tracker, "X") == []
    with pytest.raises(TrackerError, match="no case"):
        flow.run_turn("Z")


def test_running_the_same_turn_twice_never_posts_twice(world):
    flow, tracker, clients = world
    tracker.open_case("D", TITLE, BODY)
    for _ in range(2):  # e.g. a retried request: the same proposal, the same proposal id
        model_says(clients, "request_info", "Which version?")
        flow.run_turn("D")
    flow.review("D", "approve")
    assert assistant_messages(tracker, "D") == ["Which version?"]
    assert not tracker.post("D", tracker.show("D")["messages"][-1]["proposal_id"], "Which version?")


def test_the_case_review_and_tracker_commands(tmp_path, capsys):
    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline"]
    assert main(["index", *base, "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]) == 0
    capsys.readouterr()
    assert main(["case", "C-1", *base, "--title", TITLE, "--body", BODY]) == 0  # default: answers post directly
    assert json.loads(capsys.readouterr().out)["posted"] is True
    assert main(["review", "C-1", *base]) == 0
    assert json.loads(capsys.readouterr().out) == {"pending": None}
    assert main(["review", "C-1", *base, "--approve"]) == 1
    assert "nothing is waiting" in capsys.readouterr().err
    assert main(["case", "C-1", *base, "--body", "Still happens on 1.32.0"]) == 0  # a follow-up turn
    capsys.readouterr()
    reviewed = [*base, "--set", "case.review=all"]  # every proposal waits
    assert main(["case", "C-2", *reviewed, "--title", TITLE, "--body", BODY]) == 0
    assert "waiting for a maintainer" in capsys.readouterr().out
    assert main(["review", "C-2", *reviewed, "--approve"]) == 0
    assert json.loads(capsys.readouterr().out)["posted"] is True
    assert main(["case", "C-1", *base, "--title", TITLE, "--body", BODY]) == 1  # already open
    capsys.readouterr()
    assert main(["tracker", "C-1", *base]) == 0
    case = json.loads(capsys.readouterr().out)
    assert [m["role"] for m in case["messages"]] == ["user", "assistant", "user", "assistant"]
    assert main(["tracker", *base, "--reset"]) == 0 and main(["tracker", *base]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1]) == []


def test_with_review_on_escalations_only_answers_and_questions_are_posted_directly(index, clients, settings, tmp_path):
    pipeline = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)
    tracker = Tracker(tmp_path / "t.sqlite")
    flow = CaseWorkflow(pipeline, tracker, tmp_path / "c.sqlite")  # the default: case.review = escalations
    assert flow.review_policy == "escalations"
    for case_id, outcome, reply in (("A", "answer", "Use separate keys. [S1]"), ("Q", "request_info", "Which version?")):
        tracker.open_case(case_id, TITLE, BODY)
        model_says(clients, outcome, reply, ["S1"] if outcome == "answer" else [])
        out = flow.run_turn(case_id)
        assert out["status"] == "done" and out["action"]["posted"] is True
        assert assistant_messages(tracker, case_id) == [reply]
    tracker.open_case("E", TITLE, BODY)
    model_says(clients, "escalate", "Passing this to the maintainers.")
    assert flow.run_turn("E")["status"] == "awaiting_review" and assistant_messages(tracker, "E") == []
    assert flow.review("E", "approve")["status"] == "escalated"
    tracker.add_user_message("Q", "1.32.0")  # a new turn after a direct post works the same way
    model_says(clients, "answer", "Then upgrade. [S1]", ["S1"])
    assert flow.run_turn("Q")["action"]["posted"] is True and len(assistant_messages(tracker, "Q")) == 2
