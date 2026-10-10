"""The case workflow: one LangGraph graph for a whole support case.

    support nodes (intake ... summarize) -> route
        answer / request_info -> post_reply                                       (case.review = escalations)
        escalate              -> prepare_review -> human_review (pauses) -> apply_review
        (with case.review = all, every outcome takes the review path)

The generate node decides the outcome: answer when the evidence supports a confident answer,
request_info when one missing fact decides the case, escalate when a maintainer must look (no
defensible answer, a suspected bug, or an answer the model is not confident about). The rules are
enforced here in code, never left to the model or to text in a report:
  * an approval applies to one case and one proposal (its proposal_id);
  * if the user wrote again after the proposal was made, it is stale and nothing is posted;
  * rejecting posts nothing;
  * every post is keyed by its proposal_id, so a repeated or resumed call never posts twice.

Actions go to a local, resettable tracker (SQLite), never to a real repository. Checkpoints are
saved after every node, so a case waiting for review survives a restart.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from .pipelines.staged import StagedPipeline
from .pipelines.state import SupportState, case_to_dict
from .types import CaseInput

#: LangGraph's limit on node runs per case turn. The case graph has no loops, so this only guards
#: against a loop added later (for example a corrective re-search) running away.
RECURSION_LIMIT = 40
#: What an approved proposal does to the case: (status, labels added).
ON_APPROVAL = {"answer": ("answered", []), "request_info": ("waiting-for-user", ["needs-info"]),
               "escalate": ("escalated", ["needs-triage"])}


class TrackerError(RuntimeError):
    """A tracker action that is not allowed (unknown case, nothing to review, wrong proposal...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proposal_id(case_id: str, revision: int, text: str) -> str:
    return hashlib.sha256(f"{case_id}|{revision}|{text}".encode()).hexdigest()[:16]


class Tracker:
    """A small local issue tracker. ``revision`` = how many messages the user has written."""

    SCHEMA = """
    create table if not exists cases (case_id text primary key, title text not null,
        status text not null, labels text not null default '[]', updated_at text not null);
    create table if not exists messages (id integer primary key autoincrement, case_id text not null,
        role text not null, body text not null, proposal_id text unique, created_at text not null);
    create table if not exists notes (id integer primary key autoincrement, case_id text not null,
        body text not null, proposal_id text unique, created_at text not null);
    create table if not exists reviews (proposal_id text primary key, case_id text not null,
        thread_id text not null, revision integer not null, status text not null, created_at text not null);
    create table if not exists events (id integer primary key autoincrement, case_id text not null,
        kind text not null, detail text not null, created_at text not null);
    """

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(self.SCHEMA)

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self.db:
            return self.db.execute(sql, args).rowcount

    def _require(self, case_id: str) -> sqlite3.Row:
        row = self.db.execute("select * from cases where case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise TrackerError(f"no case {case_id!r}; open it with a title and a body first")
        return row

    def log(self, case_id: str, kind: str, detail: dict) -> None:
        self._write("insert into events (case_id, kind, detail, created_at) values (?, ?, ?, ?)",
                    (case_id, kind, json.dumps(detail, ensure_ascii=False), _now()))

    def open_case(self, case_id: str, title: str, body: str) -> None:
        if self.db.execute("select 1 from cases where case_id = ?", (case_id,)).fetchone():
            raise TrackerError(f"case {case_id!r} already exists; send a follow-up with --body only")
        with self.db:
            self.db.execute("insert into cases values (?, ?, 'open', '[]', ?)", (case_id, title, _now()))
            self.db.execute("insert into messages (case_id, role, body, created_at) values (?, 'user', ?, ?)",
                            (case_id, body, _now()))
        self.log(case_id, "opened", {"title": title})

    def add_user_message(self, case_id: str, body: str) -> int:
        self._require(case_id)
        self._write("insert into messages (case_id, role, body, created_at) values (?, 'user', ?, ?)",
                    (case_id, body, _now()))
        self.log(case_id, "user_message", {"revision": self.revision(case_id)})
        return self.revision(case_id)

    def revision(self, case_id: str) -> int:
        return self.db.execute("select count(*) from messages where case_id = ? and role = 'user'",
                               (case_id,)).fetchone()[0]

    def post(self, case_id: str, pid: str, body: str) -> bool:
        """Post a visible reply. False if this proposal was already posted (never posts twice)."""
        return bool(self._write("insert or ignore into messages (case_id, role, body, proposal_id, created_at) "
                                "values (?, 'assistant', ?, ?, ?)", (case_id, body, pid, _now())))

    def note(self, case_id: str, pid: str, body: str) -> bool:
        """An internal note for maintainers (not shown to the user)."""
        return bool(self._write("insert or ignore into notes (case_id, body, proposal_id, created_at) "
                                "values (?, ?, ?, ?)", (case_id, body, pid, _now())))

    def update(self, case_id: str, *, status: str, add_labels: list[str] = ()) -> None:
        labels = sorted(set(json.loads(self._require(case_id)["labels"])) | set(add_labels))
        self._write("update cases set status = ?, labels = ?, updated_at = ? where case_id = ?",
                    (status, json.dumps(labels), _now(), case_id))

    def add_review(self, case_id: str, pid: str, thread_id: str, revision: int) -> None:
        self._write("insert or ignore into reviews values (?, ?, ?, ?, 'pending', ?)",
                    (pid, case_id, thread_id, revision, _now()))

    def pending_review(self, case_id: str) -> sqlite3.Row | None:
        self._require(case_id)
        return self.db.execute("select * from reviews where case_id = ? and status = 'pending' "
                               "order by created_at desc, rowid desc limit 1", (case_id,)).fetchone()

    def close_review(self, pid: str, status: str) -> None:
        self._write("update reviews set status = ? where proposal_id = ?", (status, pid))

    def case_input(self, case_id: str) -> CaseInput:
        """What the assistant may see: the first message as the report, the rest as history."""
        case = self._require(case_id)
        messages = self.db.execute("select role, body, created_at from messages where case_id = ? "
                                   "order by id", (case_id,)).fetchall()
        first, rest = messages[0], messages[1:]
        return CaseInput(case_id=case_id, title=case["title"], body=first["body"],
                         history=[{"role": m["role"], "body": m["body"], "created_at": m["created_at"]}
                                  for m in rest])

    def show(self, case_id: str) -> dict:
        case = self._require(case_id)
        rows = lambda sql: [dict(r) for r in self.db.execute(sql, (case_id,)).fetchall()]
        return {"case_id": case_id, "title": case["title"], "status": case["status"],
                "labels": json.loads(case["labels"]),
                "messages": rows("select role, body, proposal_id, created_at from messages where case_id = ? order by id"),
                "notes": rows("select body, proposal_id, created_at from notes where case_id = ? order by id"),
                "reviews": rows("select proposal_id, revision, status from reviews where case_id = ? order by rowid"),
                "events": rows("select kind, detail, created_at from events where case_id = ? order by id")}

    def cases(self) -> list[dict]:
        return [{**dict(r), "labels": json.loads(r["labels"])}
                for r in self.db.execute("select case_id, title, status, labels from cases order by case_id")]

    def reset(self) -> None:
        with self.db:
            for table in ("cases", "messages", "notes", "reviews", "events"):
                self.db.execute(f"delete from {table}")


class CaseState(SupportState, total=False):
    revision: int    # the user's message count when this run started
    proposal: dict   # the proposal waiting for review
    review: dict     # the maintainer's decision
    action: dict     # what was done (or why not)


class CaseWorkflow:
    def __init__(self, pipeline: StagedPipeline, tracker: Tracker, checkpoints: str | Path,
                 *, review: str | None = None):
        self.pipeline, self.tracker = pipeline, tracker
        self.review_policy = review or pipeline.settings.case.review  # escalations | all
        Path(checkpoints).parent.mkdir(parents=True, exist_ok=True)
        self.saver = SqliteSaver(sqlite3.connect(str(checkpoints), check_same_thread=False))
        self.graph = self._build().compile(checkpointer=self.saver)

    # ---- graph ----

    def _build(self) -> StateGraph:
        graph = StateGraph(CaseState)
        last = self.pipeline.add_to(graph)
        graph.add_node("post_reply", self._post_reply)
        graph.add_node("prepare_review", self._prepare_review)
        graph.add_node("human_review", self._human_review)
        graph.add_node("apply_review", self._apply_review)
        graph.add_conditional_edges(last, self._route, ["post_reply", "prepare_review", END])
        graph.add_edge("post_reply", END)
        graph.add_edge("prepare_review", "human_review")
        graph.add_edge("human_review", "apply_review")
        graph.add_edge("apply_review", END)
        return graph

    def _route(self, state: CaseState) -> str:
        outcome = state["answer"]["outcome"]
        if outcome not in ON_APPROVAL:
            return END
        return "prepare_review" if self.review_policy == "all" or outcome == "escalate" else "post_reply"

    def _post_reply(self, state: CaseState) -> dict:
        """Post an answer or a question directly (case.review = escalations): same rules as an
        approval, without the wait: never on a stale turn, never twice."""
        case_id, answer = state["case"]["case_id"], state["answer"]
        if self._stale(state, case_id):
            self.tracker.log(case_id, "stale", {"revision": state["revision"]})
            return {"action": {"kind": "stale", "posted": False}}
        pid = proposal_id(case_id, state["revision"], answer["user_reply"])
        status, labels = ON_APPROVAL[answer["outcome"]]
        posted = self.tracker.post(case_id, pid, answer["user_reply"])
        self.tracker.note(case_id, pid, json.dumps(state.get("summary", {}), ensure_ascii=False))
        self.tracker.update(case_id, status=status, add_labels=labels)
        self.tracker.log(case_id, "posted" if posted else "already_posted",
                         {"proposal_id": pid, "outcome": answer["outcome"]})
        return {"action": {"kind": "posted", "posted": posted, "proposal_id": pid,
                           "outcome": answer["outcome"], "status": status}}

    def _stale(self, state: CaseState, case_id: str) -> bool:
        return self.tracker.revision(case_id) != state["revision"]

    def _prepare_review(self, state: CaseState, config) -> dict:
        case_id, answer = state["case"]["case_id"], state["answer"]
        pid = proposal_id(case_id, state["revision"], answer["user_reply"])
        self.tracker.add_review(case_id, pid, config["configurable"]["thread_id"], state["revision"])
        self.tracker.update(case_id, status="awaiting-review")
        self.tracker.log(case_id, "proposed", {"proposal_id": pid, "outcome": answer["outcome"]})
        return {"proposal": {"proposal_id": pid, "case_id": case_id, "revision": state["revision"],
                             "outcome": answer["outcome"], "reply": answer["user_reply"],
                             "summary": state.get("summary", {})}}

    @staticmethod
    def _human_review(state: CaseState) -> dict:
        return {"review": interrupt(state["proposal"])}  # pauses here; resumed by CaseWorkflow.review

    def _apply_review(self, state: CaseState) -> dict:
        proposal, decision = state["proposal"], state["review"]
        case_id, pid = proposal["case_id"], proposal["proposal_id"]
        if decision.get("proposal_id") != pid:  # an approval only ever covers the proposal it names
            self.tracker.log(case_id, "review_refused", {"proposal_id": pid, "given": decision.get("proposal_id")})
            return {"action": {"kind": "refused", "posted": False, "proposal_id": pid}}
        if self._stale(state, case_id):
            self.tracker.close_review(pid, "stale")
            self.tracker.update(case_id, status="open")
            self.tracker.log(case_id, "stale", {"proposal_id": pid})
            return {"action": {"kind": "stale", "posted": False, "proposal_id": pid}}
        if decision["decision"] == "reject":
            self.tracker.close_review(pid, "rejected")
            self.tracker.update(case_id, status="open")
            self.tracker.log(case_id, "rejected", {"proposal_id": pid})
            return {"action": {"kind": "rejected", "posted": False, "proposal_id": pid}}
        text = decision.get("text") or proposal["reply"]
        final = pid if text == proposal["reply"] else proposal_id(case_id, proposal["revision"], text)
        status, labels = ON_APPROVAL[proposal["outcome"]]
        posted = self.tracker.post(case_id, final, text)
        self.tracker.note(case_id, final, json.dumps(proposal["summary"], ensure_ascii=False))
        self.tracker.update(case_id, status=status, add_labels=labels)
        self.tracker.close_review(pid, "edited" if final != pid else "approved")
        self.tracker.log(case_id, "approved", {"proposal_id": final, "edited": final != pid,
                                               "outcome": proposal["outcome"]})
        return {"action": {"kind": "posted", "posted": posted, "proposal_id": final,
                           "outcome": proposal["outcome"], "status": status}}

    # ---- entry points ----

    def run_turn(self, case_id: str) -> dict:
        """Run the graph on the case as it is now. Returns the action, or the proposal waiting."""
        revision = self.tracker.revision(case_id)
        thread = f"{case_id}:r{revision}"
        config = {"configurable": {"thread_id": thread}, "recursion_limit": RECURSION_LIMIT}
        out = self.graph.invoke({"case": case_to_dict(self.tracker.case_input(case_id)), "revision": revision}, config)
        if out.get("__interrupt__"):
            return {"status": "awaiting_review", "proposal": out["__interrupt__"][0].value}
        return {"status": "done", "outcome": out["answer"]["outcome"], "action": out.get("action")}

    def pending(self, case_id: str) -> dict | None:
        row = self.tracker.pending_review(case_id)
        if row is None:
            return None
        snapshot = self.graph.get_state({"configurable": {"thread_id": row["thread_id"]}})
        return snapshot.values.get("proposal")

    def review(self, case_id: str, decision: str, *, text: str | None = None,
               expected_proposal: str | None = None) -> dict:
        """Resume the paused graph with a maintainer's decision: approve | edit | reject."""
        if decision not in ("approve", "edit", "reject") or (decision == "edit") != bool(text):
            raise TrackerError("decide with approve, reject, or edit plus the new text")
        row = self.tracker.pending_review(case_id)
        if row is None:
            raise TrackerError(f"nothing is waiting for review on case {case_id!r}")
        if expected_proposal and expected_proposal != row["proposal_id"]:
            raise TrackerError(f"proposal {expected_proposal} is not the one waiting ({row['proposal_id']}); "
                               "look at it again before deciding")
        config = {"configurable": {"thread_id": row["thread_id"]}, "recursion_limit": RECURSION_LIMIT}
        out = self.graph.invoke(Command(resume={"decision": decision, "text": text,
                                                "proposal_id": row["proposal_id"]}), config)
        return out["action"]
