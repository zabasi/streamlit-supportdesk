"""Symptom questions per documentation section (doc2query) and the question_doc_slots retriever."""
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from supportdesk.config import Settings
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.index import IndexUnusable
from supportdesk.retrieval.retriever import DocSlotsRetriever
from supportdesk.retrieval.section_questions import (
    QuestionBank,
    QuestionExpandedRetriever,
    generate_questions,
)
from supportdesk.types import CaseInput

ROOT = Path(__file__).resolve().parents[1]


def bank_for(index, clients):
    docs = [c for c in index.chunks if c.source_type == "documentation"][:6]
    sections = [{"source_id": c.source_id, "start_char": c.start_char, "end_char": c.end_char,
                 "title": c.title, "heading": f"h{n}"} for n, c in enumerate(docs)]
    questions = [[" ".join(c.text.split()[:12]), f"question {n} about something else"] for n, c in enumerate(docs)]
    vectors = clients.embedding.embed([q for qs in questions for q in qs])[0]
    return QuestionBank(sections, questions, vectors, clients.embedding_model)


def test_questions_are_parsed_and_a_bad_answer_gives_none(clients):
    sections = [{"source_id": "s", "start_char": 0, "end_char": 5, "title": "T", "heading": "H"}] * 2
    clients.chat.script[clients.chat.calls + 1] = json.dumps({"questions": ["  why is  it reset?", "", "x"]})
    clients.chat.script[clients.chat.calls + 2] = "not json"
    got = generate_questions(sections, {"s": "hello"}, clients.chat, prompt="write {n}", n=2)
    assert got == [["why is it reset?", "x"], []]


def test_the_bank_round_trips_and_refuses_another_model(index, clients, tmp_path):
    bank = bank_for(index, clients)
    bank.save(tmp_path / "q")
    again = QuestionBank.load(tmp_path / "q", expect_model=clients.embedding_model)
    assert again.questions == bank.questions and again.sections == bank.sections
    with pytest.raises(IndexUnusable, match="embedded with"):
        QuestionBank.load(tmp_path / "q", expect_model="another-model")
    with pytest.raises(IndexUnusable, match="build_section_questions"):
        QuestionBank.load(tmp_path / "none")


def test_each_section_is_ranked_by_its_best_question(index, clients):
    bank = bank_for(index, clients)
    last = len(bank.sections) - 1
    vec = clients.embedding.embed([bank.questions[last][0]])[0][0]
    assert bank.ranking(vec)[0][0] == last  # its own question matches best
    assert len(bank.ranking(vec)) == len(bank.sections)


def test_the_base_candidates_are_kept_and_new_sections_added(index, clients):
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=3)
    r = QuestionExpandedRetriever(base, bank_for(index, clients), extra=2)
    plain = [h.chunk.chunk_id for h in base.search("widget value lost", k=4)]
    hits = r.search("widget value lost", k=4)
    assert [h.chunk.chunk_id for h in hits[: len(plain)]] == plain
    added = hits[len(plain):]
    assert len(added) <= 2 and {h.source for h in added} <= {"question"}
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_advanced_runs_with_it_and_the_diagnostic_counts(index, clients, settings, tmp_path):
    bank_for(index, clients).save(tmp_path / "q")
    s = dataclasses.replace(settings, paths=dataclasses.replace(settings.paths, section_questions=tmp_path / "q"),
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"))
    case = CaseInput(case_id="c", title="Widget value lost", body="st.text_input value lost")
    assert make_pipeline("advanced", "question_doc_slots", index, clients, s).run(case).retrieved
    spec = importlib.util.spec_from_file_location("diagnose_questions", ROOT / "scripts" / "diagnose_questions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    doc = next(c for c in index.chunks if c.source_type == "documentation")
    gold = SimpleNamespace(case_id="g", input=case, evidence_spans=[
        {"source_id": doc.source_id, "start_char": doc.start_char, "end_char": doc.end_char}])
    found = module.diagnose(s, index, clients, [gold])
    assert found["rows"] and found["new"] <= found["missed"]


def test_question_sections_defaults_to_five():
    assert Settings.load(env={}).retrieval.question_sections == 5
