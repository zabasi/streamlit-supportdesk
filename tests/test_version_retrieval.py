"""Version-aware retrieval: version tags for the reranker (A) and newer release notes (B); invented data."""
import dataclasses

import numpy as np

from supportdesk.config import Settings
from supportdesk.pipelines.stages.llm_rerank import LLMReranker, with_rule
from supportdesk.pipelines.stages.version_notes import tag
from supportdesk.retrieval.release_slots import ReleaseSlots, newer_release_notes
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk


def chunk(cid, kind, version=None, text="some text"):
    return Chunk(chunk_id=cid, source_id=f"s-{cid}", source_type=kind, title=cid, text=text, version=version)


def test_each_candidate_gets_a_short_version_tag():
    assert tag(chunk("d", "documentation", "1.65.0"), "1.11.0") == "current documentation (Streamlit 1.65.0), newer than the reporter's 1.11.0"
    assert tag(chunk("d", "documentation", "1.65.0"), None) == "current documentation (Streamlit 1.65.0)"
    assert tag(chunk("r", "release_section", "1.46.0"), "1.11.0").endswith("newer than the reporter's 1.11.0: not in their version")
    assert tag(chunk("r", "release_section", "1.3.0"), "1.11.0").endswith("already in the reporter's 1.11.0")
    assert tag(chunk("i", "issue", "1.13"), None) == "another user's report, on Streamlit 1.13.0"


def test_the_rule_goes_just_before_the_output_instruction():
    prompt = "Rank things.\n- Be careful.\n- Return only a JSON object.\n"
    assert with_rule(prompt, "- RULE") == "Rank things.\n- Be careful.\n- RULE\n- Return only a JSON object.\n"
    assert with_rule("No instruction here.", "- RULE") == "No instruction here.\n- RULE\n"


def request(clients, settings, on):
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, rerank_versions=on))
    reranker = LLMReranker(clients, s)
    seen = []
    reranker.chat.complete = lambda messages, **kw: seen.append(messages) or ('{"ranking": ["C1", "C2"]}', {})
    case = CaseInput(case_id="c", title="Value lost", body="The value is lost.", environment={"streamlit_version": "1.11.0"})
    hits = [Hit(chunk("r", "release_section", "1.46.0"), 0.9, 1, "dense"), Hit(chunk("d", "documentation", "1.65.0"), 0.8, 2, "dense")]
    reranker.model_text(case, hits)
    return seen[-1]


def test_with_versions_on_the_request_states_the_version_and_tags_each_candidate(clients, settings):
    system, user = (m["content"] for m in request(clients, settings, True))
    assert "## The user's Streamlit version\n1.11.0" in user
    assert "Version: release note for 1.46.0, newer than the reporter's 1.11.0: not in their version" in user
    assert "Prefer sources that apply to the user's version" in system


def test_with_versions_off_the_request_is_unchanged(clients, settings):
    system, user = (m["content"] for m in request(clients, settings, False))
    assert "Streamlit version" not in user and "Version:" not in user and "user's version" not in system


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, vector, *, limit, as_of, source_types=None):
        found = [(i, 1.0 - i / 100) for i, c in enumerate(self.chunks) if not source_types or c.source_type in source_types]
        return found[:limit]


class FakeEmbedder:
    def embed(self, texts, *, stage):
        return [np.ones(3).tolist() for _ in texts], {}


class Retriever:
    def __init__(self, chunks):
        self.index, self.embedder = FakeIndex(chunks), FakeEmbedder()


CHUNKS = [chunk("r1", "release_section", "1.3.0"), chunk("r2", "release_section", "1.46.0"),
          chunk("d1", "documentation", "1.65.0"), chunk("r3", "release_section", "1.50.0"), chunk("r4", "release_section", "1.60.0")]


def test_only_release_notes_newer_than_the_reported_version_are_added():
    got = [h.chunk.chunk_id for h in newer_release_notes(Retriever(CHUNKS), "q", "1.11.0", 2)]
    assert got == ["r2", "r3"]  # newer than 1.11.0, best first, at most 2; never the documentation
    assert newer_release_notes(Retriever(CHUNKS), "q", None, 2) == []  # no stated version: nothing added
    assert newer_release_notes(Retriever(CHUNKS), "q", "1.11.0", 0) == []


def test_the_wrapper_passes_the_added_release_notes_to_the_reranker(settings):
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, release_slots=1))
    seen = []
    wrapped = ReleaseSlots(lambda case, query, hits: seen.append(hits) or hits, Retriever(CHUNKS), s)
    case = CaseInput(case_id="c", title="t", body="b", environment={"streamlit_version": "1.11.0"})
    base = [Hit(CHUNKS[2], 0.9, 1, "dense")]
    wrapped(case, "q", base)
    assert [h.chunk.chunk_id for h in seen[-1]] == ["d1", "r2"] and seen[-1][-1].rank == 2


def test_advanced_version_retrieval_is_advanced_with_only_the_two_settings(monkeypatch):
    from pathlib import Path

    from tests.test_candidates import flat
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    advanced = flat(Settings.load(env={}, overrides=["pipeline.name=advanced"]))
    other = flat(Settings.load(env={}, overrides=["pipeline.name=advanced-version-retrieval"]))
    assert {k: other[k] for k in other if other[k] != advanced[k]} == {"retrieval.rerank_versions": "True", "retrieval.release_slots": "2"}


