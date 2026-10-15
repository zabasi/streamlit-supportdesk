"""retrieval.decompose = false: advanced searches with the cleaned report alone."""
import dataclasses

from supportdesk.config import Settings
from supportdesk.pipelines.stages.clean_query import CleanQuery
from supportdesk.pipelines.stages.rewrite_query import QueryRewriter
from supportdesk.registry import make_pipeline
from supportdesk.types import CaseInput

REWRITE = "You help find the Streamlit documentation"  # a line of the decompose prompt
MECHANISMS = "Mechanisms involved"


def advanced(index, clients, settings, decompose):
    s = dataclasses.replace(settings,
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                            retrieval=dataclasses.replace(settings.retrieval, decompose=decompose,
                                                          rerank_show_concepts=True))
    s = dataclasses.replace(s, prompts=dataclasses.replace(
        s.prompts, rerank=s.prompts.rerank.with_name("rerank_docs_concepts_system.txt")))
    return make_pipeline("advanced", "dense_doc_slots", index, clients, s)


def test_without_decompose_advanced_makes_no_rewrite_call_and_shows_no_mechanisms(index, clients, settings):
    pipeline = advanced(index, clients, settings, decompose=False)
    assert isinstance(pipeline.query, CleanQuery)
    before = len(clients.chat.seen)
    result = pipeline.run(CaseInput(case_id="c", title="Session state lost", body="st.session_state lost on rerun"))
    sent = clients.chat.seen[before:]
    assert result.retrieved
    assert not any(REWRITE in m[0]["content"] for m in sent)  # no concept queries written
    assert not any(MECHANISMS in part["content"] for m in sent for part in m)  # no list, in prompt or message


def test_the_default_keeps_decompose(index, clients, settings):
    assert isinstance(advanced(index, clients, settings, decompose=True).query, QueryRewriter)
    assert Settings.load(env={}, overrides=["pipeline.name=advanced"]).retrieval.decompose is True
    assert Settings.load(env={}, overrides=["pipeline.name=advanced",
                                            "retrieval.decompose=false"]).retrieval.decompose is False
