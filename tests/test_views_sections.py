"""retrieval.report_views (title / code views as extra searches) and retrieval.rerank_show_sections."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.llm_rerank import LLMReranker, chunk_headings
from supportdesk.pipelines.stages.report_views import code_view, report_views
from supportdesk.pipelines.state import case_to_dict
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

BODY = """The state is lost.
```python
st.radio("Test", options=options, key=key)
@st.fragment(run_every=2)
def f(): pass
```
StreamlitAPIException: Values for st.data_editor cannot be set using st.session_state.
"""
CASE = CaseInput(case_id="c", title="Widget state not persisted with run_every", body=BODY)


def test_the_code_view_holds_commands_parameters_and_the_error():
    view = code_view(BODY)
    for word in ("st.radio", "st.fragment", "st.data_editor", "st.session_state", "run_every", "key", "options"):
        assert word in view, word
    assert "StreamlitAPIException: Values for st.data_editor" in view


def test_views_are_chosen_by_name_and_empty_ones_left_out():
    assert report_views(CASE, "none") == []
    assert report_views(CASE, "title") == ["Widget state not persisted with run_every"]
    assert len(report_views(CASE, "title,code")) == 2
    assert report_views(CaseInput(case_id="x", title="t", body="no code here"), "code") == []


def test_chunk_headings_skip_lines_inside_code_fences():
    text = "intro\n## Alpha\ntext\n```python\n# not a heading\n```\n### Beta ###\nmore"
    assert chunk_headings(text) == ["Alpha", "Beta"]


def test_the_reranker_sees_the_sections_a_chunk_contains(clients, settings):
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, rerank_show_sections=True))
    chunk = Chunk(chunk_id="p:0", source_id="doc:p", source_type="documentation", title="Button behavior",
                  text="## Buttons to modify st.session_state\ntext\n### Stateful button\nmore", start_char=0, end_char=60)
    LLMReranker(clients, s).model_text(CASE, [Hit(chunk, 1.0, 1, "dense")])
    assert "Sections: Buttons to modify st.session_state · Stateful button" in clients.chat.seen[-1][1]["content"]


def test_advanced_searches_the_views_besides_the_report(index, clients, settings):
    s = dataclasses.replace(settings,
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                            retrieval=dataclasses.replace(settings.retrieval, report_views="title,code"))
    p = make_pipeline("advanced", "dense_doc_slots", index, clients, s)
    searched = []
    original = p.retriever.search
    p.retriever.search = lambda q, **kw: searched.append(q) or original(q, **kw)
    p._retrieve_node({"case": case_to_dict(CASE), "queries": ["the cleaned report"]})
    assert searched[0] == "the cleaned report" and searched[1:] == report_views(CASE, "title,code")


def test_the_settings_are_checked():
    s = Settings.load(env={})
    assert (s.retrieval.report_views, s.retrieval.rerank_show_sections) == ("none", False)  # off by default
    with pytest.raises(ConfigError, match="report_views"):
        Settings.load(env={}, overrides=["retrieval.report_views=summary"])
