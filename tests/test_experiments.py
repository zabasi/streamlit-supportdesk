"""The single-change experiments: clean_query (the query) and doc_slots (the retriever)."""
import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.clean_query import CleanQuery, clean_report
from supportdesk.retrieval.retriever import DenseRetriever, DocSlotsRetriever
from supportdesk.types import CaseInput

REPORT = """### Checklist

- [X] I have searched the existing issues for similar issues.

### Summary

The text field is empty after pressing the button. See https://example.com/x

### Reproducible Code Example

```Python
import streamlit as st
from time import sleep
with st.expander("Inside"):
    st.text_area("Text", key="text")
raise StreamlitAPIException("boom")
```

### Debug info

- Streamlit version: 1.13.0
- Python version: 3.10.5

### Additional Information

<img width="799" src="https://user-images.githubusercontent.com/x.png">
Happens only with the expander.

---

Community voting on feature requests enables the Streamlit team to understand which features matter.
"""


def test_the_template_goes_and_the_meaning_stays():
    out = clean_report(REPORT)
    for kept in ("The text field is empty after pressing the button.", 'with st.expander("Inside"):',
                 'st.text_area("Text", key="text")', 'StreamlitAPIException("boom")',
                 "Happens only with the expander."):
        assert kept in out
    for dropped in ("Checklist", "searched the existing issues", "Streamlit version", "Python version",
                    "https://", "<img", "Community voting", "from time import sleep", "###", "---"):
        assert dropped not in out
    assert clean_report("Plain text with no template.") == "Plain text with no template."


def test_the_query_has_the_title_and_the_reporters_follow_ups_only():
    case = CaseInput(case_id="c", title="Expander loses state", body=REPORT,
                     history=[{"role": "assistant", "body": "Which version?"},
                              {"role": "user", "body": "Also on 1.32.0, see https://example.com"}])
    (query,) = CleanQuery(4000)(case, {})
    assert query.startswith("Expander loses state\n") and "Also on 1.32.0, see" in query
    assert "Which version?" not in query and "https://" not in query
    assert len(CleanQuery(50)(case, {})[0]) == 50


def test_the_index_can_search_one_source_type(index):
    probe = [1.0] + [0.0] * (index.dimension - 1)
    found = index.search(probe, limit=50, as_of=None, source_types=("documentation",))
    assert found and {index.chunks[i].source_type for i, _ in found} == {"documentation"}
    assert len(index.search(probe, limit=50, as_of=None)) > len(found)


def test_doc_slots_keep_places_for_documentation_in_relevance_order(index, clients):
    query = "st.session_state is empty after refresh"
    plain = DenseRetriever(index, clients.embedding).search(query, k=6)
    for slots in (1, 3):
        hits = DocSlotsRetriever(index, clients.embedding, doc_slots=slots).search(query, k=6)
        assert len(hits) == 6 and [h.rank for h in hits] == list(range(1, 7))
        assert sum(h.chunk.source_type == "documentation" for h in hits) >= slots
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    none = DocSlotsRetriever(index, clients.embedding, doc_slots=0).search(query, k=6)
    assert [h.chunk.chunk_id for h in none] == [h.chunk.chunk_id for h in plain]  # 0 slots = plain dense


def test_each_experiment_differs_from_the_baseline_in_exactly_one_way():
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    base = flat("baseline")
    for name, expected in (("clean_query", {"pipeline.name"}),
                           ("doc_slots", {"pipeline.name", "pipeline.retriever", "retrieval.doc_slots"})):
        other = flat(name)
        assert {k for k in base if base[k] != other[k]} == expected, name


def test_doc_slots_cannot_exceed_top_k():
    with pytest.raises(ConfigError, match="retrieval.doc_slots"):
        Settings.load(env={}, overrides=["retrieval.doc_slots=7"])
