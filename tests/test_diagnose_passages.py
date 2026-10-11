"""The passage diagnostic (scripts/diagnose_passages.py): where each gold passage ends up."""
import importlib.util

from tests.helpers import ROOT

spec = importlib.util.spec_from_file_location("diagnose_passages", ROOT / "scripts" / "diagnose_passages.py")
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)

TEXT = "Intro.\n\n### Caveats\n\nSession State is tied to a WebSocket connection and resets on reload.\n"
SPAN = {"source_id": "doc:a", "start_char": TEXT.index("### Caveats"), "end_char": len(TEXT) - 1,
        "text": TEXT[TEXT.index("### Caveats"):len(TEXT) - 1]}
GOLD = {"source_id": "doc:a", "start_char": TEXT.index("### Caveats"), "end_char": len(TEXT)}
INTRO = {"source_id": "doc:a", "start_char": 0, "end_char": TEXT.index("### Caveats")}
OTHER = {"source_id": "doc:b", "start_char": 0, "end_char": 40}


def test_each_passage_lands_in_exactly_one_stage():
    assert diagnose.classify(SPAN, [GOLD, OTHER], [GOLD], 0.8) == "in the prompt"
    assert diagnose.classify(SPAN, [GOLD, OTHER], [OTHER], 0.8) == "dropped by the reranker"
    assert diagnose.classify(SPAN, [INTRO, OTHER], [INTRO], 0.8) == "right page, other chunk"
    assert diagnose.classify(SPAN, [OTHER], [OTHER], 0.8) == "page not among the candidates"


def test_the_holding_chunk_is_the_one_that_covers_most_of_the_passage():
    from supportdesk.types import Chunk
    def chunk(cid, view):
        return Chunk(chunk_id=cid, source_id=view["source_id"], source_type="documentation", title="t",
                     text="x", url="u", start_char=view["start_char"], end_char=view["end_char"])
    best, cover = diagnose.holding_chunk(SPAN, [chunk("a:0", INTRO), chunk("a:1", GOLD), chunk("b:0", OTHER)])
    assert best == "a:1" and cover == 1.0


def test_the_report_counts_every_passage_once():
    from collections import Counter
    funnel = Counter({"in the prompt": 2, "page not among the candidates": 1})
    rows = [{"case": "C", "page": "p.md", "passage": "### A | B", "stage": "in the prompt", "chunk": "x",
             "cover": 1.0, "doc_rank": 3}]
    text = diagnose.render(funnel, rows, "advanced", "dev")
    assert "| in the prompt | 2 | 67% |" in text and "| dropped by the reranker | 0 | 0% |" in text
    assert "### A \\| B" in text  # a pipe inside a passage cannot break the table
