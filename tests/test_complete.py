"""retrieval.complete: after reranking, the kept documentation chunks are repaired at their edges."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.stages.evidence import complete_hits, merge_overlaps
from supportdesk.pipelines.stages.sections import section_spans
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

ALPHA = "## Alpha\n" + "alpha words here. " * 12 + "\n\n"
BETA = "## Beta\n" + "beta words here. " * 12 + "\n\n"
GAMMA = "## Gamma\n" + "gamma words here. " * 12 + "\n"
PAGE = "# Title\nIntro line.\n\n" + ALPHA + BETA + GAMMA
A, B, G = PAGE.index("## Alpha"), PAGE.index("## Beta"), PAGE.index("## Gamma")


def chunk(a, b, cid=None, kind="documentation"):
    return Chunk(chunk_id=cid or f"p:{a}", source_id="doc:p", source_type=kind, title="Page", text=PAGE[a:b],
                 start_char=a, end_char=b)


def run(hit_chunks, mode, cap=900, neighbours=None):
    hits = [Hit(c, 1.0, n, "dense") for n, c in enumerate(hit_chunks, start=1)]
    out = complete_hits(hits, {"doc:p": PAGE}, {"doc:p": neighbours or hit_chunks}, mode=mode, cap=cap)
    return [(h.chunk.start_char, h.chunk.end_char) for h in out], out


def test_the_page_has_the_expected_sections():
    assert [a for a, _ in section_spans(PAGE)] == [0, A, B, G]


def test_section_grows_a_chunk_to_the_whole_sections_it_touches():
    middle = chunk(A + 40, B + 40)  # starts inside Alpha, ends inside Beta
    spans, out = run([middle], "section")
    assert spans == [(A, G)] and out[0].chunk.text == PAGE[A:G]  # all of Alpha and Beta, exact page text


def test_trim_drops_a_short_leading_fragment_but_never_the_main_part():
    short_lead = chunk(B - 30, G - 10)  # the end of Alpha, then most of Beta
    assert run([short_lead], "trim")[0] == [(B, G - 10)]  # now starts at the Beta heading
    long_lead = chunk(A + 10, B + 20)  # mostly Alpha, with the start of Beta: keep the Alpha part
    assert run([long_lead], "trim")[0] == [(A + 10, B + 20)]


def test_section_trim_starts_at_the_heading_and_completes_the_last_section():
    piece = chunk(B - 30, G + 30)  # the end of Alpha, all of Beta, the start of Gamma
    assert run([piece], "section_trim")[0] == [(B, len(PAGE))]


def test_neighbors_add_the_chunk_before_and_after():
    tiles = [chunk(0, A + 60, "p:0"), chunk(A + 40, B + 60, "p:1"), chunk(B + 40, len(PAGE), "p:2")]
    spans, _ = run([tiles[1]], "neighbors", neighbours=tiles)
    assert spans == [(0, len(PAGE))]


def test_a_repair_beyond_the_cap_falls_back_to_one_side_then_none():
    from supportdesk.tokenizer import count_tokens
    middle = chunk(A + 40, B + 40)
    forward_only = count_tokens(PAGE[A + 40:G])
    assert run([middle], "section", cap=forward_only)[0] == [(A + 40, G)]  # the whole repair is too long
    assert run([middle], "section", cap=1)[0] == [(A + 40, B + 40)]  # nothing fits: kept as it is


def test_other_sources_and_mode_none_are_untouched():
    issue = chunk(A + 40, B + 40, kind="issue")
    assert run([issue], "section")[0] == [(A + 40, B + 40)]
    assert run([chunk(A + 40, B + 40)], "none")[0] == [(A + 40, B + 40)]


def test_repaired_pieces_that_now_overlap_are_merged():
    one, two = chunk(A + 10, A + 60, "p:1"), chunk(A + 100, A + 150, "p:2")  # both inside Alpha
    _, out = run([one, two], "section")
    merged = merge_overlaps(out, {"doc:p": PAGE})
    assert [(h.chunk.start_char, h.chunk.end_char) for h in merged] == [(A, B)]  # Alpha once


def test_advanced_runs_with_each_completion(index, clients, settings):
    for mode in ("neighbors", "section", "trim", "section_trim"):
        s = dataclasses.replace(settings,
                                pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                                retrieval=dataclasses.replace(settings.retrieval, complete=mode))
        result = make_pipeline("advanced", "dense_doc_slots", index, clients, s).run(
            CaseInput(case_id="c", title="Session state lost", body="st.session_state disappears after rerun"))
        assert result.retrieved, mode


def test_complete_is_a_checked_setting():
    assert Settings.load(env={}).retrieval.complete == "none"  # advanced unchanged by default
    with pytest.raises(ConfigError, match="retrieval.complete"):
        Settings.load(env={}, overrides=["retrieval.complete=maybe"])


def test_trim_keeps_a_page_introduction():
    """The text under the page title before the first ## section is an introduction, not a leftover."""
    opening = chunk(0, A + 60)  # "# Title / Intro line." then the start of Alpha
    assert run([opening], "trim")[0] == [(0, A + 60)]
    assert run([opening], "section_trim")[0] == [(0, B)]  # kept, and Alpha completed


NESTED = ("# Title\nIntro line.\n\n## Alpha\nAlpha opening.\n\n### A1\n" + "one words here. " * 10 + "\n\n### A2\n"
          + "two words here. " * 10 + "\n\n### A3\n" + "three words here. " * 10 + "\n\n## Beta\n" + "beta words. " * 10 + "\n")
N_ALPHA, N_A1, N_A2, N_A3, N_BETA = (NESTED.index(h) for h in ("## Alpha", "### A1", "### A2", "### A3", "## Beta"))


def nested_run(a, b, mode):
    c = Chunk(chunk_id="n:0", source_id="doc:n", source_type="documentation", title="Nested", text=NESTED[a:b],
              start_char=a, end_char=b)
    out = complete_hits([Hit(c, 1.0, 1, "dense")], {"doc:n": NESTED}, {"doc:n": [c]}, mode=mode, cap=900)
    return out[0].chunk.start_char, out[0].chunk.end_char


def test_nested_sections_have_their_own_extent():
    from supportdesk.pipelines.stages.sections import nested_sections
    assert nested_sections(NESTED) == [(N_ALPHA, N_BETA, 2), (N_A1, N_A2, 3), (N_A2, N_A3, 3), (N_A3, N_BETA, 3),
                                       (N_BETA, len(NESTED), 2)]


def test_subsection_grows_only_to_the_subsections_the_chunk_touches():
    spanning = (N_A1 + 30, N_A2 + 30)  # from inside A1 to inside A2
    assert nested_run(*spanning, "subsection") == (N_A1, N_A3)  # A1 and A2 only
    assert nested_run(*spanning, "section") == (N_ALPHA, N_BETA)  # the whole of Alpha, with A3
    assert nested_run(N_A2 + 20, N_A2 + 60, "subsection") == (N_A2, N_A3)  # inside one subsection: just it


def test_subsection_leaves_a_page_introduction_as_it_is():
    assert nested_run(5, N_A1 + 30, "subsection")[0] == 5  # before the first heading: no section to grow to
