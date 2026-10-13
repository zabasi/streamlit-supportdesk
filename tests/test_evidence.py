"""Less noise without cutting chunks: merge overlapping chunks, keep only chunks graded relevant."""
import dataclasses
import json

from supportdesk.pipelines.stages.evidence import GradedReranker, merge_overlaps
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, Chunk

PAGE = "".join(f"Line {n} of the page about session state.\n" for n in range(40))


def hit(a, b, rank, kind="documentation", source="doc:p"):
    text = PAGE[a:b] if source == "doc:p" else "Issue text."
    return Hit(Chunk(chunk_id=f"{source}:{a}", source_id=source, source_type=kind, title="t", text=text, url="u",
                     start_char=a, end_char=b), 1.0 / rank, rank, "dense")


def test_overlapping_or_touching_chunks_of_a_page_become_one_span():
    pages = {"doc:p": PAGE}
    first, overlapping, touching = hit(0, 400, 1), hit(300, 700, 3), hit(700, 900, 4)
    apart, issue = hit(1200, 1500, 2), hit(0, 11, 5, kind="issue", source="issue:1")
    merged = merge_overlaps([first, apart, overlapping, touching, issue], pages)
    spans = [(h.chunk.source_id, h.chunk.start_char, h.chunk.end_char) for h in merged]
    assert spans == [("doc:p", 0, 900), ("doc:p", 1200, 1500), ("issue:1", 0, 11)]
    assert merged[0].chunk.text == PAGE[0:900] and [h.rank for h in merged] == [1, 2, 3]
    total_before = sum(h.chunk.end_char - h.chunk.start_char for h in (first, overlapping, touching))
    assert merged[0].chunk.end_char - merged[0].chunk.start_char < total_before  # the shared text once


def graded(clients, settings, ranking, **retrieval):
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, **{"top_k": 6, "min_chunks": 2, "rerank_keep_docs": 1, **retrieval}))
    hits = [hit(n * 100, n * 100 + 90, n + 1, kind="documentation" if n < 3 else "issue",
                source="doc:p" if n < 3 else f"issue:{n}") for n in range(8)]
    clients.chat.script[clients.chat.calls + 1] = json.dumps({"ranking": ranking}) if ranking else "not json"
    case = CaseInput(case_id="c", title="Session state lost", body="st.session_state is empty")
    return hits, GradedReranker(clients, s)(case, "q", hits)


def test_chunks_graded_not_relevant_are_dropped(clients, settings):
    ranking = [{"label": "C4", "grade": "relevant"}, {"label": "C1", "grade": "relevant"},
               {"label": "C5", "grade": "partly"}] + [{"label": f"C{n}", "grade": "not"} for n in (2, 3, 6, 7, 8)]
    hits, kept = graded(clients, settings, ranking)
    assert [h.chunk.chunk_id for h in kept] == [hits[3].chunk.chunk_id, hits[0].chunk.chunk_id, hits[4].chunk.chunk_id]


def test_the_floor_and_the_documentation_minimum_never_bring_back_irrelevant_docs(clients, settings):
    ranking = [{"label": "C5", "grade": "relevant"}] + [{"label": f"C{n}", "grade": "not"} for n in (1, 2, 3, 4, 6, 7, 8)]
    hits, kept = graded(clients, settings, ranking, rerank_keep_docs=3)
    assert len(kept) == 2 and kept[0].chunk.chunk_id == hits[4].chunk.chunk_id  # floor of 2, filled by rank
    assert kept[1].chunk.chunk_id == hits[0].chunk.chunk_id  # the next in the ranking, not a promoted doc


def test_unusable_or_ungraded_output_behaves_like_the_plain_reranker(clients, settings):
    _, kept = graded(clients, settings, None)
    assert len(kept) == 6
    _, kept = graded(clients, settings, ["C2", "C1"])  # a ranking without grades
    assert len(kept) == 6 and kept[0].chunk.chunk_id.endswith(":100")


def test_the_merged_and_graded_experiments_change_only_their_stage():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    base = flat(d)
    assert {k for k in base if base[k] != flat(f"{d}_merged")[k]} == {"pipeline.name"}
    g = flat(f"{d}_graded")
    assert {k for k in base if base[k] != g[k]} == {"pipeline.name", "prompts.rerank"}
    assert str(g["prompts.rerank"]).endswith("rerank_graded_system.txt") and g["retrieval.min_chunks"] == 2


SECTIONED = ("Session State\n\nIntro text.\n\n## Initialize values\n\nSet defaults first, before widgets.\n\n"
             "## Caveats\n\nButtons cannot be set through Session State.\n\n## Serialization\n\nPickle rules apply.\n")


def extractor(clients, settings, output):
    from supportdesk.pipelines.stages.evidence import ExtractingReranker
    page = Chunk(chunk_id="doc:s:0", source_id="doc:s", source_type="documentation", title="Session State",
                 text=SECTIONED, url="u", start_char=0, end_char=len(SECTIONED))
    issue = Chunk(chunk_id="issue:1:0", source_id="issue:1", source_type="issue", title="t", text="Issue body.",
                  url="u", start_char=0, end_char=11)
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, top_k=6, doc_slots=0))
    stage = ExtractingReranker(clients, s, [page])
    hits = [Hit(page, 0.9, 1, "dense"), Hit(issue, 0.5, 2, "dense")]
    clients.chat.script[clients.chat.calls + 1] = json.dumps({"ranking": ["C1", "C2"]})  # the reranking call
    clients.chat.script[clients.chat.calls + 2] = output                                 # the extraction call
    case = CaseInput(case_id="c", title="Button state", body="Setting a button through session state fails")
    return stage, stage(case, "q", hits)


def test_the_model_keeps_only_the_parts_that_help(clients, settings):
    _, kept = extractor(clients, settings, json.dumps({"keep": ["P3", "P5"]}))
    shown = clients.chat.seen[-1][1]["content"]
    assert "P5: [issue]" in shown and "Pickle rules apply." in shown  # full text of every part
    assert [h.chunk.text.strip().splitlines()[0] for h in kept] == ["## Caveats", "Issue body."]
    caveats = kept[0].chunk
    assert SECTIONED[caveats.start_char:caveats.end_char] == caveats.text  # exact offsets


def test_neighbouring_kept_parts_are_joined_and_unusable_output_keeps_everything(clients, settings):
    _, kept = extractor(clients, settings, json.dumps({"keep": ["P2", "P3"]}))
    assert len(kept) == 1 and kept[0].chunk.text.startswith("## Initialize values") and "## Caveats" in kept[0].chunk.text
    for output in ("not json", json.dumps({"keep": []})):
        _, kept = extractor(clients, settings, output)
        assert [h.chunk.chunk_id for h in kept] == ["doc:s:0", "issue:1:0"]  # as merged would give


def test_the_extract_experiment_differs_from_merged_only_in_its_stage():
    from supportdesk.config import Settings
    from supportdesk.registry import PIPELINES
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    a, b = flat(f"{d}_merged"), flat(f"{d}_merged_extract")
    assert {k for k in a if a[k] != b[k]} == {"pipeline.name"}
    assert PIPELINES[f"{d}_merged_extract"].__doc__.startswith("decompose_merged, then one model call")


def extractor_v2(clients, settings, output, *, protect=1, concepts=True):
    from supportdesk.pipelines.stages.evidence import ExtractingReranker
    page = Chunk(chunk_id="doc:s:0", source_id="doc:s", source_type="documentation", title="Session State",
                 text=SECTIONED, url="u", start_char=0, end_char=len(SECTIONED))
    issue = Chunk(chunk_id="issue:1:0", source_id="issue:1", source_type="issue", title="t", text="Issue body.",
                  url="u", start_char=0, end_char=11)
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, top_k=6, doc_slots=0, extract_mode="drop", extract_protect_top=protect,
        extract_show_concepts=concepts, keep_report_query=True, max_queries=2))
    stage = ExtractingReranker(clients, s, [page])
    hits = [Hit(page, 0.9, 1, "dense"), Hit(issue, 0.5, 2, "dense")]
    calls = clients.chat.calls
    clients.chat.script[calls + 1] = json.dumps({"ranking": ["C2", "C1"]})  # reranking: the issue first
    if concepts:
        clients.chat.script[calls + 2] = json.dumps({"queries": ["widget initialization from session state",
                                                                 "button state cannot be set"]})
    clients.chat.script[calls + (3 if concepts else 2)] = output
    case = CaseInput(case_id="c", title="Button state", body="Setting a button through session state fails")
    return stage(case, "q", hits)


def test_drop_mode_keeps_everything_but_the_dropped_parts(clients, settings):
    kept = extractor_v2(clients, settings, json.dumps({"drop": ["P5"]}), protect=0)
    shown = clients.chat.seen[-1][1]["content"]
    assert "## Mechanisms involved\n- widget initialization from session state" in shown
    starts = [h.chunk.text.strip().splitlines()[0] for h in kept]
    assert starts == ["Issue body.", "Session State"]  # the issue, then the page minus Serialization, joined
    assert "## Serialization" not in kept[1].chunk.text and "## Caveats" in kept[1].chunk.text


def test_parts_of_the_top_chunk_are_never_dropped_and_unusable_output_keeps_all(clients, settings):
    kept = extractor_v2(clients, settings, json.dumps({"drop": ["P1", "P5"]}))  # P1 is the top-ranked issue
    assert kept[0].chunk.text == "Issue body."
    kept = extractor_v2(clients, settings, "not json")
    assert [h.chunk.chunk_id for h in kept] == ["issue:1:0", "doc:s:0"]
    kept = extractor_v2(clients, settings, json.dumps({"drop": []}), concepts=False)
    assert "Mechanisms involved" not in clients.chat.seen[-1][1]["content"] and len(kept) == 2


def test_the_drop_experiment_changes_only_the_extraction_settings():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    a, b = flat(f"{d}_merged_extract"), flat(f"{d}_merged_extract_drop")
    assert {k for k in a if a[k] != b[k]} == {"pipeline.name", "retrieval.extract_mode",
                                                "retrieval.extract_show_concepts", "retrieval.extract_protect_top",
                                                "prompts.extract"}
    assert (b["retrieval.extract_mode"], b["retrieval.extract_show_concepts"], b["retrieval.extract_protect_top"]) == (
        "drop", True, 1)


def test_the_smaller_fixed_chunk_experiments_change_only_size_room_and_index():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    base = flat(f"{d}_merged")
    size_keys = {"pipeline.name", "chunking.max_tokens", "chunking.overlap_tokens", "paths.index_dir"}
    for size, overlap, k in ((320, 80, 9), (384, 96, 8)):
        six, more = flat(f"{d}_merged_fixed{size}"), flat(f"{d}_merged_fixed{size}_top{k}")
        assert {key for key in base if base[key] != six[key]} == size_keys
        assert (six["chunking.max_tokens"], six["chunking.overlap_tokens"]) == (size, overlap)
        assert {key for key in base if base[key] != more[key]} == size_keys | {
            "retrieval.top_k", "retrieval.candidate_k", "retrieval.doc_slots"}
        assert (more["retrieval.top_k"], more["context.max_tokens"]) == (k, 3200)
        assert six["paths.index_dir"] == more["paths.index_dir"]


LONG = ("Widget behavior\n\nIntro.\n\n" + "## Anatomy\n\n" + "a " * 120 + "\n\n## Order of operations\n\n"
        + "b " * 200 + "\n\n### Example\n\n" + "c " * 120 + "\n\n## Life cycle\n\n" + "d " * 120 + "\n")


def page_chunks():
    step = len(LONG) // 5
    bounds = [(n * step, min(len(LONG), (n + 1) * step + 20)) for n in range(5)]
    bounds[-1] = (bounds[-1][0], len(LONG))
    return [Chunk(chunk_id=f"doc:w:{n}", source_id="doc:w", source_type="documentation", title="Widget behavior",
                  text=LONG[a:b], url="u", start_char=a, end_char=b) for n, (a, b) in enumerate(bounds)]


def expander(clients, settings, mode, cap=30):
    from supportdesk.pipelines.stages.evidence import ExpandingReranker
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, expand=mode, expand_max_candidates=cap))
    return ExpandingReranker(clients, s, page_chunks())


def test_page_expansion_adds_the_pages_other_chunks_nearest_first(clients, settings):
    chunks = page_chunks()
    issue = Hit(Chunk(chunk_id="issue:1:0", source_id="issue:1", source_type="issue", title="t", text="x", url="u",
                      start_char=0, end_char=1), 0.5, 2, "dense")
    found = [Hit(chunks[2], 0.9, 1, "dense"), issue]
    out = expander(clients, settings, "page").expand(found)
    ids = [h.chunk.chunk_id for h in out]
    assert ids[0] == "doc:w:2" and set(ids[1:5]) == {"doc:w:1", "doc:w:3", "doc:w:0", "doc:w:4"}
    assert ids[1:3] == ["doc:w:1", "doc:w:3"] or ids[1:3] == ["doc:w:3", "doc:w:1"]  # nearest first
    assert ids[-1] == "issue:1:0" and [h.source for h in out[1:5]] == ["expand"] * 4  # issues are not expanded
    assert [h.rank for h in out] == list(range(1, len(out) + 1))
    assert len(expander(clients, settings, "page", cap=3).expand(found)) == 3  # the cap holds
    assert expander(clients, settings, "none").expand(found) == found


def test_section_expansion_adds_only_chunks_sharing_a_section(clients, settings):
    chunks = page_chunks()
    order = (LONG.index("## Order of operations"), LONG.index("## Life cycle"))
    inside = next(c for c in chunks if order[0] < c.start_char < order[1])
    out = expander(clients, settings, "section").expand([Hit(inside, 0.9, 1, "dense")])
    for h in out[1:]:  # every added chunk overlaps the "Order of operations" section
        assert h.chunk.start_char < order[1] and h.chunk.end_char > order[0]
    page = expander(clients, settings, "page").expand([Hit(inside, 0.9, 1, "dense")])
    assert len(out) < len(page)  # fewer than the whole page


def test_the_expansion_experiments_change_only_the_expansion():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    base = flat(f"{d}_merged")
    for mode in ("page", "section"):
        other = flat(f"{d}_merged_expand_{mode}")
        assert {k for k in base if base[k] != other[k]} == {"pipeline.name", "retrieval.expand"}
        assert other["retrieval.expand"] == mode and other["retrieval.expand_max_candidates"] == 30


def test_the_reranker_sees_the_concept_queries_only_when_asked(clients, settings):
    from supportdesk.pipelines.stages.llm_rerank import LLMReranker
    hits = [Hit(c, 1.0 / (n + 1), n + 1, "dense") for n, c in enumerate(page_chunks()[:3])]
    case = CaseInput(case_id="c", title="Widget loses value", body="The value resets after a rerun")
    plain = LLMReranker(clients, settings)
    clients.chat.script[clients.chat.calls + 1] = json.dumps({"ranking": ["C1", "C2", "C3"]})
    plain(case, "q", hits)
    assert "Mechanisms involved" not in clients.chat.seen[-1][1]["content"]  # the default is unchanged
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, rerank_show_concepts=True, keep_report_query=True, max_queries=2))
    with_concepts = LLMReranker(clients, s)
    calls = clients.chat.calls
    clients.chat.script[calls + 1] = json.dumps({"queries": ["widget life cycle across reruns", "widget keys"]})
    clients.chat.script[calls + 2] = json.dumps({"ranking": ["C2", "C1", "C3"]})
    ranked = with_concepts(case, "q", hits)
    shown = clients.chat.seen[-1][1]["content"]
    assert "## Mechanisms involved\n- widget life cycle across reruns\n- widget keys" in shown
    assert ranked[0].chunk.chunk_id == hits[1].chunk.chunk_id


def test_the_reranker_view_experiments_change_only_what_the_reranker_sees():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    base = flat(f"{d}_merged")
    for suffix, changed in (("_merged_focused", {"retrieval.rerank_snippet"}),
                            ("_merged_concepts", {"retrieval.rerank_show_concepts", "prompts.rerank"}),
                            ("_merged_focused_concepts", {"retrieval.rerank_snippet", "retrieval.rerank_show_concepts",
                                                          "prompts.rerank"})):
        other = flat(d + suffix)
        assert {k for k in base if base[k] != other[k]} == {"pipeline.name", *changed}, suffix


def test_branch_expansion_adds_the_sections_parent_and_its_children(clients, settings):
    from supportdesk.pipelines.stages.evidence import ExpandingReranker
    tree = [("intro", ["Widget behavior"]), ("anatomy", ["Widget behavior", "Anatomy"]),
            ("life", ["Widget behavior", "Widget life cycle"]),
            ("not-yet", ["Widget behavior", "Widget life cycle", "Calling a widget command when it doesn't exist"]),
            ("exists", ["Widget behavior", "Widget life cycle", "Calling a widget command when it exists"]),
            ("cleanup", ["Widget behavior", "Widget life cycle", "Widget clean-up process"]),
            ("binding", ["Widget behavior", "Binding widgets to query parameters"])]
    chunks = [Chunk(chunk_id=f"doc:w:{name}", source_id="doc:w", source_type="documentation", title="Widget behavior",
                    text=name, url="u", start_char=n * 10, end_char=n * 10 + 9, meta={"heading_path": path})
              for n, (name, path) in enumerate(tree)]
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(settings.retrieval, expand="branch"))
    stage = ExpandingReranker(clients, s, chunks)
    found = stage.expand([Hit(chunks[4], 0.9, 1, "dense")])  # "### Calling a widget command when it exists"
    assert {h.chunk.chunk_id.rsplit(":", 1)[-1] for h in found} == {"exists", "life", "not-yet", "cleanup"}
    assert [h.source for h in found[1:]] == ["expand"] * 3
    assert stage.expand([Hit(chunks[0], 0.9, 1, "dense")]) == [Hit(chunks[0], 0.9, 1, "dense")]  # intro: no branch


def test_the_structure_experiments_change_only_chunking_room_and_expansion():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    base, d = flat("advanced"), "clean_doc_slots_rerank_docs_prompt_decompose"
    structure = {"pipeline.name", "chunking.method", "chunking.context_header", "retrieval.top_k",
                 "retrieval.candidate_k", "retrieval.doc_slots", "retrieval.rerank_keep_docs", "paths.index_dir"}
    ref, branch = flat(f"{d}_structure_advanced"), flat(f"{d}_structure_advanced_branch")
    assert {k for k in base if base[k] != ref[k]} == structure
    assert {k for k in ref if ref[k] != branch[k]} == {"pipeline.name", "retrieval.expand", "retrieval.expand_max_candidates"}
    assert (branch["retrieval.expand"], branch["context.max_tokens"]) == ("branch", 3200)


def corrective(clients, settings, verdicts, turns=1):
    from supportdesk.pipelines.stages.evidence import CorrectiveReranker
    chunks = page_chunks()
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, corrective_turns=turns, top_k=2, doc_slots=0))

    class Retriever:
        def search(self, query, *, k, as_of=None):
            self.query = query
            return [Hit(chunks[4], 0.8, 1, "dense"), Hit(chunks[0], 0.7, 2, "dense")]

    retriever = Retriever()
    stage = CorrectiveReranker(clients, s, chunks, retriever)
    hits = [Hit(chunks[0], 0.9, 1, "dense"), Hit(chunks[1], 0.5, 2, "dense")]
    calls = clients.chat.calls
    script = [json.dumps({"ranking": ["C1", "C2"]})]  # the first reranking
    for verdict in verdicts:
        script.append(verdict)  # the judge
        if verdict == "no":
            script += ["widget clean-up after page switch", json.dumps({"ranking": ["C3", "C1", "C2"]})]
    for n, text in enumerate(script, start=1):
        clients.chat.script[calls + n] = text
    case = CaseInput(case_id="c", title="Widget value lost", body="My widget loses its value on another page")
    return stage, retriever, stage(case, "q", hits), clients.chat.calls - calls


def test_one_corrective_turn_searches_again_and_reranks_old_and_new_together(clients, settings):
    stage, retriever, kept, calls = corrective(clients, settings, ["no"])
    assert calls == 4  # rerank, judge, rewrite, rerank
    assert retriever.query == "widget clean-up after page switch"
    assert kept[0].chunk.chunk_id == "doc:w:4" and kept[0].source == "corrective"
    assert stage.last_trace == {"judgements": ["no"], "queries": ["widget clean-up after page switch"], "turns": 1}


def test_a_yes_stops_at_once_and_zero_turns_does_nothing_extra(clients, settings):
    stage, _, _, calls = corrective(clients, settings, ["yes"])
    assert calls == 2 and stage.last_trace == {"judgements": ["yes"], "queries": [], "turns": 0}
    stage, _, _, calls = corrective(clients, settings, [], turns=0)
    assert calls == 1 and stage.last_trace is None  # only the reranking, as advanced


def test_retrieval_runs_record_the_corrective_decisions(tmp_path, capsys):
    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]
    assert main(["index", *base]) == 0
    assert main(["eval-retrieval", *base, "--pipeline", "advanced", "--set", "retrieval.corrective_turns=1",
                 "--run-id", "c"]) == 0
    capsys.readouterr()
    run = tmp_path / "art" / "runs" / "c"
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert all("judgements" in r["rerank_trace"] for r in rows)
    metrics = json.loads((run / "summary.json").read_text())["metrics"]
    assert "corrective_cases" in metrics and "judged_enough_first" in metrics


def coverage(clients, settings, judge_answer, max_missing=2):
    from supportdesk.pipelines.stages.evidence import CorrectiveReranker
    chunks = page_chunks()
    s = dataclasses.replace(settings, retrieval=dataclasses.replace(
        settings.retrieval, corrective_turns=1, corrective_judge="coverage", corrective_max_missing=max_missing,
        top_k=2, doc_slots=0, keep_report_query=True, max_queries=3))

    class Retriever:
        def __init__(self):
            self.queries = []

        def search(self, query, *, k, as_of=None):
            self.queries.append(query)
            return [Hit(chunks[4], 0.8, 1, "dense")]

    retriever = Retriever()
    stage = CorrectiveReranker(clients, s, chunks, retriever)
    hits = [Hit(chunks[0], 0.9, 1, "dense"), Hit(chunks[1], 0.5, 2, "dense")]
    calls = clients.chat.calls
    concepts = json.dumps({"queries": ["callbacks updating session state", "order of operations on rerun",
                                       "widget life cycle"]})
    script = [json.dumps({"ranking": ["C1", "C2"]}), concepts, judge_answer]  # rerank, concept queries, judge
    gaps = json.loads(judge_answer).get("missing", []) if judge_answer.startswith("{") else []
    script += [f"new query {n}" for n in range(min(len(gaps), max_missing))]
    if gaps:
        script.append(json.dumps({"ranking": ["C3", "C1", "C2"]}))  # the joint reranking
    for n, text in enumerate(script, start=1):
        clients.chat.script[calls + n] = text
    case = CaseInput(case_id="c", title="Callback runs late", body="My callback sees the old value")
    kept = stage(case, "q", hits)
    return stage, retriever, kept, [m[1]["content"] for m in clients.chat.seen[calls:]]


def test_the_coverage_judge_searches_once_per_missing_mechanism(clients, settings):
    stage, retriever, kept, seen = coverage(clients, settings, json.dumps({"missing": ["M2", "M3"]}))
    judge_input = next(c for c in seen if "## Mechanisms involved\nM1: " in c)
    assert "M2: order of operations on rerun" in judge_input and "## Passages" in judge_input
    rewrite_input = next(c for c in seen if "## Missing mechanism\norder of operations on rerun" in c)
    assert "callbacks updating session state" in rewrite_input  # the concept queries count as previous queries
    assert retriever.queries == ["new query 0", "new query 1"]
    assert stage.last_trace == {"judgements": [{"missing": ["order of operations on rerun", "widget life cycle"]}],
                                "queries": ["new query 0", "new query 1"], "turns": 1}
    assert kept[0].source == "corrective"


def test_the_coverage_judge_stops_when_nothing_is_missing_or_the_answer_is_unusable(clients, settings):
    stage, retriever, _, _ = coverage(clients, settings, json.dumps({"missing": []}))
    assert retriever.queries == [] and stage.last_trace["turns"] == 0
    stage, retriever, _, _ = coverage(clients, settings, "not json")
    assert retriever.queries == [] and stage.last_trace["judgements"] == ["unusable"]
    stage, retriever, _, _ = coverage(clients, settings, json.dumps({"missing": ["M1", "M2", "M3"]}), max_missing=1)
    assert retriever.queries == ["new query 0"]  # at most corrective_max_missing searches


def test_judged_enough_counts_both_kinds_of_judge():
    from supportdesk.eval.metrics import _enough
    assert _enough("yes") and not _enough("no")
    assert _enough({"missing": []}) and not _enough({"missing": ["x"]}) and not _enough("unusable")
