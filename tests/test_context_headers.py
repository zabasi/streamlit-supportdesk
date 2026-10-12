"""Contextual chunk headers: documentation chunks embedded with "page title › headings" in front."""
from supportdesk.data.context import context_header, embedding_texts, heading_path
from supportdesk.types import Chunk

PAGE = """Session State

Intro text about session state.

### Initialize values

Set defaults first.

```python
# Not a heading: inside a code fence
st.session_state.x = 1
```

### Caveats and limitations

Setting the state of button-like widgets is not allowed.
"""


def doc(start, end, title="Session State", kind="documentation"):
    return Chunk(chunk_id=f"c:{start}", source_id="doc:p", source_type=kind, title=title,
                 text=PAGE[start:end], url="u", start_char=start, end_char=end)


def test_the_heading_path_ignores_code_fences_and_follows_levels():
    assert heading_path(PAGE, PAGE.index("Set defaults")) == ["Initialize values"]
    assert heading_path(PAGE, PAGE.index("st.session_state.x")) == ["Initialize values"]  # "# Not a heading" skipped
    assert heading_path(PAGE, PAGE.index("Setting the state")) == ["Caveats and limitations"]
    assert heading_path(PAGE, 0) == []


def test_a_chunk_that_opens_with_a_heading_is_under_that_heading():
    caveats = PAGE.index("### Caveats")
    assert context_header(doc(caveats, len(PAGE)), PAGE) == "Session State › Caveats and limitations"
    middle = PAGE.index("Set defaults")
    assert context_header(doc(middle, caveats), PAGE) == "Session State › Initialize values"
    assert context_header(doc(0, 20), PAGE) == "Session State"


def test_only_documentation_gets_a_header_and_the_option_is_off_by_default():
    caveats = PAGE.index("### Caveats")
    chunks = [doc(caveats, len(PAGE)), doc(0, 20, kind="issue")]
    plain = embedding_texts(chunks, {"doc:p": PAGE}, enabled=False)
    assert plain == [c.text for c in chunks]
    headed = embedding_texts(chunks, {"doc:p": PAGE}, enabled=True)
    assert headed[0] == "Session State › Caveats and limitations\n\n" + chunks[0].text
    assert headed[1] == chunks[1].text  # issues and comments are embedded as before


def test_an_index_built_with_headers_is_refused_without_them_and_the_other_way_round(tmp_path, capsys):
    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl")]
    assert main(["index", *base, "--set", "chunking.context_header=true"]) == 0
    assert main(["eval-retrieval", *base, "--set", "chunking.context_header=true", "--run-id", "with"]) == 0
    capsys.readouterr()
    assert main(["eval-retrieval", *base, "--run-id", "without"]) == 1  # built with headers, asked without
    assert "context headers" in capsys.readouterr().err


def test_the_header_experiments_change_only_the_header_and_its_index():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    for prefix in ("clean_doc_slots_rerank_docs_prompt", "clean_doc_slots_rerank_docs_prompt_decompose"):
        check_header_variants(flat, prefix)


def check_header_variants(flat, prefix):
    for parent, child in (("", "_ctx"), ("_structure", "_structure_ctx"), ("_token_structure", "_token_structure_ctx"),
                          ("_structure_top8_keep4", "_structure_ctx_top8_keep4"),
                          ("_token_structure_top8_keep4", "_token_structure_ctx_top8_keep4")):
        base, other = flat(prefix + parent), flat(prefix + child)
        assert {k for k in base if base[k] != other[k]} == {"pipeline.name", "chunking.context_header", "paths.index_dir"}
        assert other["chunking.context_header"] is True and str(other["paths.index_dir"]).endswith("_ctx")


def test_both_designs_share_the_header_indexes():
    from supportdesk.config import Settings
    def index_of(name):
        return Settings.load(env={}, overrides=[f"pipeline.name={name}"]).paths.index_dir
    a, d = "clean_doc_slots_rerank_docs_prompt", "clean_doc_slots_rerank_docs_prompt_decompose"
    for suffix in ("_ctx", "_structure_ctx", "_token_structure_ctx", "_structure_ctx_top8_keep4",
                   "_token_structure_ctx_top8_keep4"):
        assert index_of(a + suffix) == index_of(d + suffix)


def test_the_all_scope_names_every_section_inside_the_chunk():
    from supportdesk.data.context import headings_inside
    start = PAGE.index("Set defaults")
    chunk = doc(start, len(PAGE))  # from inside "Initialize values" to the end, across "Caveats"
    assert headings_inside(chunk) == ["Caveats and limitations"]  # "# Not a heading" is inside a fence
    assert context_header(chunk, PAGE, "start") == "Session State › Initialize values"
    assert context_header(chunk, PAGE, "all") == "Session State › Initialize values; Caveats and limitations"
    caveats = PAGE.index("### Caveats")
    assert context_header(doc(caveats, len(PAGE)), PAGE, "all") == "Session State › Caveats and limitations"


def test_the_structure_chunkers_follow_their_size_settings():
    from supportdesk.config import Settings
    from supportdesk.data.chunking import build_chunks, make_chunker
    from supportdesk.data.corpus import load_corpus
    from tests.helpers import ROOT
    records = [r for r in load_corpus(ROOT / "data" / "processed").records if r.source_type == "documentation"]
    def chunks(*sets):
        return build_chunks(records, chunker=make_chunker(Settings.load(env={}, overrides=list(sets))))
    small, large = chunks("chunking.method=structure"), chunks(
        "chunking.method=structure", "chunking.structure_min_chars=2000", "chunking.structure_max_chars=4000")
    assert len(large) < len(small)
    t_small, t_large = chunks("chunking.method=token_structure"), chunks(
        "chunking.method=token_structure", "chunking.token_target=600", "chunking.token_soft_max=900",
        "chunking.token_min_section=300")
    assert len(t_large) < len(t_small)


def test_an_index_is_refused_when_its_chunk_sizes_differ_and_old_indexes_read_as_defaults(tmp_path, capsys):
    import json

    from supportdesk.cli import main
    from tests.helpers import ROOT, SAMPLE
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"), "--artifacts-dir",
            str(tmp_path / "art"), "--mode", "offline", "--cases", str(ROOT / "data" / "eval" / "sample_cases.jsonl"),
            "--set", "chunking.method=structure"]
    assert main(["index", *base]) == 0
    large = ["--set", "chunking.structure_min_chars=2000", "--set", "chunking.structure_max_chars=4000"]
    capsys.readouterr()
    assert main(["eval-retrieval", *base, *large, "--run-id", "x"]) == 1
    assert "chunking settings" in capsys.readouterr().err
    manifest_path = tmp_path / "idx" / "manifest.json"  # an index from before the size settings existed
    manifest = json.loads(manifest_path.read_text())
    for key in ("structure_min_chars", "structure_max_chars"):
        manifest["chunking"].pop(key)
    manifest_path.write_text(json.dumps(manifest))
    assert main(["eval-retrieval", *base, "--run-id", "y"]) == 0  # read as built with the defaults


def test_the_large_chunk_experiments_change_only_sizes_headers_and_index():
    from supportdesk.config import Settings
    def flat(name):
        d = Settings.load(env={}, overrides=[f"pipeline.name={name}"]).to_dict()
        return {f"{section}.{key}": value for section, keys in d.items() for key, value in keys.items()}
    d = "clean_doc_slots_rerank_docs_prompt_decompose"
    sizes = {"structure": {"chunking.structure_min_chars", "chunking.structure_max_chars"},
             "token_structure": {"chunking.token_target", "chunking.token_soft_max", "chunking.token_min_section"}}
    for method, size_keys in sizes.items():
        base = flat(f"{d}_{method}")
        for suffix, header_keys in (("", set()), ("_ctx", {"chunking.context_header"}),
                                    ("_ctx_all", {"chunking.context_header", "chunking.context_header_scope"})):
            other = flat(f"{d}_{method}_large{suffix}")
            assert {k for k in base if base[k] != other[k]} == {"pipeline.name", "paths.index_dir", *size_keys,
                                                                *header_keys}, method + suffix


def test_the_new_chunking_settings_are_validated():
    import pytest

    from supportdesk.config import ConfigError, Settings
    for bad in (["chunking.context_header_scope=middle"],
                ["chunking.structure_min_chars=3000", "chunking.structure_max_chars=2000"],
                ["chunking.token_target=800", "chunking.token_soft_max=700"]):
        with pytest.raises(ConfigError):
            Settings.load(env={}, overrides=bad)
