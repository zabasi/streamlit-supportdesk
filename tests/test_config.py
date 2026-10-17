"""Configuration: layering, validation, provenance, and (most importantly) that every setting is
wired to behaviour, so a value that parses but changes nothing cannot hide."""
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from supportdesk.cli import main
from supportdesk.config import SECTIONS, ConfigError, Settings
from supportdesk.data.chunking import FixedTokenChunker, build_chunks
from supportdesk.eval.cases import exclusions_from_cases
from supportdesk.eval.report import render_report
from supportdesk.eval.runner import run_eval
from supportdesk.indexing import run_indexing
from supportdesk.llm.cache import DiskCache
from supportdesk.llm.errors import InputTooLarge
from supportdesk.llm.factory import build_clients
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.registry import make_pipeline, make_retriever
from supportdesk.retrieval.index import IndexUnusable, VectorIndex, build_index
from supportdesk.tokenizer import count_tokens
from supportdesk.types import CaseInput
from tests.helpers import ROOT, SAMPLE, make_settings, override
from tests.test_budget_and_gateway import PRICING, StubTransport, make_gateway

DEFAULT_TOML = ROOT / "configs" / "default.toml"
CASE = CaseInput(case_id="t", title="Session state is empty after browser refresh",
                 body="I store the user in st.session_state. After F5 the value is gone. "
                      "Streamlit 1.31.1, Chrome.", created_at="2024-04-20T10:00:00Z")


def build_world(tmp_path, cases, **sections):
    """settings, clients, index, pipeline for the sample data, all driven by ``sections``."""
    settings = make_settings(tmp_path, **sections)
    clients = build_clients(settings)
    index = run_indexing(settings, clients, cases=cases, exclusions=exclusions_from_cases(cases))["index"]
    pipeline = make_pipeline(settings.pipeline.name, settings.pipeline.retriever, index, clients,
                             settings)
    return settings, clients, index, pipeline


def _records():
    from supportdesk.data.corpus import load_corpus
    return load_corpus(SAMPLE).records


def _chunker(settings):
    return FixedTokenChunker(max_tokens=settings.chunking.max_tokens,
                             overlap_tokens=settings.chunking.overlap_tokens)


def test_default_toml_is_exactly_the_built_in_defaults():
    from_file = Settings.load(config_files=[DEFAULT_TOML], env={})
    assert from_file.to_dict() == Settings.from_dict({}).to_dict()


def test_every_setting_is_documented_in_default_toml():
    text = DEFAULT_TOML.read_text()
    missing = [f"{s}.{f.name}" for s, cls in SECTIONS.items() for f in fields(cls)
               if not re.search(rf"^\s*#?\s*{f.name}\s*=", text, re.MULTILINE)]
    assert not missing, f"add these to configs/default.toml: {missing}"


def test_precedence_is_default_then_file_then_env_then_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no configs/default.toml here, so 'default' means built-in
    cfg = tmp_path / "exp.toml"
    cfg.write_text("[retrieval]\ntop_k = 7\nrrf_k = 33\n[models]\nchat = \"from-file\"\n")
    env = {"SUPPORTDESK__RETRIEVAL__TOP_K": "9", "CHAT_MODEL": "from-alias"}
    s = Settings.load(config_files=[cfg], env=env, overrides=["retrieval.top_k=11"])
    assert s.retrieval.top_k == 11 and s.sources["retrieval.top_k"] == "cli"
    assert s.retrieval.rrf_k == 33 and s.sources["retrieval.rrf_k"] == f"file:{cfg}"
    assert s.models.chat == "from-alias" and s.sources["models.chat"] == "env:CHAT_MODEL"
    assert s.retrieval.bm25_k1 == 1.5 and s.sources["retrieval.bm25_k1"] == "default"
    no_cli = Settings.load(config_files=[cfg], env=env)
    assert no_cli.retrieval.top_k == 9 and no_cli.sources["retrieval.top_k"].startswith("env:")


def test_later_config_files_override_earlier_ones(tmp_path):
    a, b = tmp_path / "a.toml", tmp_path / "b.toml"
    a.write_text("[retrieval]\ntop_k = 4\nrrf_k = 10\n")
    b.write_text("[retrieval]\ntop_k = 5\n")
    s = Settings.load(config_files=[a, b], env={})
    assert (s.retrieval.top_k, s.retrieval.rrf_k) == (5, 10)


def test_legacy_env_names_still_work_and_the_generic_form_wins(tmp_path):
    s = Settings.load(env={"SUPPORT_MODE": "replay", "SUPPORT_INDEX_DIR": "/tmp/x",
                           "BUDGET_CEILING_USD": "2.5"})
    assert (s.mode, s.paths.index_dir, s.budget.ceiling_usd) == ("replay", Path("/tmp/x"), 2.5)
    both = Settings.load(env={"SUPPORT_MODE": "replay", "SUPPORTDESK__RUN__MODE": "offline"})
    assert both.mode == "offline"


def test_the_default_file_is_always_the_base_layer(tmp_path, monkeypatch):
    """Editing configs/default.toml must affect every run, including ones that add --config files."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.toml").write_text(
        '[retrieval]\ntop_k = 2\nrrf_k = 11\n[models]\nchat = "custom-model"\n')
    exp = tmp_path / "exp.toml"
    exp.write_text("[retrieval]\ntop_k = 3\n")
    env_file = tmp_path / "env.toml"
    env_file.write_text("[retrieval]\ntop_k = 4\n")
    assert Settings.load(env={}).retrieval.top_k == 2
    s = Settings.load(env={}, config_files=[exp])
    assert (s.retrieval.top_k, s.retrieval.rrf_k, s.models.chat) == (3, 11, "custom-model")
    assert s.config_files == ("configs/default.toml", str(exp))
    from_env = Settings.load(env={"SUPPORTDESK_CONFIG": str(env_file)})
    assert from_env.retrieval.top_k == 4 and from_env.retrieval.rrf_k == 11
    assert Settings.load(env={"SUPPORTDESK_CONFIG": str(env_file)}, config_files=[exp]).retrieval.top_k == 3


def test_relative_paths_resolve_against_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings.load(env={})
    assert s.paths.index_dir.resolve() == (tmp_path / "index").resolve()


def test_optional_values_can_be_unset_from_text(tmp_path):
    s = Settings.load(env={}, overrides=["retrieval.candidate_k=12"])
    assert s.pool_size == 12
    cleared = override(s, retrieval={"candidate_k": None})
    assert cleared.retrieval.candidate_k is None and cleared.pool_size == cleared.retrieval.top_k
    assert Settings.load(env={}, overrides=["retrieval.candidate_k=none"]).retrieval.candidate_k is None


def test_raising_top_k_alone_raises_what_the_prompt_may_hold():
    assert Settings.load(env={}).prompt_chunks == 6
    assert Settings.load(env={}, overrides=["retrieval.top_k=9"]).prompt_chunks == 9
    explicit = Settings.load(env={}, overrides=["retrieval.top_k=9", "context.max_chunks=4"])
    assert explicit.prompt_chunks == 4


@pytest.mark.parametrize("override, fragment", [
    ("retrival.top_k=3", "unknown section 'retrival'. Did you mean 'retrieval'"),
    ("retrieval.top_K=3", "unknown key. Did you mean 'top_k'"),
    ("retrieval.top_k=abc", "cannot read 'abc' as int"),
    ("retrieval.top_k=2.5", "expected int"),
    ("retrieval.historical_filter=1", "expected bool"),
    ("retrieval.top_k=0", "must be >= 1"),
    ("context.max_chunks=9", "could never be filled"),
    ("retrieval.candidate_k=3", "must be >= retrieval.top_k"),
    ("chunking.overlap_tokens=600", "must be >= 0 and < chunking.max_tokens"),
    ("chunking.method=llm", "chunking.method 'llm' is unknown"),
    ("generation.temperature=5", "between 0 and 2"),
    ("budget.reserve_usd=9", "must be >= 0 and < budget.ceiling_usd"),
    ("run.mode=turbo", "run.mode must be one of"),
    ("eval.recall_ks=[]", "non-empty list"),
    ("retrieval.bm25_b=3", "between 0 and 1"),
    ("web.port=70000", "web.port"),
    ("novalue", "section.key=value"),
])
def test_bad_configuration_fails_loudly_and_names_the_key(override, fragment):
    with pytest.raises(ConfigError) as info:
        Settings.load(env={}, overrides=[override])
    assert fragment in str(info.value)


def test_all_problems_are_reported_together():
    with pytest.raises(ConfigError) as info:
        Settings.from_dict({"retrieval": {"top_k": 0}, "chunking": {"max_tokens": 10, "overlap_tokens": 10}})
    assert "retrieval.top_k" in str(info.value) and "chunking.overlap_tokens" in str(info.value)


def test_config_files_are_checked_for_secrets_typos_and_syntax(tmp_path):
    secret = tmp_path / "s.toml"
    secret.write_text('[models]\nchat = "sk-abcdefghijklmnop123456"\n')
    with pytest.raises(ConfigError, match="API key"):
        Settings.load(config_files=[secret], env={})
    typo = tmp_path / "t.toml"
    typo.write_text("[retreival]\ntop_k = 3\n")
    with pytest.raises(ConfigError, match="Did you mean 'retrieval'"):
        Settings.load(config_files=[typo], env={})
    broken = tmp_path / "b.toml"
    broken.write_text("[retrieval\ntop_k = ")
    with pytest.raises(ConfigError, match="not valid TOML"):
        Settings.load(config_files=[broken], env={})
    with pytest.raises(ConfigError, match="not found"):
        Settings.load(config_files=[tmp_path / "nope.toml"], env={})
    with pytest.raises(ConfigError, match="SECTION__KEY"):
        Settings.load(env={"SUPPORTDESK__ONLYONE": "1"})


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs" / "experiments").glob("*.toml")) + [ROOT / "configs" / "sample.toml"],
    ids=lambda p: p.name)
def test_shipped_experiment_files_are_valid_layers_over_the_defaults(path):
    s = Settings.load(config_files=[DEFAULT_TOML, path], env={})
    assert s.fingerprint() != Settings.load(config_files=[DEFAULT_TOML], env={}).fingerprint()


def test_resolved_config_round_trips_and_has_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretkey1234567890")
    s = Settings.load(env={}, overrides=["retrieval.top_k=8"])
    again = Settings.from_dict(s.to_dict())
    assert again == s and again.fingerprint() == s.fingerprint()
    blob = json.dumps(s.public())
    assert "sk-" not in blob and s.public()["api_key_present"] is True
    assert s.public()["config_sources"]["retrieval.top_k"] == "cli"


def test_top_k_controls_both_retrieval_and_the_prompt(tmp_path, cases):
    for k in (3, 8):
        _, clients, _, pipe = build_world(tmp_path / f"k{k}", cases,
                                          retrieval={"top_k": k, "hybrid_pool": 30},
                                          context={"max_tokens": 10_000})
        result = pipe.run(CASE)
        assert len(result.retrieved) == k and len(result.trace["packet"]) == k
        assert clients.chat.seen[-1][-1]["content"].count("<evidence") == k


def test_context_limits_cap_the_prompt_independently(tmp_path, cases):
    _, _, _, few = build_world(tmp_path / "a", cases, retrieval={"top_k": 6}, context={"max_chunks": 2})
    r = few.run(CASE)
    assert len(r.retrieved) == 6 and len(r.trace["packet"]) == 2
    _, _, _, tight = build_world(tmp_path / "b", cases, context={"max_tokens": 150})
    assert len(tight.run(CASE).trace["packet"]) < 6


def test_query_and_body_length_limits_are_applied(tmp_path, cases):
    _, _, _, long_q = build_world(tmp_path / "a", cases, context={"max_query_chars": 4000})
    _, _, _, short_q = build_world(tmp_path / "b", cases, context={"max_query_chars": 20})
    assert long_q.run(CASE).trace["query_sha256"] != short_q.run(CASE).trace["query_sha256"]
    _, clients, _, pipe = build_world(tmp_path / "c", cases, context={"max_body_chars": 10})
    pipe.run(CASE)
    prompt = clients.chat.seen[-1][-1]["content"]
    report = prompt.split("## Conversation")[0]
    assert "I store th\n" in report and "I store the " not in report  # cut at exactly 10 characters


def test_max_retries_controls_how_many_times_the_model_is_asked(tmp_path, cases):
    for retries, failures, expected_calls, outcome in ((0, 1, 1, "error"), (3, 3, 4, "answer")):
        _, clients, _, pipe = build_world(tmp_path / f"r{retries}", cases,
                                          generation={"max_retries": retries})
        clients.chat.script = {n: "bad_json" for n in range(1, failures + 1)}
        result = pipe.run(CASE)
        assert clients.chat.calls == expected_calls and result.outcome == outcome


def test_chunking_settings_change_the_chunks_and_are_recorded_in_the_index(tmp_path, cases):
    s_small = make_settings(tmp_path / "s", chunking={"max_tokens": 20, "overlap_tokens": 5})
    s_default = make_settings(tmp_path / "d")
    exclusions = exclusions_from_cases(cases)
    small = build_chunks(_records(), exclusions, chunker=_chunker(s_small))
    default = build_chunks(_records(), exclusions, chunker=_chunker(s_default))
    assert len(small) > len(default) and max(count_tokens(c.text) for c in small) <= 20
    clients = build_clients(s_small)
    idx = build_index(small, clients.embedding, embedding_model="fake-embedding",
                      chunking=s_small.chunking)
    assert idx.manifest["chunking"] == {"method": "fixed", "max_tokens": 20, "overlap_tokens": 5, "context_header": False}


def test_an_index_built_with_other_chunking_is_refused(tmp_path, cases):
    s = make_settings(tmp_path)
    _, _, index, _ = build_world(tmp_path, cases)
    index.save(s.paths.index_dir)
    VectorIndex.load(s.paths.index_dir, expect_chunking=s.chunking)  # same settings: fine
    other = make_settings(tmp_path, chunking={"max_tokens": 100, "overlap_tokens": 10})
    with pytest.raises(IndexUnusable, match="rebuild"):
        VectorIndex.load(s.paths.index_dir, expect_chunking=other.chunking)


def test_gateway_uses_batch_size_temperature_output_cap_and_input_limit(tmp_path):
    stub = StubTransport()
    seen = {}
    real_chat = stub.chat

    def spy(**kw):
        seen.update(kw)
        return real_chat(**kw)
    stub.chat = spy
    from supportdesk.llm.gateway import Gateway
    settings = Settings.from_dict({
        "run": {"mode": "live"}, "models": {"chat": "chat-x", "embedding": "emb-x"},
        "budget": {"ceiling_usd": 1.0, "reserve_usd": 0.0},
        "generation": {"temperature": 0.7, "max_output_tokens": 123},
        "retrieval": {"embed_batch_size": 2}, "limits": {"max_input_chars": 1000},
        "paths": {"cache_path": str(tmp_path / "c.sqlite")}})
    from supportdesk.llm.accounting import BudgetLedger
    gw = Gateway(mode="live", settings=settings, ledger=BudgetLedger(1.0), pricing=PRICING,
                 cache=DiskCache(tmp_path / "c.sqlite"), transport=stub)
    gw.complete([{"role": "user", "content": "hi"}])
    assert seen["temperature"] == 0.7 and seen["max_output_tokens"] == 123
    stub.calls = 0
    gw.embed(["a", "b", "c", "d", "e"])
    assert stub.calls == 3  # ceil(5 / 2)
    with pytest.raises(InputTooLarge, match="limit is 1000"):
        gw.complete([{"role": "user", "content": "y" * 1500}])


def test_changing_temperature_or_output_cap_does_not_reuse_a_stale_cached_answer(tmp_path):
    gw1, _ = make_gateway(tmp_path)
    gw1.complete([{"role": "user", "content": "same prompt"}])
    cache = gw1.cache
    s2 = Settings.from_dict({"run": {"mode": "live"}, "models": {"chat": "chat-x", "embedding": "emb-x"},
                             "budget": {"ceiling_usd": 1.0, "reserve_usd": 0.0},
                             "generation": {"temperature": 0.9}})
    from supportdesk.llm.accounting import BudgetLedger
    from supportdesk.llm.gateway import Gateway
    gw2 = Gateway(mode="live", settings=s2, ledger=BudgetLedger(1.0), pricing=PRICING, cache=cache,
                  transport=StubTransport())
    _, usage = gw2.complete([{"role": "user", "content": "same prompt"}])
    assert usage["mode"] == "live"  # a different temperature is a different request


def test_retriever_hyperparameters_come_from_settings(tmp_path, cases):
    s1, c1, idx, _ = build_world(tmp_path / "a", cases, retrieval={"bm25_k1": 1.5, "rrf_k": 60})
    s2 = override(s1, retrieval={"bm25_k1": 4.0, "rrf_k": 5})
    q = "WebSocket Upgrade Connection headers nginx"
    a = make_retriever("bm25", idx, c1, s1).search(q, k=3)
    b = make_retriever("bm25", idx, c1, s2).search(q, k=3)
    assert [h.score for h in a] != [h.score for h in b]
    ha = make_retriever("hybrid", idx, c1, s1).search(q, k=3)
    hb = make_retriever("hybrid", idx, c1, s2).search(q, k=3)
    assert ha[0].score != hb[0].score  # RRF constant changed the fused scores
    assert make_retriever("hybrid", idx, c1, s2).pool == s2.retrieval.hybrid_pool


def test_candidate_k_setting_sizes_the_rerank_pool(tmp_path, cases):
    seen = {}

    class Spy:
        def __call__(self, case, query, hits):
            seen["n"] = len(hits)
            return hits
    _, _, _, pipe = build_world(tmp_path, cases, retrieval={"candidate_k": 9})
    pipe.variant(reranker=Spy()).run(CASE)
    assert seen["n"] == 9


def test_eval_recall_ks_and_the_report_follow_the_configuration(tmp_path, cases):
    settings, clients, _, pipe = build_world(tmp_path, cases, eval={"recall_ks": [2, 4]})
    summary = run_eval(pipe, cases[:2], clients, settings, run_id="r", out_root=tmp_path / "runs")
    assert set(summary["metrics"]["recall_at_k"]) == {"2", "4"}
    html = render_report([tmp_path / "runs" / "r"])
    assert "Recall@2" in html and "Recall@4" in html and "Recall@6" not in html


def test_runs_record_the_exact_configuration_that_produced_them(tmp_path, cases):
    shas = []
    for k in (4, 5):
        settings, clients, _, pipe = build_world(tmp_path / str(k), cases, retrieval={"top_k": k})
        s = run_eval(pipe, cases[:1], clients, settings, run_id="r", out_root=tmp_path / str(k))
        assert s["settings"]["config"]["retrieval"]["top_k"] == k
        shas.append(s["settings"]["config_sha256"])
    assert shas[0] != shas[1]


def test_web_limits_and_port_come_from_settings(tmp_path, cases):
    import http.client
    import threading

    from supportdesk.web.server import App, create_server
    settings, clients, index, _ = build_world(
        tmp_path, cases, web={"max_title_chars": 10, "port": 0, "max_request_bytes": 1000})
    app = App(settings, clients=clients, index=index)
    assert app.status()["limits"]["title"] == 10
    httpd = create_server(app)  # no explicit port: it comes from web.port
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True).start()
    try:
        port = httpd.server_address[1]
        assert port != 8765
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/ask", body=json.dumps({"title": "x" * 11}),
                     headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        assert resp.status == 400 and "longer than 10" in json.loads(resp.read())["error"]
        big = json.dumps({"title": "t", "body": "x" * 2000})  # far below the default 64 000 limit
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/ask", body=big,
                     headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        assert resp.status == 413 and "1000 bytes" in json.loads(resp.read())["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _cli(tmp_path, *args):
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline"]
    return main([*args[:1], *base, *args[1:]])


def test_cli_config_show_prints_values_sources_and_absolute_paths(tmp_path, capsys):
    assert _cli(tmp_path, "config", "--set", "retrieval.top_k=8") == 0
    out = capsys.readouterr().out
    assert re.search(r"top_k\s+= 8\s+\(cli\)", out) and str(tmp_path / "idx") in out
    assert "chunks allowed in the prompt = 8" in out
    assert _cli(tmp_path, "config", "--json", "--set", "retrieval.top_k=8") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["retrieval"]["top_k"] == 8 and data["sources"]["retrieval.top_k"] == "cli"


def test_cli_set_errors_exit_cleanly(tmp_path, capsys):
    assert _cli(tmp_path, "config", "--set", "retrieval.top_K=8") == 1
    assert "Did you mean 'top_k'" in capsys.readouterr().err


def test_cli_index_reports_where_it_wrote_and_status_agrees(tmp_path, capsys):
    cases = str(ROOT / "data" / "eval" / "sample_cases.jsonl")
    assert _cli(tmp_path, "status") == 0
    assert json.loads(capsys.readouterr().out)["index_exists"] is False
    assert _cli(tmp_path, "index", "--cases", cases) == 0
    out = capsys.readouterr().out
    assert f"index written to {tmp_path / 'idx'}" in out
    for name in ("qdrant", "chunks.jsonl", "manifest.json"):
        assert (tmp_path / "idx" / name).exists()
    assert _cli(tmp_path, "status") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["index_exists"] is True and status["paths"]["index_dir"] == str(tmp_path / "idx")
    manifest = json.loads((tmp_path / "idx" / "manifest.json").read_text())
    assert manifest["chunking"] == {"method": "fixed", "max_tokens": 512, "overlap_tokens": 128,
                                    "context_header": False}
    assert manifest["sources_without_chunks"] == [] and len(manifest["config_sha256"]) == 16


def test_cli_refuses_to_run_against_an_index_built_with_other_chunking(tmp_path, capsys):
    cases = str(ROOT / "data" / "eval" / "sample_cases.jsonl")
    assert _cli(tmp_path, "index", "--cases", cases) == 0
    capsys.readouterr()
    code = _cli(tmp_path, "ask", "--title", "t", "--body", "b", "--set", "chunking.max_tokens=100",
                "--set", "chunking.overlap_tokens=10")
    assert code == 1 and "rebuild it" in capsys.readouterr().err


def test_cli_picks_pipeline_and_retriever_from_the_configuration(tmp_path, capsys):
    cases = str(ROOT / "data" / "eval" / "sample_cases.jsonl")
    assert _cli(tmp_path, "index", "--cases", cases) == 0
    exp = tmp_path / "exp.toml"
    exp.write_text('[pipeline]\nretriever = "bm25"\n')
    capsys.readouterr()
    assert _cli(tmp_path, "ask", "--config", str(exp), "--title", "session state refresh",
                "--body", "st.session_state lost") == 0
    assert json.loads(capsys.readouterr().out)["trace"]["stages"]["retriever"] == "BM25Retriever"


def test_sources_that_produce_no_chunks_are_reported_not_silently_dropped(cases):
    from supportdesk.data.chunking import sources_without_chunks
    from supportdesk.data.corpus import SourceRecord
    records = [*_records(), SourceRecord(source_id="doc-symbols", source_type="documentation", title="",
                                         text="!!! ??? ...", url="u")]
    chunks = build_chunks(records, exclusions_from_cases(cases), chunker=FixedTokenChunker(max_tokens=512, overlap_tokens=64))
    assert sources_without_chunks(records, chunks, exclusions_from_cases(cases)) == ["doc-symbols"]


def test_baseline_class_still_builds_from_settings(tmp_path, cases):
    settings, clients, index, _ = build_world(tmp_path, cases)
    from supportdesk.retrieval.retriever import DenseRetriever
    assert build_baseline(DenseRetriever(index, clients.embedding), clients, settings).name == "baseline"


def test_sample_config_runs_the_whole_flow_without_flags(tmp_path, monkeypatch, capsys):
    """configs/sample.toml is what a newcomer runs first: index + ask with no other options."""
    monkeypatch.chdir(ROOT)
    out = tmp_path / "out"
    base = ["--config", "configs/sample.toml", "--set", f"paths.index_dir={out / 'idx'}",
            "--set", f"paths.artifacts_dir={out / 'art'}"]
    assert main(["index", *base, "--cases", "data/eval/sample_cases.jsonl"]) == 0
    capsys.readouterr()
    assert main(["ask", *base, "--title", "Session state empty after refresh",
                 "--body", "st.session_state value is gone after F5"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "answer"


def test_report_lists_exactly_the_settings_that_differ_between_runs(tmp_path, cases):
    for name, k in (("a", 4), ("b", 7)):
        settings, clients, _, pipe = build_world(tmp_path / name, cases, retrieval={"top_k": k})
        run_eval(pipe, cases[:1], clients, settings, run_id=name, out_root=tmp_path / "runs")
    html = render_report([tmp_path / "runs" / "a", tmp_path / "runs" / "b"])
    section = html.split("Settings that differ")[1].split("<h2>Metrics")[0]
    assert "retrieval.top_k" in section and "<td>4</td><td>7</td>" in section
    assert "retrieval.rrf_k" not in section and "paths." not in section  # identical or irrelevant
    single = render_report([tmp_path / "runs" / "a"])
    assert "Settings that differ" not in single


def test_every_setting_is_documented_in_the_configuration_guide():
    text = (ROOT / "docs" / "configuration.md").read_text()
    missing = [f"{s}.{f.name}" for s, cls in SECTIONS.items() for f in fields(cls)
               if f"| `{f.name}` |" not in text or f"### `[{s}]`" not in text]
    assert not missing, f"run `python scripts/gen_config_docs.py`: undocumented settings {missing}"


def test_generated_reference_is_up_to_date():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_config_docs", ROOT / "scripts" / "gen_config_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.reference() in (ROOT / "docs" / "configuration.md").read_text()


def _doc_commands():
    import shlex
    commands = []
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        in_block, pending = False, ""
        for line in doc.read_text().splitlines():
            if line.strip().startswith("```"):
                in_block = line.strip() in ("```bash", "```sh") if not in_block else False
                pending = ""
                continue
            if not in_block:
                continue
            text = pending + line.strip()
            pending = ""
            if text.endswith("\\"):
                pending = text[:-1].strip() + " "
                continue
            text = re.sub(r"\s+#.*$", "", text)  # drop trailing comments
            if text.startswith("#") or not text:
                continue
            tokens = shlex.split(text)
            while tokens and re.fullmatch(r"[A-Z_][A-Z0-9_]*=\S*", tokens[0]):  # VAR=x supportdesk ...
                tokens = tokens[1:]
            if tokens and tokens[0] == "supportdesk":
                commands.append((doc.name, text, tokens[1:]))
    return commands


DOC_COMMANDS = _doc_commands()


def test_the_docs_contain_commands_to_check():
    assert len(DOC_COMMANDS) >= 15


@pytest.mark.parametrize("doc, text, args", DOC_COMMANDS, ids=lambda v: v if isinstance(v, str) and len(v) < 50 else "")
def test_every_documented_supportdesk_command_is_valid(doc, text, args, capsys):
    from supportdesk.cli import build_parser
    try:
        parsed = build_parser().parse_args(args)
    except SystemExit:
        pytest.fail(f"{doc}: not accepted by the CLI: supportdesk {' '.join(args)}\n"
                    f"{capsys.readouterr().err.strip().splitlines()[-1]}")
    # a documented --set / --config must also be a *valid* setting, not just well-formed
    overrides = [v for v in (getattr(parsed, "set", None) or []) if "..." not in v]
    Settings.load(env={}, overrides=overrides)


def _profile_tree(tmp_path, monkeypatch):
    """default.toml says top_k=20; baseline pins 6; advanced uses 10 (+ pool 30)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "pipelines").mkdir(parents=True)
    (tmp_path / "configs" / "default.toml").write_text("[retrieval]\ntop_k = 20\nrrf_k = 11\n")
    (tmp_path / "configs" / "pipelines" / "baseline.toml").write_text("[retrieval]\ntop_k = 6\n")
    (tmp_path / "configs" / "pipelines" / "advanced.toml").write_text(
        '[retrieval]\ntop_k = 10\ncandidate_k = 30\n[pipeline]\nretriever = "hybrid"\n')


def test_each_pipeline_gets_its_own_settings_and_the_baseline_stays_pinned(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    base = Settings.load(env={})
    assert base.pipeline.name == "baseline" and base.retrieval.top_k == 6  # not default.toml's 20
    assert base.sources["retrieval.top_k"] == "file:configs/pipelines/baseline.toml"
    assert base.retrieval.rrf_k == 11  # still inherits the shared defaults
    assert base.profile == "configs/pipelines/baseline.toml"
    adv = Settings.load(env={}, overrides=["pipeline.name=advanced"])
    assert (adv.retrieval.top_k, adv.pool_size, adv.pipeline.retriever) == (10, 30, "hybrid")
    assert adv.prompt_chunks == 10  # the prompt follows the pipeline's own top_k
    assert Settings.load(env={}, overrides=["pipeline.name=other"]).retrieval.top_k == 20  # no profile


def test_profile_selection_follows_every_way_of_choosing_the_pipeline(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    chooser = tmp_path / "choose.toml"
    chooser.write_text('[pipeline]\nname = "advanced"\n')
    assert Settings.load(env={"SUPPORTDESK__PIPELINE__NAME": "advanced"}).retrieval.top_k == 10
    assert Settings.load(env={}, config_files=[chooser]).retrieval.top_k == 10
    assert Settings.load(env={}, config_files=[chooser]).profile == "configs/pipelines/advanced.toml"


def test_experiments_and_overrides_beat_the_profile(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    exp = tmp_path / "exp.toml"
    exp.write_text("[retrieval]\ntop_k = 3\n")
    s = Settings.load(env={}, config_files=[exp], overrides=["pipeline.name=advanced"])
    assert s.retrieval.top_k == 3 and s.retrieval.candidate_k == 30  # exp wins on top_k only
    env = Settings.load(env={"SUPPORTDESK__RETRIEVAL__TOP_K": "4"}, config_files=[exp])
    assert env.retrieval.top_k == 4
    cli = Settings.load(env={}, config_files=[exp], overrides=["retrieval.top_k=2"])
    assert cli.retrieval.top_k == 2 and cli.sources["retrieval.top_k"] == "cli"


def test_a_profile_may_only_set_what_defines_a_pipeline(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    profile = tmp_path / "configs" / "pipelines" / "baseline.toml"
    for body, fragment in (('[models]\nchat = "x"\n', "found [models]"),
                           ('[generation]\ntemperature = 0.5\n', "found [generation]"),
                           ('[pipeline]\nname = "advanced"\n', "pipeline.['generator', 'retriever', 'scope_gate', 'version_notes']; found pipeline.name"),
                           ('[paths]\ndata_dir = "x"\n', "paths.['index_dir']; found paths.data_dir")):
        profile.write_text(body)
        with pytest.raises(ConfigError, match="may only set") as info:
            Settings.load(env={})
        assert fragment in str(info.value)
    profile.write_text("[retrieval]\ntop_kk = 3\n")
    with pytest.raises(ConfigError, match="Did you mean 'top_k'"):
        Settings.load(env={})


def test_pipeline_names_cannot_escape_the_profile_directory(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    (tmp_path / "evil.toml").write_text("[retrieval]\ntop_k = 99\n")
    s = Settings.load(env={}, overrides=["pipeline.name=../../evil"])
    assert s.retrieval.top_k == 20 and s.profile is None


def test_the_profile_is_recorded_for_status_and_run_summaries(tmp_path, monkeypatch):
    _profile_tree(tmp_path, monkeypatch)
    pub = Settings.load(env={}, overrides=["pipeline.name=advanced"]).public()
    assert pub["config_profile"] == "configs/pipelines/advanced.toml"
    assert "configs/pipelines/advanced.toml" in pub["config_files"]


def test_shipped_baseline_profile_equals_the_golden_tested_defaults(monkeypatch):
    monkeypatch.chdir(ROOT)
    s = Settings.load(env={})
    assert s.profile == "configs/pipelines/baseline.toml"
    assert s.retrieval.top_k == 6 and s.to_dict() == Settings.from_dict({}).to_dict()


def test_shipped_advanced_template_is_valid_and_differs_from_the_baseline(monkeypatch):
    monkeypatch.chdir(ROOT)
    adv = Settings.load(env={}, overrides=["pipeline.name=advanced"])
    assert adv.profile == "configs/pipelines/advanced.toml"
    base = Settings.load(env={})
    # advanced shows more evidence than the baseline (8 pieces within 4,300 tokens against 6 within 3,200),
    # so a fair comparison runs the baseline at advanced's budget too; the profile says so, and the baseline
    # itself stays frozen
    assert (adv.retrieval.top_k, adv.context.max_tokens) == (8, 4300)
    assert (base.retrieval.top_k, base.context.max_tokens) == (6, 3200)
    profile = (ROOT / "configs/pipelines/advanced.toml").read_text(encoding="utf-8")
    assert "--set retrieval.top_k=8 --set context.max_tokens=4300" in profile  # the fairness rule, stated
    assert adv.pipeline.retriever != base.pipeline.retriever and adv.pool_size > adv.retrieval.top_k


def test_cli_shows_each_pipelines_own_chunk_count_and_where_it_comes_from(capsys, monkeypatch):
    """Each pipeline's chunk counts come from its own profile, and the CLI shows where."""
    monkeypatch.chdir(ROOT)
    assert main(["config", "--pipeline", "baseline"]) == 0
    base = capsys.readouterr().out
    assert re.search(r"top_k\s+= 6\s+\(file:configs/pipelines/baseline\.toml\)", base)
    assert "chunks allowed in the prompt = 6" in base and "pipeline: baseline | retriever: dense | top_k: 6" in base
    assert main(["config", "--pipeline", "advanced"]) == 0
    adv = capsys.readouterr().out
    assert re.search(r"top_k\s+= 8\s+\(file:configs/pipelines/advanced\.toml\)", adv)
    assert "chunks allowed in the prompt = 8" in adv and "retrieval pool = 15" in adv
    assert re.search(r"retriever\s+= question_doc_slots\s+\(file:configs/pipelines/advanced\.toml\)", adv)
    # ...and the shared settings are the same for both
    assert re.search(r"chat\s+= gpt-4o-mini\s+\(file:configs/default\.toml\)", adv)
    assert main(["ask", "--pipeline", "nope", "--title", "t", "--body", "b"]) == 1
    err = capsys.readouterr().err
    assert "unknown pipeline 'nope'" in err and "supportdesk pipelines" in err  # before any index is read


def test_changing_the_default_does_not_move_the_baseline(tmp_path, monkeypatch):
    """default.toml is shared; the baseline profile pins the baseline's own values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "pipelines").mkdir(parents=True)
    (tmp_path / "configs" / "default.toml").write_text("[retrieval]\ntop_k = 15\n[context]\nmax_tokens = 5000\n")
    (tmp_path / "configs" / "pipelines" / "baseline.toml").write_text(
        "[retrieval]\ntop_k = 6\n[context]\nmax_tokens = 1800\n")
    base = Settings.load(env={})
    assert (base.retrieval.top_k, base.context.max_tokens) == (6, 1800)
    other = Settings.load(env={}, overrides=["pipeline.name=something-else"])
    assert (other.retrieval.top_k, other.context.max_tokens) == (15, 5000)


def test_schema_reference_is_up_to_date():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_schema_docs", ROOT / "scripts" / "gen_schema_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.render() == (ROOT / "docs" / "schemas.md").read_text(encoding="utf-8"), \
        "run: python scripts/gen_schema_docs.py"
