"""The command line as a user meets it: choosing a pipeline, config-driven cases, safety guards."""
import json
import re
from pathlib import Path

import pytest

from supportdesk.cli import main
from tests.helpers import ROOT, SAMPLE

CASES = str(ROOT / "data" / "eval" / "sample_cases.jsonl")


@pytest.fixture()
def run(tmp_path, capsys):
    """Run a command against the sample data with every output under tmp_path."""
    base = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline"]

    def go(command, *args, bare=False):
        code = main([command, *args] if bare else [command, *base, *args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err
    return go


def test_pipelines_lists_what_you_can_select_and_how_each_is_configured(monkeypatch, run):
    monkeypatch.chdir(ROOT)
    code, out, _ = run("pipelines", bare=True)
    assert code == 0
    assert re.search(r"baseline\s+dense\s+6\s+configs/pipelines/baseline\.toml\s+Reference", out)
    assert re.search(r"advanced\s+dense_doc_slots\s+8\s+configs/pipelines/advanced\.toml\s+Improved", out)
    assert "selected: baseline" in out
    assert "selected: advanced" in run("pipelines", "--pipeline", "advanced", bare=True)[1]


def test_every_command_that_runs_a_pipeline_says_which_one(run):
    cases = ["--cases", CASES]
    assert run("index", *cases)[0] == 0
    code, out, err = run("ask", "--title", "session state refresh", "--body", "lost after F5")
    assert code == 0 and err.startswith("pipeline: baseline | retriever: dense | top_k: 6")
    assert json.loads(out)["pipeline"] == "baseline"
    code, out, _ = run("eval", *cases)
    assert out.startswith("pipeline: baseline | retriever: dense | top_k: 6 | profile: ")
    code, out, err = run("ask", "--pipeline", "advanced", "--retriever", "bm25", "--title", "t",
                         "--body", "session state")
    # an overridden retriever is announced, so the run is never mistaken for the pipeline as defined
    assert err.startswith("pipeline: advanced | retriever: bm25 (OVERRIDE: advanced's own is dense_doc_slots) | top_k: 8")
    assert json.loads(out)["pipeline"] == "advanced" and len(json.loads(out)["retrieved"]) <= 6


def test_an_unknown_pipeline_is_rejected_with_the_list_before_anything_loads(run):
    code, _, err = run("ask", "--pipeline", "nope", "--title", "t", "--body", "b")
    assert code == 1 and "unknown pipeline 'nope'" in err and "advanced" in err and "baseline" in err
    code, _, err = run("eval", "--retriever", "nope", "--cases", CASES)
    assert code == 1 and "unknown retriever 'nope'" in err and "dense" in err


def test_case_files_in_the_config_mean_no_cases_flag_anywhere(tmp_path, run):
    cfg = tmp_path / "exp.toml"
    cfg.write_text(f'[eval]\ncase_files = ["{CASES}"]\n')
    use = ["--config", str(cfg)]
    assert run("index", *use)[0] == 0
    assert run("check-cases", *use)[0] == 0
    code, out, _ = run("eval", *use, "--run-id", "x")
    assert code == 0 and "cases: 2" in out
    assert run("eval", *use, "--split", "test")[0] == 2  # the test split still needs --final


def test_a_missing_case_file_is_reported_by_name(run):
    run("index", "--cases", CASES)
    code, _, err = run("eval", "--cases", "/nope/cases.jsonl")
    assert code == 1 and "/nope/cases.jsonl" in err and "eval.case_files" in err


def test_index_without_case_files_warns_that_nothing_is_excluded(run):
    code, out, err = run("index", "--cases", "/nope/cases.jsonl")
    assert code == 0 and "nothing is excluded" in err and '"chunk_count": 17' in out


def test_eval_refuses_to_run_when_the_index_contains_evaluation_issues(run):
    assert run("index", "--cases", "/nope/cases.jsonl")[0] == 0  # built without exclusions
    for command in ("eval", "eval-retrieval"):
        code, _, err = run(command, "--cases", CASES)
        assert code == 2 and "LEAK:" in err and "rebuild it with `supportdesk index`" in err
    code, out, _ = run("check-cases", "--cases", CASES)
    assert code == 2 and "FAIL: leak:" in out


def test_split_uses_the_configured_fraction_and_seed(tmp_path, run):
    rows = [{"case_id": f"c{i}", "family_id": f"f{i}", "split": "dev"} for i in range(10)]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def split(*flags):
        return run("split", str(src), *flags, bare=True)[1]

    def n_test(text):
        return sum(json.loads(x)["split"] == "test" for x in text.splitlines())
    assert n_test(split("--test-fraction", "0.3")) == 3
    assert n_test(split("--test-fraction", "0.3", "--seed", "99")) == 3  # any seed keeps the fraction
    assert split("--seed", "99") == split("--seed", "99")  # and a seed is deterministic




def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for name in ("status", "config", "pipelines", "index", "ask", "eval",
                 "eval-retrieval", "check-cases", "web", "report", "split"):
        assert name in out
    assert Path(CASES).exists()


def test_dotenv_in_the_working_directory_is_loaded_and_set_variables_win(tmp_path, monkeypatch):
    import os

    from supportdesk.cli import load_dotenv_file
    (tmp_path / ".env").write_text("SUPPORTDESK_TEST_A=from-file\nSUPPORTDESK_TEST_B=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SUPPORTDESK_NO_DOTENV")
    monkeypatch.setenv("SUPPORTDESK_TEST_B", "already-set")
    monkeypatch.delenv("SUPPORTDESK_TEST_A", raising=False)
    load_dotenv_file()
    assert os.environ["SUPPORTDESK_TEST_A"] == "from-file" and os.environ["SUPPORTDESK_TEST_B"] == "already-set"
    monkeypatch.delenv("SUPPORTDESK_TEST_A")
