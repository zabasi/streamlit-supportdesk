"""The retriever belongs to the pipeline. Swapping it is an experiment: allowed on the command line,
announced everywhere it shows, and not offered on the web page."""
import json
import shutil

import pytest

from supportdesk.cli import main
from supportdesk.config import Settings
from tests.helpers import ROOT, SAMPLE
from tests.test_ui_and_report import _status_page

CASES = str(ROOT / "data" / "eval" / "sample_cases.jsonl")


def test_override_is_detected_only_when_the_retriever_differs_from_the_pipelines_own():
    assert Settings.load(env={}).retriever_override() is None
    assert Settings.load(env={}, overrides=["pipeline.name=advanced"]).retriever_override() is None
    swapped = Settings.load(env={}, overrides=["pipeline.name=advanced", "pipeline.retriever=dense"])
    assert swapped.own_retriever() == "hybrid" and swapped.retriever_override() == "hybrid"
    # the same choice made through the environment is still an override
    via_env = Settings.load(env={"SUPPORTDESK__PIPELINE__RETRIEVER": "bm25"})
    assert via_env.retriever_override() == "dense"


@pytest.fixture()
def base(tmp_path):
    args = ["--data-dir", str(SAMPLE), "--index-dir", str(tmp_path / "idx"),
            "--artifacts-dir", str(tmp_path / "art"), "--mode", "offline", "--cases", CASES]
    assert main(["index", *args]) == 0
    return args


def test_runs_record_an_override_and_the_report_and_snapshot_show_it(base, tmp_path, capsys):
    assert main(["eval", *base, "--run-id", "own"]) == 0
    assert main(["eval", *base, "--retriever", "bm25", "--run-id", "swapped"]) == 0
    runs = tmp_path / "art" / "runs"
    own = json.loads((runs / "own" / "summary.json").read_text())
    swapped = json.loads((runs / "swapped" / "summary.json").read_text())
    assert own["retriever"] == "dense" and own["retriever_override_of"] is None
    assert swapped["retriever"] == "bm25" and swapped["retriever_override_of"] == "dense"

    assert main(["report", "--artifacts-dir", str(tmp_path / "art"), "own", "swapped",
                 "--output", str(tmp_path / "r.html")]) == 0
    html = (tmp_path / "r.html").read_text()
    # every table heads its columns with the run titles: the marker follows the swapped run each time
    marker = "retriever override: bm25 (own: dense)"
    assert html.count(marker) == html.count("swapped<br>") > 0
    assert html.count("retriever override") == html.count(marker)  # never on the run as defined

    assert main(["snapshot", *base, "--retriever", "bm25", "--output-dir", str(tmp_path / "s")]) == 0
    assert "(**override**: baseline's own is `dense`)" in (tmp_path / "s" / "baseline.md").read_text()
    capsys.readouterr()


def test_the_web_command_no_longer_takes_a_retriever(capsys):
    with pytest.raises(SystemExit):
        main(["web", "--retriever", "bm25"])
    assert "unrecognized arguments: --retriever" in capsys.readouterr().err


STATUS = {"settings": {"mode": "offline"}, "index": {"chunk_count": 1, "embedding_model": "m"},
          "budget": {}, "pipelines": ["advanced", "baseline"], "defaults": {"pipeline": "baseline"},
          "retriever_by_pipeline": {"baseline": "dense", "advanced": "hybrid"},
          "limits": {"title": 300, "body": 8000, "created_at": 40}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_page_shows_the_retriever_and_marks_an_override(tmp_path):
    plain = _status_page(tmp_path, {**STATUS, "retriever_override_of": {"baseline": None, "advanced": None}},
                         switch_to="advanced")
    assert plain == {"first": {"pipeline": "baseline", "retriever": "dense"}, "afterSwitch": "hybrid"}
    marked = _status_page(tmp_path, {**STATUS, "retriever_override_of": {"baseline": None, "advanced": "hybrid"},
                                     "retriever_by_pipeline": {"baseline": "dense", "advanced": "bm25"}},
                          switch_to="advanced")
    assert marked["afterSwitch"] == "bm25 (override; advanced's own: hybrid)"
