from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from supportdesk.data.corpus import load_corpus
from supportdesk.eval.cases import exclusions_from_cases, load_cases
from supportdesk.indexing import run_indexing
from supportdesk.llm.factory import build_clients
from tests.helpers import SAMPLE, make_settings

CASES = ROOT / "data" / "eval" / "sample_cases.jsonl"


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    """Tests never see the developer's shell settings (SUPPORTDESK_CONFIG from the README quick-start,
    SUPPORTDESK__ overrides, alias variables, API keys), so they pass the same on every machine. A test
    that needs one sets it itself with monkeypatch."""
    import os

    from supportdesk.config import ALIASES, BASE_URL_ENV, ENV_PREFIX
    for name in list(os.environ):
        if name.startswith(ENV_PREFIX) or name == "SUPPORTDESK_CONFIG" or name in ALIASES:
            monkeypatch.delenv(name)
    for name in ("OPENAI_API_KEY", BASE_URL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUPPORTDESK_NO_DOTENV", "1")  # a developer's ./.env never reaches the tests


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(SAMPLE)


@pytest.fixture(scope="session")
def cases():
    return load_cases(CASES)


@pytest.fixture()
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture()
def clients(settings):
    return build_clients(settings)


@pytest.fixture()
def indexed(cases, clients, settings):
    """The real indexing graph (load -> clean -> chunk -> audit -> embed -> store) on the sample."""
    return run_indexing(settings, clients, cases=cases, exclusions=exclusions_from_cases(cases))


@pytest.fixture()
def chunks(indexed):
    return indexed["chunks"]


@pytest.fixture()
def index(indexed):
    return indexed["index"]
