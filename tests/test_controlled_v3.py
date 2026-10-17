"""controlled_v3: v2 with one added prompt rule, its own manifest, and no evaluation-case content."""
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from supportdesk.eval.cases import load_cases
from supportdesk.pipelines.stages.controlled import ControlledGenerator
from tests.test_controlled_v2 import CASE, assessment, packet

ROOT = Path(__file__).resolve().parents[1]
V2, V3 = (ROOT / "src/supportdesk/generation" / v for v in ("controlled_v2", "controlled_v3"))
RULE = ("When the report already supplies a minimal reproduction or exact steps, and the reported behavior "
        "contradicts applicable documentation or is reported on a recent release, record internal_investigation.")


def test_only_the_prompt_differs_from_v2_by_the_one_added_rule():
    for name in ("__init__.py", "adapter.py", "context.py", "generation.py"):
        assert (V3 / name).read_bytes() == (V2 / name).read_bytes(), name
    v2, v3 = (p.read_text(encoding="utf8") for p in (V2 / "prompt.txt", V3 / "prompt.txt"))
    anchor = "Use null when no internal investigation is established."
    added = v3.split(anchor, 1)[1][: len(v3) - len(v2)]
    assert v3 == v2.replace(anchor, anchor + added, 1) and added.startswith(" " + RULE)


def test_v3_has_its_own_manifest_and_no_development_inputs():
    manifest = json.loads((V3 / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["frozen_files"].items():
        assert hashlib.sha256((V3 / name).read_bytes()).hexdigest() == digest, name
    assert manifest["based_on"].startswith("controlled-routing-v2") and not (V3 / "inputs").exists()


def test_the_v3_prompt_contains_no_evaluation_case_ids_or_titles():
    prompt = (V3 / "prompt.txt").read_text(encoding="utf8")
    assert not re.search(r"\b(DEV|TEST)-\d+", prompt)
    for split in ("dev", "test"):
        try:
            cases = load_cases(ROOT / "data/eval", split=split)
        except FileNotFoundError:
            continue
        for case in cases:
            title = " ".join(case.input.title.split()).casefold()
            assert len(title) < 12 or title not in " ".join(prompt.split()).casefold(), case.case_id


@pytest.mark.parametrize("version, package, has_rule", [("v2", "controlled_v2", False), ("v3", "controlled_v3", True)])
def test_the_stage_sends_the_chosen_versions_prompt(clients, settings, version, package, has_rule):
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, controlled_version=version))
    gen = ControlledGenerator(clients, s)
    gen.chat.script[1] = json.dumps(assessment())
    seen = []
    original = gen.chat.complete
    gen.chat.complete = lambda messages, **kw: seen.append(messages) or original(messages, **kw)
    answer = gen(CASE, {}, packet())
    assert (RULE in seen[-1][0]["content"]) is has_rule
    assert answer.attempts[0]["package"] == package
