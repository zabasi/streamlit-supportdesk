"""controlled_v4: v2 with three added prompt rules, its own manifest, and no evaluation-case content."""
import dataclasses
import hashlib
import json
import re
from pathlib import Path

from supportdesk.eval.cases import load_cases
from supportdesk.pipelines.stages.controlled import ControlledGenerator
from tests.test_controlled_v2 import CASE, assessment, packet

ROOT = Path(__file__).resolve().parents[1]
V2, V4 = (ROOT / "src/supportdesk/generation" / v for v in ("controlled_v2", "controlled_v4"))
RULES = {  # anchor in v2's prompt -> the complete rule added right after it
    "Inspect the original report/history before asking again.":
        " Before recording a gap, check the report's title, body, code and environment; if any of them states the "
        "fact, set already_supplied_or_tested to true.",
    "Use null when no internal investigation is established.":
        " If the report already supplies runnable code or exact steps, the version, and the observed versus expected "
        "behavior, and the evidence does not explain the observed behavior, record internal_investigation. Optional "
        "confirmations, such as trying a workaround or retesting on another version, are not material gaps.",
    "Keep text concise and factual.":
        " Keep user-facing text short: an answer message of at most 120 words, a gap's why_it_matters of at most 40 "
        "words, and a question of at most 30 words.",
}


def test_only_the_prompt_differs_from_v2_by_exactly_the_three_rules():
    for name in ("__init__.py", "adapter.py", "context.py", "generation.py"):
        assert (V4 / name).read_bytes() == (V2 / name).read_bytes(), name
    expected = (V2 / "prompt.txt").read_text(encoding="utf8")
    for anchor, rule in RULES.items():
        assert expected.count(anchor) == 1, anchor
        expected = expected.replace(anchor, anchor + rule, 1)
    assert (V4 / "prompt.txt").read_text(encoding="utf8") == expected


def test_v4_has_its_own_manifest_and_no_development_inputs():
    manifest = json.loads((V4 / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["frozen_files"].items():
        assert hashlib.sha256((V4 / name).read_bytes()).hexdigest() == digest, name
    assert manifest["based_on"].startswith("controlled-routing-v2") and not (V4 / "inputs").exists()


def test_the_v4_prompt_contains_no_evaluation_case_ids_or_titles():
    prompt = " ".join((V4 / "prompt.txt").read_text(encoding="utf8").split()).casefold()
    assert not re.search(r"\b(dev|test)-\d+", prompt)
    for split in ("dev", "test"):
        try:
            cases = load_cases(ROOT / "data/eval", split=split)
        except FileNotFoundError:
            continue
        for case in cases:
            title = " ".join(case.input.title.split()).casefold()
            assert len(title) < 12 or title not in prompt, case.case_id


def test_the_stage_sends_the_v4_prompt(clients, settings):
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, controlled_version="v4"))
    gen = ControlledGenerator(clients, s)
    gen.chat.script[1] = json.dumps(assessment())
    seen = []
    original = gen.chat.complete
    gen.chat.complete = lambda messages, **kw: seen.append(messages) or original(messages, **kw)
    answer = gen(CASE, {}, packet())
    assert all(rule.strip() in seen[-1][0]["content"] for rule in RULES.values())
    assert answer.attempts[0]["package"] == "controlled_v4"
