"""pipeline.version_notes: what version each evidence piece describes, and how it may be used."""
import dataclasses

import pytest

from supportdesk.pipelines.packet import Packet
from supportdesk.pipelines.stages.controlled import ControlledGenerator
from supportdesk.pipelines.stages.protocols import GenerationFailed
from supportdesk.pipelines.stages.version_notes import annotate, notes, versions
from supportdesk.types import CaseInput, Chunk


def chunk(kind, version=None, as_of=None, meta=None, cid="c1"):
    return Chunk(chunk_id=cid, source_id=f"s-{cid}", source_type=kind, title="t", text="some evidence text",
                 version=version, as_of=as_of, meta=meta or {})


def test_versions_are_read_from_ranges_and_hashes_are_not_versions():
    assert versions("0.89.0 to 1.5.0") == [(0, 89, 0), (1, 5, 0)]
    assert versions("1.13") == [(1, 13, 0)]
    assert versions("695e6ce2fb1b23afc6df66b40420ab0013ad72fd") == []
    assert versions(None) == []


def test_documentation_is_flagged_when_the_report_is_on_an_older_version():
    older = notes(chunk("documentation", "1.65.0"), "1.11.0")
    assert "describing Streamlit 1.65.0" in older["version_applicability"]
    assert "1.11.0, which is older" in older["version_applicability"]
    assert "check that it exists in the reporter's version" in older["usage_constraint"]
    assert "older" not in notes(chunk("documentation", "1.65.0"), None)["version_applicability"]
    snapshot = notes(chunk("documentation", "695e6ce2fb1b"), "1.11.0")["version_applicability"]
    assert "the latest Streamlit" in snapshot and "1.11.0, which is older" in snapshot  # a commit, not a version
    assert "older" not in notes(chunk("documentation", "1.65.0"), "1.65.0")["version_applicability"]


@pytest.mark.parametrize("reported, expected", [
    ("1.11.0", "older than this release, so this change is not in it"),
    ("1.59.2", "already includes this change"),
    ("1.40.0 to 1.50.0", "span this release"),
])
def test_a_release_note_is_compared_with_the_reporters_version(reported, expected):
    n = notes(chunk("release_section", "1.46.0", "2025-06-01T00:00:00Z"), reported)
    assert "Release note for Streamlit 1.46.0 (2025-06-01)." in n["version_applicability"]
    assert expected in n["version_applicability"]
    assert "does not show the cause" in n["usage_constraint"]


def test_reports_and_comments_are_another_users_observation():
    assert "Another user's report on Streamlit 1.13.0, 2022-10-26." == notes(
        chunk("issue", "1.13", "2022-10-26T09:00:00Z"), None)["version_applicability"]
    assert "not verified documentation" in notes(chunk("comment"), None)["usage_constraint"]


def test_notes_a_source_already_has_are_kept():
    kept = annotate([chunk("documentation", "1.65.0", meta={"usage_constraint": "own rule", "evidence_role": "x"})], "1.0.0")[0]
    assert kept.meta["usage_constraint"] == "own rule" and kept.meta["evidence_role"] == "x"
    assert "older" in kept.meta["version_applicability"]


def generator(clients, settings, on):
    s = dataclasses.replace(settings, pipeline=dataclasses.replace(settings.pipeline, version_notes=on))
    return ControlledGenerator(clients, s)


def run(gen):
    case = CaseInput(case_id="C-1", title="Widget state lost", body="state lost between pages",
                     environment={"streamlit_version": "1.11.0"})
    packet = Packet(chunks=[chunk("release_section", "1.46.0", "2025-06-01T00:00:00Z", cid="r1"),
                            chunk("documentation", "1.65.0", cid="d1")], labels={"S1": "r1", "S2": "d1"})
    with pytest.raises(GenerationFailed):  # the offline fake model does not write an assessment
        gen(case, {}, packet)
    return gen.chat.seen[-1][1]["content"], packet


def test_off_by_default_the_request_and_packet_are_unchanged(clients, settings):
    request, packet = run(generator(clients, settings, False))
    assert "older than this release" not in request and "version_applicability" not in request
    assert all(not c.meta for c in packet.chunks)


def test_on_the_notes_reach_the_models_request_through_the_package(clients, settings):
    request, packet = run(generator(clients, settings, True))
    assert "The reporter's version (1.11.0) is older than this release, so this change is not in it." in request
    assert "which is older: options, defaults and behaviour" in request
    assert all(not c.meta for c in packet.chunks)  # the recorded packet is not changed
