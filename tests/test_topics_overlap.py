"""Topics with overlap (several per chunk) and topics discovered by the LLM from the whole corpus."""
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from supportdesk.registry import make_pipeline
from supportdesk.retrieval.class_routing import (
    ClassRoutedRetriever,
    build_class_retriever,
    classify_chunks_multi,
    load_class_labels,
    topic_prompt,
    topics_file,
)
from supportdesk.retrieval.topic_discovery import clean_topics, discover_topics, sample_items
from supportdesk.types import CaseInput

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [{"name": "state", "definition": "Session State"}, {"name": "forms", "definition": "forms and callbacks"},
          {"name": "other", "definition": "none of the above"}]


def answer(clients, payload):
    clients.chat.script[clients.chat.calls + 1] = json.dumps(payload)


def test_each_chunk_gets_up_to_three_valid_topics(clients, index):
    chunks = index.chunks[:3]
    answer(clients, {"assignments": [{"i": 0, "labels": ["state", "forms", "nonsense", "other", "state"]},
                                     {"i": 1, "label": "forms"}, {"i": 9, "labels": ["state"]}]})
    got = classify_chunks_multi(chunks, clients.chat, prompt=topic_prompt(TOPICS),
                                allowed=[t["name"] for t in TOPICS])
    assert got[0] == ["state", "forms", "other"] and got[1] == ["forms"]  # invalid dropped, at most three
    assert got[2] == ["other"]  # unanswered: its keyword label, since "other" is one of these topics
    answer(clients, {"assignments": []})
    assert classify_chunks_multi(chunks[:1], clients.chat, prompt="p", allowed=["state", "forms"]) == [[]]


def test_a_chunk_with_several_topics_is_a_member_of_each(index, clients):
    labels = [["state", "forms"]] + [["other"]] * (len(index.chunks) - 1)
    r = ClassRoutedRetriever(index, clients.embedding, labels=labels, doc_slots=3, route="centroid",
                             topics=("state", "forms", "other"))
    assert 0 in r.members["state"] and 0 in r.members["forms"]


def test_labels_files_hold_one_or_several_topics(index, tmp_path):
    path = tmp_path / "l.jsonl"
    rows = [{"chunk_id": c.chunk_id, "labels": ["state", "forms"]} if n == 0 else {"chunk_id": c.chunk_id, "label": "other"}
            for n, c in enumerate(index.chunks)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    labels = load_class_labels(path, index)
    assert labels[0] == ["state", "forms"] and labels[1] == ["other"]


def test_routing_uses_the_topic_list_saved_with_the_labels(index, clients, settings, tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "labels": ["state"]}) + "\n" for c in index.chunks))
    topics_file(path).write_text(json.dumps({"topics": TOPICS}))
    s = dataclasses.replace(settings, paths=dataclasses.replace(settings.paths, class_labels=path),
                            retrieval=dataclasses.replace(settings.retrieval, class_route="llm"))
    r = build_class_retriever(index, clients, s)
    assert r.topics == ("state", "forms", "other") and "- forms: forms and callbacks" in r.prompt


def test_discovery_grows_then_merges_and_survives_a_bad_answer(clients):
    items = [f"text {n}" for n in range(6)]
    answer(clients, {"topics": [{"name": "Session State!", "definition": "state across reruns"}]})
    clients.chat.script[clients.chat.calls + 2] = "not json"  # the second batch: keep the list
    clients.chat.script[clients.chat.calls + 3] = json.dumps(
        {"topics": [{"name": "session_state", "definition": "state across reruns, merged"}]})
    topics = discover_topics(items, clients.chat, grow_prompt="grow {max_topics}", refine_prompt="{low}-{high}",
                             batch=3)
    assert [t["name"] for t in topics] == ["session_state", "other"]
    assert topics[0]["definition"] == "state across reruns, merged"
    assert clean_topics({"topics": [{"name": "A b", "definition": "x"}, {"name": "a_b", "definition": "y"}]}, 5) == \
        [{"name": "a_b", "definition": "x"}]


def test_the_sample_covers_every_summary_and_is_reproducible():
    records = [SimpleNamespace(source_type="issue", source_id=f"issue:{n}", title=f"t{n}", text="body " * 80)
               for n in range(10)]
    summaries = [{"title": "Page", "heading": f"H{n}", "summary": f"s{n}"} for n in range(4)]
    a = sample_items(records, summaries, per_type={"issue": 3}, seed=1)
    assert a == sample_items(records, summaries, per_type={"issue": 3}, seed=1) and len(a) == 7
    assert sum(x.startswith("[documentation]") for x in a) == 4


def test_the_diagnostic_counts_a_passage_reachable_through_any_of_its_topics(index, clients, settings, tmp_path):
    spec = importlib.util.spec_from_file_location("diagnose_clusters", ROOT / "scripts" / "diagnose_clusters.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "l.jsonl"
    path.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "labels": ["state", "forms"]}) + "\n"
                            for c in index.chunks))
    s = dataclasses.replace(
        settings, paths=dataclasses.replace(settings.paths, class_labels=path),
        pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off", retriever="class_doc_slots"),
        retrieval=dataclasses.replace(settings.retrieval, class_route="centroid", class_top=1, decompose=False))
    doc = next(c for c in index.chunks if c.source_type == "documentation")
    case = SimpleNamespace(case_id="g", input=CaseInput(case_id="g", title="widget", body="value lost"),
                           evidence_spans=[{"source_id": doc.source_id, "start_char": doc.start_char,
                                            "end_char": doc.end_char}])
    found = module.diagnose(s, index, clients, [case])
    assert found["reach"] == 1.0  # one routed topic is enough: the chunk carries both
    assert found["cases"][0]["gold_clusters"] == ["forms", "state"]
    assert make_pipeline("advanced", "class_doc_slots", index, clients, s).run(case.input).retrieved


def test_the_labelling_script_handles_chunks_left_without_a_topic(index, tmp_path, capsys):
    """Offline, the fake model's answers carry no topics: every chunk is left without one (an empty list),
    which the script must write, count and warn about rather than fail on."""
    spec = importlib.util.spec_from_file_location("build_class_labels", ROOT / "scripts" / "build_class_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index.save(tmp_path / "index")
    topics = tmp_path / "topics.json"
    topics.write_text(json.dumps({"topics": TOPICS[:2]}))  # no "other": the keyword fallback cannot apply
    labels = tmp_path / "discovered.jsonl"
    assert module.main(["--mode", "offline", "--method", "llm", "--topics", str(topics),
                        "--set", f"paths.index_dir={tmp_path / 'index'}", "--set", f"paths.class_labels={labels}"]) == 0
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert rows and all(r["labels"] == [] for r in rows) and topics_file(labels).exists()
    assert "WARNING" in capsys.readouterr().out
