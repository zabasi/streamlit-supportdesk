"""Before any retrieval run: would the section questions reach gold the search misses?

For every gold documentation passage of the development cases: whether the base search (dense with
documentation places, the pipeline's own query) already has its chunk among the candidates, and how high
its section ranks through the symptom questions. A passage is newly reachable when the base search misses
it and its section is among the best retrieval.question_sections by question. Free in replay mode.

    python scripts/diagnose_questions.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from supportdesk.config import Settings
from supportdesk.eval.cases import load_cases
from supportdesk.llm.factory import build_clients
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.index import VectorIndex
from supportdesk.retrieval.retriever import DocSlotsRetriever
from supportdesk.retrieval.section_questions import QuestionBank


def diagnose(settings: Settings, index: VectorIndex, clients, cases) -> dict:
    bank = QuestionBank.load(settings.paths.section_questions, expect_model=clients.embedding_model)
    base = DocSlotsRetriever(index, clients.embedding, doc_slots=settings.retrieval.doc_slots)
    writer = make_pipeline(settings.pipeline.name, "dense_doc_slots", index, clients, settings).query
    extra = settings.retrieval.question_sections
    by_source = defaultdict(list)
    for i, c in enumerate(index.chunks):
        by_source[c.source_id].append(i)
    rows = []
    for case in cases:
        query = writer(case.input, {})[0]
        candidates = {h.chunk.chunk_id for h in base.search(query, k=settings.pool_size)}
        rank_of = {section: n for n, (section, _) in enumerate(bank.ranking(
            clients.embedding.embed([query], stage="query")[0][0]), start=1)}
        for span in case.evidence_spans:
            a, b = span["start_char"], span["end_char"]
            chunks = [i for i in by_source.get(span["source_id"], []) if index.chunks[i].source_type == "documentation"]
            if not chunks:
                continue  # not documentation
            best = max(chunks, key=lambda i: max(0, min(b, index.chunks[i].end_char) - max(a, index.chunks[i].start_char)))
            section = next((n for n, s in enumerate(bank.sections) if s["source_id"] == span["source_id"]
                            and s["start_char"] <= a < s["end_char"]), None)
            found = index.chunks[best].chunk_id in candidates
            rank = rank_of.get(section) if section is not None else None
            rows.append({"case": case.case_id, "heading": bank.sections[section]["heading"] if section is not None else "?",
                         "in_candidates": found, "question_rank": rank,
                         "new": (not found) and rank is not None and rank <= extra})
    missed = [r for r in rows if not r["in_candidates"]]
    return {"rows": rows, "missed": len(missed), "new": sum(r["new"] for r in rows), "extra": extra}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["replay", "live", "offline"], default="replay")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", "pipeline.name=advanced", *args.set])
    clients = build_clients(settings)
    index = VectorIndex.load(settings.paths.index_dir, expect_model=clients.embedding_model)
    found = diagnose(settings, index, clients, load_cases(*settings.eval.case_files, split="dev"))
    print("gold documentation passages the base search misses (rank of their section through the questions):")
    for r in found["rows"]:
        if not r["in_candidates"]:
            mark = "NEW" if r["new"] else ""
            print(f"  {r['case']:>10}  {r['heading'][:60]:<60} question rank {r['question_rank']!s:>4}  {mark}")
    print(f"\n{found['missed']} gold documentation passages are not among the base candidates; "
          f"{found['new']} of them have their section among the best {found['extra']} by question "
          "(the retriever would add them as candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
