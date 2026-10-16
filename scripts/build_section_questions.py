"""Write and embed symptom questions for every documentation section (the question_doc_slots retriever).

For each section in paths.summaries (the documentation's top-level sections), the model writes a few
problem descriptions a user might write when that section is what they need (prompts.section_questions);
they are embedded with the index's model and saved with their vectors in paths.section_questions.
Built from the documentation only. About 146 calls (about $0.02), once per corpus:

    python scripts/build_section_questions.py --mode live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supportdesk.config import Settings
from supportdesk.data.corpus import load_corpus
from supportdesk.llm.factory import build_clients
from supportdesk.retrieval.section_questions import QuestionBank, generate_questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["live", "offline", "replay"], default="live")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    parser.add_argument("--per-section", type=int, default=5)
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", *args.set])
    clients = build_clients(settings)
    sections = [json.loads(line) for line in Path(settings.paths.summaries).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    sections = [{k: s[k] for k in ("source_id", "start_char", "end_char", "title", "heading")} for s in sections]
    texts = {r.source_id: r.text for r in load_corpus(settings.paths.data_dir).records}
    mark = clients.ledger.mark()
    questions = generate_questions(sections, texts, clients.chat, n=args.per_section,
                                   prompt=Path(settings.prompts.section_questions).read_text(encoding="utf-8"))
    flat = [q for qs in questions for q in qs]
    vectors = clients.embedding.embed(flat, stage="section_questions")[0] if flat else []
    QuestionBank(sections, questions, vectors, clients.embedding_model).save(settings.paths.section_questions)
    empty = sum(not qs for qs in questions)
    for s, qs in list(zip(sections, questions))[:3]:
        print(f"  {s['title']} > {s['heading']}")
        for q in qs:
            print(f"      - {q}")
    if empty > 0.05 * len(sections):
        print(f"WARNING: {empty} of {len(sections)} sections have no questions (unusable model answers): "
              "they can only be found through the normal search")
    spent = clients.ledger.summarize(mark)
    print(f"{len(flat)} questions for {len(sections)} sections ({empty} without any) -> "
          f"{settings.paths.section_questions} | cost ${spent.get('cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
