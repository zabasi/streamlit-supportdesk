# Streamlit Session State selected sources — v03

This is the complete reviewed source package before chunking/embedding. Use only the four files listed below; do not recursively ingest this package.

| Source | Count |
|---|---:|
| Issues | 126 |
| Selected comment excerpts | 214 |
| Documentation pages, with selected relevant sections | 21 |
| Selected release-note entries | 111 |

The source text preserves useful prose and code. Images/videos are neither interpreted nor supplied as embedding payloads. A media-containing comment is admitted only when its selected text is understandable without the media.

Load `retrieval_corpus/issues.jsonl`, `comments.jsonl`, `documents.jsonl`, and `release_sections.jsonl` with `python scripts/load_retrieval_sources.py`. Embed only `text`; keep `metadata` alongside the resulting chunks for citations and version/evidence interpretation. These records are sources, not generated chunks.

Comment selection was performed by an AI assistant using issue/thread context. The 214 selected excerpts include historic observations and diagnoses; they are not 214 runtime-verified fixes. The 13 uncertain comments stay outside the input in `review_pending`. Exclusion and excerpt decisions are in `config` and `audit`; these are not retrieval input.

Rebuild offline with `python scripts/build_dataset.py`, then `python scripts/build_review_cards.py`, then `python scripts/validate_dataset.py`. Review decisions are fixed supplied inputs to rebuilding; rebuilding does not rerun an LLM. Validation checks source identities, provenance, selection, integrity and evaluation separation, not semantic truth or RAG performance.

The preserved evaluation has 15 development cases, 15 test cases and 10 multi-turn specifications. Evaluation threads, future turns and gold are excluded from retrieval; the scenario specifications have not been executed.

The project brief fixes a total team API cap of USD 5 via Metis, including development, corpus/query embeddings, tests, model evaluation, final and bonus runs. Model choice is deferred. This preparation makes no Metis API or embedding calls. Overall prior spending and actual future model rates are unknown. Count tokens with the chosen tokenizer and measure a small pilot before broad execution; reuse unchanged embeddings.

See `REVIEW_REPORT.md`, `Embedding_Structure_v03.json`, `validation.json` and `audit/preparation_summary.json` for decisions, limits and measured volume.
