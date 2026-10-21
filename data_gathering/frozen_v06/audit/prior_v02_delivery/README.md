# Streamlit Session State — Reviewed Sources v02

A selective merge of two supplied source collections, preserving the existing fixed 30-case evaluation package. Communication can be Persian; all guides, metadata, and scripts in this delivery are English. Original source text is not translated.

**Status:** 164 issue reports, 21 canonical documentation pages, 263 release sections across 67 versions; 343 comments pending review and 0 approved comments. These are source records, not final chunks. No embeddings or agent evaluation have been run.

Read `REVIEW_REPORT.md` for the colleague review, merge rationale, media/claim policies, limitations, and evaluation design. `manifest.json` is the authoritative count/provenance summary. `validation.json` records structural checks, not a correctness score.

## Offline commands

From the extracted package directory, using Python 3.10 or later and the standard library:

```bash
python scripts/load_retrieval_sources.py
python scripts/validate_dataset.py
python scripts/build_review_cards.py
```

`load_retrieval_sources.py` returns only the four files under `retrieval_corpus/`. It rejects an unreviewed comment. Submit only each record's `text` to a future embedding service; retain `metadata` separately.

```python
import sys
sys.path.insert(0, "scripts")
from load_retrieval_sources import load_sources
sources = load_sources()
embedding_input = [source["text"] for source in sources]
# Source records still need code-aware chunking; this makes no API request.
```

## File map

| Path | Purpose |
|---|---|
| `retrieval_corpus/issues.jsonl` | Report text and separate provenance/environment metadata |
| `retrieval_corpus/comments.jsonl` | Explicitly approved excerpts only; initially empty |
| `retrieval_corpus/documents.jsonl` | One chosen representation per official page |
| `retrieval_corpus/release_sections.jsonl` | Version/date-specific changelog text |
| `review_pending/comments.jsonl` | Unreviewed technical candidates, outside retrieval |
| `review_pending/comment_review_cards.jsonl` | Parent and neighboring context for review |
| `agent_inputs/` | Initial development/test reports, one assigned case at a time |
| `evaluator_only/` | Gold/rubrics, future turns, full held-out threads; evaluator-only |
| `audit/` | Supplied snapshots, exclusions, merge/family/media records and historical scripts |
| `config/comment_reviews.json` | Explicit review decisions; initially `[]` |
| `config/comment_review_template.json` | Blank form; not an approval |
| `Embedding_Structure_v02.json` | Actual source envelope and proposed future chunk contract |
| `SHA256SUMS.json` | Delivery integrity, excluding this checksum file itself |

The ZIP is a developer/reviewer package. Give the agent only the allowed retrieval data and its current case. Never recursively index the ZIP or preload future test turns.

## Updating reviews and rebuilding

Read a review card and, when necessary, the full archived thread. Add a real decision to `config/comment_reviews.json`. Approval requires an exact selected excerpt, reviewer, rationale, a version-applicability note, allowed use, and evidence source IDs available in the retrieval corpus. `corroborated_claim` additionally requires an official documentation or release source. Human review must establish actual support and version applicability; merely listing an evidence ID does not do so.

After editing a configuration, remove the stale delivery checksums, rebuild, regenerate cards, and validate:

```bash
python -c "from pathlib import Path; Path('SHA256SUMS.json').unlink(missing_ok=True)"
python scripts/build_dataset.py
python scripts/build_review_cards.py
python scripts/validate_dataset.py
```

Counts and this frozen delivery's review report may need updating after new approvals. Recreate delivery checksums when publishing a changed package. `build_dataset.py` uses only bundled snapshots and makes no network/model calls. It preserves fixed evaluation selection and rubrics; historical audit scripts are not active build dependencies.

## Evaluation boundary

15 development + 15 test cases remain fixed. Each split has 6 answerable, 6 clarification, and 3 escalation cases. Ten scenario specifications include five test cases; no scenarios have been executed. Case selection is preserved, but rubric/scenario human signoff remains pending. Future turns must be released by the evaluator only when due.

Issue reports and community statements are not verified fixes. Media payloads have not been interpreted. Scope/family rules are conservative and require human review for a final benchmark claim.
