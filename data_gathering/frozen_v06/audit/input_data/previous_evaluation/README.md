# Streamlit Session State: 30-case Evaluation Split

This is a fixed selection from the 2026-10-06 source snapshot, not an executed evaluation or an implemented agent. All authored files are English; source text is preserved in its original language.

## What is ready

- 15 development and 15 final-test cases; five topics with three cases per split per topic.
- 10 multi-turn scenario specifications, five in each split. Four use real user-comment follow-ups and six use explicitly marked designed follow-ups. All local-tracker events are designed specifications.
- Per-case proposed evaluation rubrics, acceptable document sections with character offsets, and withheld source threads for curator review.
- A separate retrieval corpus: 154 issues, 580 comments, 20 distinct documentation pages, and 269 release-note sections across 67 versions.
- 66 issue threads withheld: 30 evaluation cases plus 36 conservatively linked/semantically similar reports. Direct-answer release/document blocks are removed and recorded in the audit.

## Access boundaries

Only index `retrieval_corpus/`. Never index `agent_inputs/`, `evaluator_only/`, `audit/`, `selected_cases.csv`, or the original unsplit dataset. Never make the evaluator directory accessible through the agent's file/search tools.

`agent_inputs/development_inputs.jsonl` and `agent_inputs/test_inputs.jsonl` contain only each scenario's initial report; they omit final labels, outcomes, all comments, gold answers, and future turns. Give the agent one current case at a time. The runner must reveal one follow-up only after the preceding assistant turn. Its tool access must enforce that boundary; file layout alone is not a security control.

Development cases may be used repeatedly to adjust prompts, retrieval, and models. Final-test rubrics may be inspected by the designated dataset curator to validate the dataset, but must not be used to tune the system. Freeze the system before final testing. Compare baseline and final system on the same cases and source snapshot.

## What still needs implementation or review

The proposed rubrics have been reviewed against the source text by the assistant, not independently signed off by a human domain expert. The team should review the 30 rubrics, accept multiple defensible responses, and correct any evidence/decision ambiguity before running evaluation. This does not require line-by-line human review of all 220 source issues.

The multi-turn JSON is an evaluator specification, not a working scenario runner. Implement persistent local tracker storage, proposal-specific/revision-specific approval, idempotent updates, fault injection, sequential turn delivery, and stored-state assertions. No operational tests or LLM response scoring have run in this preparation step.

The original source bodies are current snapshots, not reconstructed historical reports. Therefore evaluation is **current-snapshot replay**, not historical evaluation. Source documentation may describe newer behavior than an old report; every rubric requires version-aware interpretation. An exact fixed release is not required when its only direct supporting evidence is withheld. Do not claim a fix version from a closure label.

Explicit references are a conservative quarantine mechanism, not proof of duplicates. Additional semantic quarantines protect closely similar prepopulation/callback reports. Some topical overlap is necessary for retrieval; direct duplicates, linked answers, and exact outcome references are withheld. The package does not prove that all possible semantic leakage has been eliminated; the team should review remaining corpus items if evaluation identifies suspicious overlaps.

The release-note subset is keyword-selected after direct-reference filtering and still benefits from relevance review. Canonical documentation selects pinned complete Markdown when available and rendered content otherwise. Original hashes and filtered retrieval hashes have separate meanings.

## Multi-turn operational coverage

Normal approval, rejection, maintainer edits, restart while awaiting approval, repeated action request, user fact correction, case changes before execution, timeout after a committed write, failed evidence search, and switching between two cases are each specified once. At least five scenarios belong to final test. Designed turns are marked as designed and must not be presented as actual historical user replies or successful executed tests.

## Rebuild and validate

Use Python 3.10+; scripts depend only on the standard library.

```bash
python scripts/build_evaluation.py --dataset /path/to/Streamlit_Session_State_Dataset_2026-10-06 --output /path/to/new_split
python scripts/validate_split.py --package /path/to/new_split
```

Review `selected_cases.csv` for assignments, `summary.json` for counts, and `audit/exclusion_manifest.json` for exact exclusions. Source collection and this split preparation incurred zero model or embedding API calls. No chunking, embeddings, retrieval index, agent, or evaluated response metrics are included.
