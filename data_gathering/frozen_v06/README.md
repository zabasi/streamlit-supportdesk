# Streamlit Session State v06

540 selected source records. Every v05 text and text hash is unchanged. Only runtime metadata and evaluator schemas were revised. No chunking, embedding, network collection, paid API call or execution of source examples was performed.

## Index input
Use only the five explicit files in retrieval_corpus. scripts/load_retrieval_sources.py loads them. Embed text only. Do not recursively load audit, evaluator_only, config, reports or agent_inputs. Top-level source_id is authoritative; duplicated metadata IDs were removed.

## Labels and offsets
All inherited 60 evidence spans were remapped from archived canonical document bodies to exact delivered corpus text; four additional official witnesses were selected from existing sources. Each evidence span has source_id, start_char, end_char, exact text, evidence hash and parent text hash. Offsets are Python Unicode code points, with an exclusive end. Source URLs and snapshot versions are obtained from corpus metadata. Legacy evidence and offsets remain in audit/alignment_v06/evidence_span_mapping.json.

After actual chunking, run scripts/map_evidence_to_chunks.py with a supplied chunks.jsonl. It checks each chunk against the parent text and hash, reports overlap and complete/partial interval coverage, and fails on stale or mismatched data. No chunk IDs are fabricated in this source-only delivery.

## Evaluation
agent_inputs/development_inputs.jsonl and test_inputs.jsonl retain 15 cases each and all original report text. They now include dates and exact explicit environment lines. The corresponding evaluator rubrics contain required_missing_information, known_facts, verification_tasks, permitted answer scope and claim support. Historical facts must not be requested again as though missing.

Ten multi_turn_scenarios have explicit steps and response checkpoints. scripts/evaluation_session.py provides staged conversation delivery and never returns gold, oracle fields, or unreleased follow-ups in its agent view. Tracker approval/restart/timeout specifications still need actual integration; no agent or tracker behavior score is claimed.

Six history_variant_inputs are separate fixtures, three per split, with an explicitly designed assistant question and a recorded or designed user follow-up. Their evaluator-only rubric asks the system to resume with updated facts. They are derivatives, not six independent new incidents.

Eight source_usage_inputs are separate tasks for historical reports, comment proposals, release statements and linked discussions. Their rubrics require correct evidence roles and version scope. They do not import the answers to held-out issues into the index.

## Verification and rebuilding
Run python3 scripts/validate_dataset.py for offline differential and schema checks. Run python3 scripts/build_dataset.py to regenerate v06 sources and evaluator data from the immutable v05 baseline and recorded case review config; then validate. Run python3 scripts/finalize_v06.py after any dataset edits to validate and refresh the package SHA256SUMS inventory. scripts/test_v06_contracts.py checks staged disclosure, Unicode offsets, cross-boundary chunk mapping and stale-hash rejection using fixtures only. Suffixed older builders/validators are archived tools, not active v06 entrypoints.

## Review limits
The 30 cases were reviewed for a bounded response, discriminating question or escalation summary. This does not certify an exact historical fix, every technical claim, or end-to-end RAG performance. Cases that require more observation retain that uncertainty. Human signoff remains pending. Older CHANGE_REPORT_v04_FA.md and CHANGE_REPORT_v05_FA.md are historical snapshots; use ALIGNMENT_REPORT_v06_FA.md and the current manifest, README and validation as the v06 contract.
