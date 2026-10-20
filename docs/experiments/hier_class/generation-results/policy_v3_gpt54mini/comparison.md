# Generation policy v3

Generated 2026-10-16T15:39:59+00:00.

This is the tested support-record generator (one JSON call, literal report quotes, official `answer` / `request_info` / `escalate`). It is not the grounded checker's claim-support checker.

## Measured run (2026-10-16, gold passages, gpt-5.4-mini)

15/15 contract-valid. Action match **9/15 (60%)**. Evidence was 81 curated selected passages, not retrieval chunks. Content accuracy was not scored.

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 5 | 0 | 1 | 6 |
| request_info | 0 | 2 | 4 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 6 | 2 | 7 | 15 |

Read a row to see where that gold label went. Clarification is underused (4 of 6 `request_info` became `escalate`). Direct app-code answers are the strong cell (5/6).

The preselected policy v3 completes all 15 fresh calls and preserves report-text origin, but does not establish a strong overall generation solution. Direct visible application mistakes are handled more reliably than clarification, historical-version interpretation and multi-part product concerns. Stored action agreement is 9/15; the unchanged older full-development run also had 9/15, with different errors and a different prompt/schema/token ceiling. On this assistant content review, 3 are usable by inspection, 4 need minor revision and 8 need material revision; these are review categories, not independent content accuracy. Clarification is underused and escalation overused. Do not modify labels or tune on held-out test data to inflate this result. No further paid calls or prompt revisions are part of this evaluation.

## This repo run

evidence `gold`  model `gpt-5.4-mini`  mode `live`  outcome accuracy 0.6

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 3 | 1 | 2 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 10 | 1 | 4 | 15 |

cases: 15   outcomes {'answer': 10, 'escalate': 4, 'request_info': 1}   outcome accuracy 0.6
           ambiguous 0.1667 (n=6)  answerable 1.0 (n=6)  escalation 0.6667 (n=3)

confusion (gold \ predicted)
| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 3 | 1 | 2 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 10 | 1 | 4 | 15 |
grounding  claim support 0.0   citation precision 0.0   outside packet 56   answered after failure 0
refusals   abstention P/R/F1 0.0/n/a/n/a   answered when packet had gold 0.6667
usefulness completeness vs gold spans 0.2005   answers cite gold 0.0   instruction echo 0
