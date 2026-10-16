# Policy v3 prompt revision vs original

Generated 2026-10-16T15:53:33+00:00.

Same 15 annotated development cases, same gold passages, same model `gpt-5.4-mini`. Only the system prompt changed.

## Why the original prompt was weak

- It wrote workflow findings before choosing an action, so the model often shipped a patch (`answer`) or a product handoff (`escalate`).
- The line “Missing environment details do not block a visible application-code correction” pushed it to answer when gold wanted `request_info` (missing version, unshown write, missing example).
- `escalate` was the leftover bucket whenever the model failed to name a useful discriminating question.
- A dummy-key/cleanup workaround could convert a reproduced framework mismatch into `answer` (DEV-10805).

The new prompt is `policy_v3b_system.txt`. It forces `request_info` → `answer` → `escalate` in that order.

## Outcome comparison

| method | prompt | correct | accuracy | cost USD |
|---|---|---:|---:|---:|
| generation folder policy v3 | original | 9/15 | 0.6000 | 0.215597 |
| this repo original live v3 | `policy_v3_system.txt` | 9/15 | 0.6 | 0.256039 |
| this repo new prompt | `policy_v3b_system.txt` | 12/15 | 0.8 | 0.197478 |

### Original live v3 confusion

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 3 | 1 | 2 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 10 | 1 | 4 | 15 |

### New prompt confusion

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 0 | 5 | 1 | 6 |
| escalate | 2 | 0 | 1 | 3 |
| **total** | 8 | 5 | 2 | 15 |

## Per-case actions

| case_id | gold | original live v3 | new prompt | changed | new cost |
|---|---|---|---|---|---:|
| DEV-5604 | request_info | escalate | request_info | yes | 0.01329225 |
| DEV-7044 | request_info | answer | request_info | yes | 0.00695925 |
| DEV-9342 | request_info | request_info | request_info |  | 0.00928125 |
| DEV-4338 | answer | answer | answer |  | 0.0174285 |
| DEV-6166 | escalate | escalate | escalate |  | 0.0156765 |
| DEV-16388 | escalate | escalate | answer | yes | 0.016482 |
| DEV-3903 | request_info | answer | request_info | yes | 0.0128475 |
| DEV-4325 | answer | answer | answer |  | 0.016182 |
| DEV-7305 | answer | answer | answer |  | 0.0146145 |
| DEV-3585 | request_info | escalate | escalate |  | 0.0063315 |
| DEV-3842 | answer | answer | answer |  | 0.01166175 |
| DEV-4082 | answer | answer | answer |  | 0.01759425 |
| DEV-5591 | request_info | answer | request_info | yes | 0.0051 |
| DEV-4989 | answer | answer | answer |  | 0.016125 |
| DEV-10805 | escalate | answer | answer |  | 0.01790175 |

## Which prompt is best

**The new prompt is better** on this 15-case set: 12/15 vs previous live v3 9/15.

Gold evidence is not a RAG ranking. Do not treat contract-valid JSON as a verified Streamlit patch.
