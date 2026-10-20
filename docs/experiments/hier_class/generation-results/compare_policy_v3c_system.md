# Policy v3 prompt revision

Generated 2026-10-16T16:01:21+00:00.

Same 15 annotated development cases, same gold passages, same model `gpt-5.4-mini`. Only the system prompt changed.

The new prompt is `policy_v3c_system.txt`.

## Outcome comparison

| method | prompt | correct | accuracy | cost USD |
|---|---|---:|---:|---:|
| generation folder policy v3 | original | 9/15 | 0.6000 | 0.215597 |
| this repo original live v3 | `policy_v3_system.txt` | 9/15 | 0.6 | 0.256039 |
| this repo v3b | `policy_v3b_system.txt` | 12/15 | 0.8 | 0.197478 |
| this repo new prompt | `policy_v3c_system.txt` | 11/15 | 0.7333 | 0.184565 |

### Original live v3 confusion

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 3 | 1 | 2 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 10 | 1 | 4 | 15 |

### v3b confusion

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 0 | 5 | 1 | 6 |
| escalate | 2 | 0 | 1 | 3 |
| **total** | 8 | 5 | 2 | 15 |

### New prompt confusion

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 5 | 0 | 1 | 6 |
| request_info | 1 | 3 | 2 | 6 |
| escalate | 0 | 0 | 3 | 3 |
| **total** | 6 | 3 | 6 | 15 |

## Per-case actions

| case_id | gold | original v3 | v3b | new prompt | vs v3b | new cost |
|---|---|---|---|---|---|---:|
| DEV-5604 | request_info | escalate | request_info | escalate | yes | 0.0154335 |
| DEV-7044 | request_info | answer | request_info | request_info |  | 0.006279 |
| DEV-9342 | request_info | request_info | request_info | escalate | yes | 0.009267 |
| DEV-4338 | answer | answer | answer | escalate | yes | 0.01492125 |
| DEV-6166 | escalate | escalate | escalate | escalate |  | 0.01140525 |
| DEV-16388 | escalate | escalate | answer | escalate | yes | 0.01305225 |
| DEV-3903 | request_info | answer | request_info | answer | yes | 0.01566825 |
| DEV-4325 | answer | answer | answer | answer |  | 0.01192875 |
| DEV-7305 | answer | answer | answer | answer |  | 0.01330425 |
| DEV-3585 | request_info | escalate | escalate | request_info | yes | 0.00584475 |
| DEV-3842 | answer | answer | answer | answer |  | 0.0110085 |
| DEV-4082 | answer | answer | answer | answer |  | 0.0135525 |
| DEV-5591 | request_info | answer | request_info | request_info |  | 0.00834375 |
| DEV-4989 | answer | answer | answer | answer |  | 0.01678575 |
| DEV-10805 | escalate | answer | answer | escalate | yes | 0.0177705 |

## Which prompt is best

**Escalate is better; overall is not.** v3c scores 11/15 vs v3b 12/15.

The weak cell this prompt targeted is gold `escalate`: **3/3** (v3b was 1/3). DEV-16388 and DEV-10805 are now handoffs instead of workarounds, and DEV-3585 is now `request_info`. The stronger handoff rule also over-fired: DEV-4338 (gold answer) and DEV-5604 / DEV-9342 (gold request_info) became `escalate`.

**Best overall prompt on this set is still v3b (12/15).** Use v3c only if the goal is not missing maintainer handoffs.

Gold evidence is not a RAG ranking. Do not treat contract-valid JSON as a verified Streamlit patch.
