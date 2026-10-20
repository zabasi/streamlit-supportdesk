# Generation policy v3 (tested 2026-10-16)

Exact experimental generation used for the 15-case development run. It is **not** the
HW03 grounded checker (`src/supportdesk/generation/grounded.py`) and **not** later
policy v4.

One LLM call returns a structured support record: observation reference IDs, workflow
findings, one official action (`answer` / `request_info` / `escalate`), and a
maintainer summary. After validation the controller copies the original report blocks
verbatim. No fuzzy matching and no output repair.

## What was measured (do not treat as RAG quality)

| | |
|---|---|
| Model | `gpt-5.4-mini` (returned `gpt-5.4-mini-2026-03-17`), reasoning medium, 8000 completion tokens |
| Evidence | 81 curated selected passages / 22 sources — **not** six retrieval chunks |
| Contract-valid | 15/15 |
| Action label match | **9/15 (60%)** |
| Content accuracy | not measured |
| Assistant review | 3 usable by inspection, 4 minor revision, 8 material revision |

Confusion (gold \ predicted) from that run:

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 5 | 0 | 1 | 6 |
| request_info | 0 | 2 | 4 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 6 | 2 | 7 | 15 |

Clarification was underused; escalation was overused. Direct application-code mistakes
were handled more reliably than version/product questions.

Prompt: `configs/prompts/policy_v3_system.txt`.
Code: `src/supportdesk/generation/policy_v3.py`.
Frozen gold-evidence rows: `inputs/model_inputs.jsonl`.
Measured snapshot: `results/metrics.json` and `results/case-results.csv`.

Run (no API by default):

```powershell
.\.venv\Scripts\python.exe scripts\policy_v3_generate.py
.\.venv\Scripts\python.exe -m pytest tests/test_policy_v3.py tests/test_grounded_generation.py
```
