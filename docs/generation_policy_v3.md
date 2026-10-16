# Generation policy v3

A second generation layer, next to the HW03 grounded checker. The HW03 path
(`docs/hier_class_generation.md`) answers from a retrieval packet and **refuses**
when claim-support fails. Policy v3 writes a **support record**: it picks
`answer` / `request_info` / `escalate`, cites labelled evidence, and the controller
copies original report text so a quote cannot be silently reworded.

It was tested on 2026-10-16 as `generation_policy_v3` (not v4, not the production
agentic pipeline). Code lives in `src/supportdesk/generation/policy_v3.py`.
Prompts: original `configs/prompts/policy_v3_system.txt`, revision
`policy_v3b_system.txt` (best overall), `policy_v3c_system.txt` (best escalate).

Stay on `--split dev`. Do not commit `.env`.

```powershell
.\.venv\Scripts\python.exe scripts\policy_v3_generate.py
.\.venv\Scripts\python.exe -m pytest tests/test_policy_v3.py
```

Default is **offline**: rebuilds the 15 frozen requests and reprints the measured
confusion matrix. No API call. A live rerun is paid.

## What one turn does

```text
case (title, body, history, environment)
  + evidence passages (gold spans, or a retrieval packet adapted to the same slots)
  → labelled prompt: [R0][R1] report, [H1] history, [S1][S2] evidence
  → one JSON call
  → schema + reference-ID check (never repair)
  → controller copies original report blocks for the selected IDs
  → user reply = action.message (+ question if request_info)
```

The model is **not** allowed to paste observation quotes. It only returns IDs such
as `R1`. That is the v3 fix for a v2 failure: a capitalization change in an exact
quote used to fail validation and withhold the whole reply.

Official actions are the same three outcomes as eval gold:

| outcome | gold when the case is | meaning in this prompt |
|---|---|---|
| `answer` | `answerable` | shown code + evidence support a useful correction |
| `request_info` | `ambiguous` | one missing fact would change the next action |
| `escalate` | `escalation` | unresolved product/docs issue; human handoff |

## Measured numbers (gold evidence, not RAG)

Model: `gpt-5.4-mini` / medium / 8000 completion tokens. 15 annotated development
cases. **81 curated selected passages** (22 sources). Retrieval was not in the loop.

| | |
|---|---|
| Contract-valid | 15/15 |
| Action match | **9/15 (60%)** |
| Cost | about $0.22 |
| Content accuracy | not measured |
| Assistant review | 3 usable, 4 minor revision, 8 material revision |

Confusion (rows gold, columns predicted):

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 5 | 0 | 1 | 6 |
| request_info | 0 | 2 | 4 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 6 | 2 | 7 | 15 |

Compare with HW03 `advanced` on **retrieval packets** and `gpt-4o-mini`: 6/15 (0.40),
and that run *did* check claim support. These two scores are not interchangeable:

- Policy v3 9/15 used **gold passages** and a stronger model.
- HW03 6/15 used **retrieved chunks** and `gpt-4o-mini`.
- Policy v3 does not refuse an answer whose words fail lexical claim support; it
  only checks IDs, schema, and Python syntax as a warning.

Per-case actions: `experiments/generation_policy_v3/results/case-results.csv`.

## Prompt revisions on the same 15 new-dev cases

Same model (`gpt-5.4-mini`), same gold passages. Only the system prompt changed.

| prompt | file | correct | request_info | escalate | cost USD |
|---|---|---:|---:|---:|---:|
| original | `policy_v3_system.txt` | 9/15 | 1/6 | 2/3 | 0.256 |
| **v3b (best overall)** | `policy_v3b_system.txt` | **12/15** | **5/6** | 1/3 | 0.197 |
| v3c (best escalate) | `policy_v3c_system.txt` | 11/15 | 3/6 | **3/3** | 0.185 |

v3b chooses `request_info` → `answer` → `escalate` before writing a patch.
v3c keeps that order but forbids turning a reproduced product mismatch into
`answer` just because a workaround exists; it over-escalates some clarification
cases. HW03 `advanced` on retrieved packets remains 6/15.

Result files:

- `artifacts/hier_class_rag/generation/compare_prompt_v3b_vs_v3.md`
- `artifacts/hier_class_rag/generation/compare_policy_v3c_system.md`
- `artifacts/hier_class_rag/generation/compare_gpt54mini_vs_generation_folder.md`
- per-run `costs.csv` under `policy_v3_gpt54mini/`, `policy_v3b_gpt54mini/`, `policy_v3c_gpt54mini/`

## How this repo uses it

1. Frozen gold rows: `experiments/generation_policy_v3/inputs/model_inputs.jsonl`
   (the same 15 inputs as the measured run).
2. Retrieval adapter: `sources_from_packet` turns an `advanced` (or other) packet
   into the same `[S1]…` list. That is the RAG integration the original handoff
   asked for; it has **not** been live-scored yet.
3. Metrics: `PolicyTurn.to_dict()` feeds the same confusion-matrix code as HW03.

## What would make it better

The measured 9/15 is already stronger on **action labels** than HW03 `advanced`
(6/15), but it is not a finished generator. Highest-leverage changes, in order:

1. **Score it on retrieval packets, not gold passages.** The 9/15 used reviewer
   spans. `sources_from_packet` is in the repo; a live `--evidence packet` run is
   the first number that can be compared with HW03.
2. **Keep clarification and escalate from fighting each other.** v3b fixed
   most `request_info` misses (5/6) but dropped escalate to 1/3. v3c restored
   escalate (3/3) and lost two clarification cases. A prompt that holds both
   cells is still open.
3. **Keep the HW03 claim-support check on `answer`.** Policy v3 guarantees quote
   *origin*, not that the reply’s words are in `[S1]`. Attach `validate_grounded`
   as warnings or a withhold, without repairing JSON.
4. **Do not treat 15/15 contract-valid as quality.** Assistant review still wanted
   material revision on 8/15 replies. Structured JSON ≠ a working Streamlit patch.
5. **Stay on a cheap model for iteration.** The measured run was `gpt-5.4-mini`
   (~$0.22 / 15). Debug schema and request_info on `gpt-4o-mini` first.
6. **One sample per case is noisy.** The handoff itself saw a same-request action
   flip (DEV-9342 request_info → escalate). Repeat a few cases before trusting a
   prompt tweak.

Do not tune on held-out test labels. Do not send rubrics or gold outcomes in the
prompt.

Links: [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs) ·
[RARR](https://arxiv.org/abs/2305.14627) · [RAG evaluation](https://www.pinecone.io/learn/series/rag/rag-evaluation/)
