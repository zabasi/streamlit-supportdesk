# The answer step: controlled routing v2

Controlled routing v2 (packaged 2026-10-16, in `docs/generation/controlled-v2/`) is the answer step of
`advanced` and the pipelines built on it. The baseline's answer step is unchanged.

## How it works

One model call makes a **structured assessment** of the case: the user's main need, any information gaps
(and whether the user can supply them), an answer candidate with its evidence references, and whether an
internal investigation or human review is needed. The model never picks the action. **Python routes it**
from the assessment with fixed rules: a required human review escalates; a material gap the user can fill
asks for it (request_info, at most five follow-up questions); a supported answer candidate with no
material gap answers; an open internal investigation escalates. An assessment that establishes none of
these is an error, never a default escalation.

Model: gpt-5.4-mini, reasoning_effort medium, at most 8,000 completion tokens, a strict JSON schema.

## Where it lives

| Path | What it is |
| --- | --- |
| `src/supportdesk/generation/controlled_v2/` | the package, **unchanged** from the handoff (its frozen files are checksum-verified by `manifest.json`; excluded from lint for that reason) |
| `src/supportdesk/pipelines/stages/controlled.py` | the pipeline stage: gives the package a transport backed by the project's gateway (cache, cost ledger, budget guard), converts its result with the package's own `to_pipeline_answer`, and reports an unusable assessment as a per-case error with the call's usage |
| `docs/generation/controlled-v2/` | the complete handoff as delivered: README, results (a self-contained HTML report), the frozen source, the replay data, the package's own tests and `verify_handoff.py` |
| `tests/test_controlled_v2.py` | the runtime copy equals the delivered one byte for byte; the package's own 26 tests and verifier pass; the gateway, the stage and the wiring |

The gateway now also reports how an answer ended (`finish_reason`, `refusal`), live and from the cache;
the package needs that to reject cut-off or refused answers. The cost ledger records exactly the same
fields as before.

## Running it

    supportdesk eval --pipeline candidate-5-dense-rerank-section-controlled-v2 --split dev --mode live \
      --set pipeline.scope_gate=off --run-id ans-controlled-v2

or any advanced-based pipeline with `--set pipeline.generator=controlled_v2`. The baseline ignores the
setting.

## Cost: check the budget first

The package's own 15-case development run cost about **$0.26** (about $0.018 per case: 57,000 input and
49,000 output tokens, most of the output being the model's reasoning). That is about 20 times the
gpt-4o-mini answer step. Check the remaining budget before a live run.

## What has and has not been measured

- Measured by the package, on curated gold passages (not retrieval): 11 of 15 actions matched the
  development labels (73.3%): answer 5 of 6, request_info 3 of 6, escalate 3 of 3. This is action
  agreement, not answer quality.
- Measured on **retrieved** evidence (our pipeline's packets), development cases: the previous advanced
  12 of 15 (answer 6/6, request_info 5/6, escalate 1/3); the final advanced, with the two fixes below,
  12 of 15 (6/6, 4/6, 2/3).
- The test split, run once on the design chosen beforehand: `advanced` 6 of 15 (answer 4/6,
  request_info 2/6, escalate 0/3, and one case where the assessment supported no action), every answer
  citing gold evidence; `advanced-gpt5mini` 8 of 15 (4/6, 4/6, 0/3); the baseline 6 of 15 with no gold
  evidence at all. The decision gain on development did not carry over; the grounding gain did.
- Not measured: several conversation turns.
- The pipeline evaluates single turns: questions_used is 0 and no follow-up loop runs.
- The package's own limits: it can omit an information gap or a correction, and reference validation
  does not prove that the evidence entails the claim (see `controlled-v2/README.md`).

## Added since the handoff

The package's files are unchanged; everything below is in the stage (`pipelines/stages/controlled.py`) or
in new, separately versioned copies of the package. Each switch is a setting, off or unchanged by default,
so a pipeline uses it only when its profile says so.

| Setting | What it does | In `advanced` |
| --- | --- | --- |
| `pipeline.version_notes` | Before the model: each evidence piece gets two notes, computed in code from its type, version and date and the reporter's version (`pipelines/stages/version_notes.py`). Documentation is today's snapshot; a release is older or newer than the reporter's version; a release note shows a change, not a cause; another user's report is an observation, not documentation. Without a stated version, documentation gets no instruction to check one. | on |
| `pipeline.complete_questions` | After routing (the decision never changes): a question asks for every material gap the user can supply, with a numbered list, instead of only the first (`pipelines/stages/complete_questions.py`). | on |
| `pipeline.answer_model` | The model the package's request is sent to: `gpt-5.4-mini` (as built and validated), `gpt-5-mini` or `gpt-4.1-mini`. Each answer records which model wrote it. | `gpt-5.4-mini` |
| `pipeline.controlled_version` | `v2` (as delivered), `v3` or `v4`: copies of v2 with added prompt rules, each with its own `manifest.json` saying exactly what changed, and a test that its prompt holds no evaluation case. | `v2` |
| `generation.max_retries` | A dropped connection is retried (default once); a budget stop, refusal, cut-off answer or invalid assessment never is. | 1 |

What was tried and not adopted, on the development cases:

- **`gpt-5-mini`** as the answer model (`advanced-gpt5mini`): about 2.5 times cheaper, 11 of 15 with the
  two fixes, the best clarifying questions, but it never escalated in any of its runs, with any of three
  prompts. It is kept as the cheaper alternative; on the test split it made 8 correct decisions.
- **v3** (one rule: record an internal investigation when a reproduction meets unexplained behavior, and
  do not answer a partly explained symptom): with `gpt-5-mini`, two correct answers became questions and
  nothing was escalated.
- **v4** (three rules: facts the report already gives are not gaps; escalate when the report's controls
  are complete; word limits): with `gpt-5-mini`, no decision changed, but replies became about three
  quarters shorter (median 175 to 47 words).
