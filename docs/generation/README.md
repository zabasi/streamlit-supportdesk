# The answer step: controlled routing v2

The teammate's generation handoff (controlled routing v2, 2026-10-16) is merged as an alternative answer
step for advanced-based pipelines. The baseline's answer step is unchanged.

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

The teammate's 15-case development run cost about **$0.26** (about $0.018 per case: 57,000 input and
49,000 output tokens, most of the output being the model's reasoning). That is about 20 times the
gpt-4o-mini answer step. Check the remaining budget, and keep the reserve for the final test-split
evaluation, before a live run.

## What has and has not been measured

- Measured (by the teammate, on curated gold passages, not retrieval): 11 of 15 actions matched the
  development labels (73.3%): answer 5 of 6, request_info 3 of 6, escalate 3 of 3. This is action
  agreement, not answer quality.
- Not yet measured: on **retrieved** evidence (our pipeline's packets), on the test split, and over
  several conversation turns. The first live run of candidate 5 gives the first two.
- The pipeline evaluates single turns: questions_used is 0 and no follow-up loop runs.
- The package's own limits: it can omit an information gap or a correction, and reference validation
  does not prove that the evidence entails the claim (see `controlled-v2/README.md`).
