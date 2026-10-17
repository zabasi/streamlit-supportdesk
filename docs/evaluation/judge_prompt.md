# Case analysis: criteria, judge prompt and human comparison

This document publishes how each case of a run is analysed. Everything that can be measured is measured
by code; a judge (a model, then a person on a sample) scores only what needs reading. The system's own
output is never the only evidence that it is right: every judgement is made against the case's rubric
(`judge_reference`, written and reviewed for the evaluation and never shown to the system), and a sample
of the model judgements is compared with a person's.

## What is measured by code, and what is judged

| Criterion | Measured by code (no model) | Judged per case (rubric in `judge_reference`) |
| --- | --- | --- |
| **Evidence quality** | gold passages and gold documents in the prompt, Recall@k, MRR, on the labelled cases (`supportdesk eval-retrieval`, `supportdesk report`) | are the pieces in the prompt relevant; is every claim in the reply and the summary stated by an evidence text; is labelled evidence present |
| **Decision quality** | accuracy by category (answerable, needs clarification, needs escalation) and the confusion matrix (`supportdesk report`, `scripts/confusion_matrix.py`) | is the decision defensible and fitted to the available information; does a question target the required missing information; forbidden claims; version and uncertainty handling |
| **Operational success** | the stored decision is one of the case's acceptable decisions; errors and budget stops; in the web UI, the stored conversation status and question count | does the reply say what the stored decision is; does the routing follow the assessment; any action without confirmation, or a question already answered |
| **Cost and time** | calls, input and output tokens, cost, time per case and stages, baseline and final design on the same cases (`supportdesk report`); the one-off cost of building the index and the question bank | not judged |
| **Failure analysis** | which cases fail, and how (confusion matrix) | the likely cause of each failure, a concrete fix and its expected effect |

Accuracy is always reported by category, so that asking or escalating on every case cannot pass as
success: a design that never answers scores 0 of 6 on the answerable cases.

## The judge

- **Prompt:** `configs/prompts/judge_case_system.txt` (reproduced below). Its input is one case file from
  `scripts/export_cases.py`, which holds the report, the stored decision, every evidence piece with the
  exact text the model saw, the summaries, the controlled assessment and routing, and `judge_reference`.
- **Model:** not the model that produced the answers (`gpt-5.4-mini`), so that the system does not grade
  itself; a model of another family, at temperature 0 where it can be set. The judge model and its
  version are recorded with every judgement.
- **Scores:** 0 (fails), 1 (partly), 2 (fully), per item; a verdict of pass, partial or fail per case.
- **Limits:** a judge can misread; that is why its output is a second opinion beside the code's metrics,
  and why a sample is compared with a person's judgements.

## The human comparison

1. Choose the sample before reading any judgement: every failure of the final design, and as many passing
   cases, chosen at random by case ID (for example, 3 failures and 3 passes on the development cases).
2. A person reads each sampled case file exported with `--no-reference` first (blind), notes a decision
   and a verdict, then reads `judge_reference` and fills `docs/evaluation/human_review_form.md`, which has
   the judge's fields.
3. Compare field by field with `python scripts/judge_agreement.py <judge dir> <human dir>`: agreement on
   the verdict, on decision acceptability, on each 0-2 score (exact and within one point), and the list of
   disagreements. Report the agreement, and discuss every disagreement in the failure analysis.
4. Where they disagree, the person's judgement is the one reported, and the disagreement is a finding about
   the judge.

## Running it

```bash
python scripts/export_cases.py e2e-advanced-full                   # case files, with evidence text and rubric
python scripts/export_cases.py e2e-advanced-full --no-reference \
  --output-dir artifacts/runs/e2e-advanced-full/cases-blind        # for the blind human reading
```

Give the judge model `configs/prompts/judge_case_system.txt` as its system prompt and one case file as
the user message; save each JSON answer as `<judge dir>/<case_id>.json`.

## The prompt

See `configs/prompts/judge_case_system.txt`; it is the exact text the judge receives.
