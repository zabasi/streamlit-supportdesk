# Controlled routing v2 — limited diagnostic candidate

Preserves controlled-v1 routing, schema, limits, model/settings, input data and
composer. Adds a generic support-entailment/code-trace check to assessment prompting.
Discouraged code patterns and possible diagnostic experiments are not by themselves
established explanations of a historical symptom. No case IDs, expected labels or
case-specific reference maps are part of that prompt.

Answer assembly now includes every workflow correction, with report/source refs,
in the final user reply. The model is asked for a brief explanation and an integrated
finding list to limit duplication. No extra model/critic call is introduced. This
renderer can preserve an identified edit; it cannot invent an edit the model missed.

The model can still omit gaps or corrections. Reference validation does not prove
entailment. This adaptive diagnostic uses two studied clarification cases plus one
answerable case; it does not establish general or held-out accuracy. The offline
evidence audit and reference examples are evaluator artifacts, never model inputs.
Prior artifacts and the teammate handoff are not replaced.

Prompt structure consulted official OpenAI documentation:
https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4
