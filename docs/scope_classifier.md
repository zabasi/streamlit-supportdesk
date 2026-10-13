# Pre-retrieval scope classifier

`scope_gate` is the LLM-based pre-retrieval gate. It classifies each request as
`IN_SCOPE` or `OUT_OF_SCOPE` before query construction and retrieval. The baseline pipeline is unchanged.

`scope_gate_rules` is kept only as a zero-cost rules baseline. It is intentionally conservative and is
not the recommended production gate; ambiguous general-Python questions can pass through it.

## Quick live check

```bash
supportdesk ask --pipeline scope_gate --mode live --body "How do I sort a Python list?"
supportdesk ask --pipeline scope_gate --mode live --body "Why does st.session_state reset when I change page?"
```

Expected: the first request is rejected before retrieval; the second enters retrieval.

## 30-case boundary benchmark

The development matrix lives in `data/eval/scope_classifier_dev.jsonl` and contains 15 `IN_SCOPE`
and 15 `OUT_OF_SCOPE` cases. It includes generic Python, competing frameworks, greetings, code-only
Streamlit, implicit Streamlit wording, historical versions, and ambiguous widget/state language.

Run the classifier benchmark without loading the vector index:

```bash
python scripts/eval_scope_classifier.py --mode live
```

Optionally compare a different configured chat model:

```bash
python scripts/eval_scope_classifier.py --mode live --model gpt-4o-mini
```

The report prints accuracy, IN_SCOPE precision/recall, false negatives, false positives, tokens,
model latency and measured cost. For this routing task, false negatives are the most important error:
a real Streamlit question should not be blocked before retrieval.

Use `--json` to save a machine-readable result:

```bash
python scripts/eval_scope_classifier.py --mode live --json > artifacts/scope_classifier_eval.json
```


## Advanced integration

The registry-facing `advanced` pipeline now runs `LLMScopeClassifier` immediately after intake.
`OUT_OF_SCOPE` stops before query rewriting, embedding retrieval, reranking, and generation.
`IN_SCOPE` continues through the existing advanced retrieval stages. The standalone
`scope_gate` and `scope_gate_rules` pipelines remain available for isolated experiments.

## Benchmark suites

- `data/eval/scope_classifier_dev.jsonl`: 30-case fast smoke suite.
- `data/eval/scope_classifier_benchmark.jsonl`: 120-case balanced benchmark (60 in-scope,
  60 out-of-scope) with explicit APIs, implicit Streamlit symptoms, code-only reports, historical
  versions, generic Python, competing frameworks, and ambiguous non-Streamlit UI/session cases.

Run the full suite with:

```bash
python3 scripts/eval_scope_classifier.py --mode live
```

Run the fast suite with:

```bash
python3 scripts/eval_scope_classifier.py --mode live --cases data/eval/scope_classifier_dev.jsonl
```
