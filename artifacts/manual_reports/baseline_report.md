# Baseline: development-set results

Run on 2026-10-10, live mode, 15 development cases (6 answerable, 6 ambiguous, 3 escalation).

| | |
|---|---|
| Pipeline | `baseline`: dense retrieval, top 6 chunks, one model call |
| Models | chat `gpt-4o-mini`, embedding `text-embedding-3-small` |
| Chunking | fixed, 512 tokens, 128 overlap; evidence budget 3,200 tokens |
| Query | the report as written (up to 4,000 characters) |
| Runs | `baseline-retrieval-dev-20261010-080126`, `baseline-dev-20261010-080355` |
| Config fingerprint | `7f4782abadd71892` |

## Results

| Area | Metric | Value |
|---|---|---|
| Retrieval | Hit@6 (any gold source in the top 6) | **0.07** (1 of 15 cases) |
| | Recall@1 / @3 / @6 / @10 | 0.00 / 0.00 / 0.03 / 0.03 |
| | MRR | 0.01 |
| | Gold evidence in the prompt | 0.03 |
| Decisions | Outcome accuracy | **0.33** (answerable 0.17, ambiguous 0.33, escalation 0.67) |
| Grounding | Citation violations / answers without citation | 0.00 / 0.00 |
| | Answers that cite gold evidence | 0.25 |
| Cost and time | Full run | $0.0116 (15 calls, about 4,400 input tokens each), mean 3.6 s per case |
| | Retrieval-only run | $0.0001 |

## What it shows

- **Retrieval is the bottleneck.** The gold evidence reached the model in 1 case out of 15.
- **The cause is a mismatch between query and evidence.** All gold evidence is documentation, but
  the query is the whole bug report (template headings, code, debug info), and the texts most
  similar to a bug report are other bug reports. In the traced case (DEV-5604, "text_area inside
  an expander doesn't load from session state"), all 6 retrieved chunks are issues and neither
  gold documentation page appears.
- **Generation behaves safely.** No invented citations, and the model rarely answers without
  evidence: answerable cases mostly end as questions or escalations, which is why answerable
  accuracy is low (0.17) while escalation is high (0.67).
- **It is cheap.** Retrieval experiments cost about $0.0001 each, so they can be run often.

With 15 cases, one case moves a rate by about 7 points; the retrieval result is clear regardless.

## Next steps

Each one goes into the `advanced` pipeline and is measured with `supportdesk eval-retrieval`
(Hit@6, Recall@6, gold in the prompt) before a full `eval`:

1. **Clean query:** title, error message and key lines, without the template boilerplate
   (`### Checklist`, `### Debug info`, ...). Cheapest, likely the largest gain.
2. **Documentation slots:** reserve part of the top 6 for documentation chunks, so similar issues
   cannot crowd it out.
3. **Query rewriting or HyDE:** turn the report into a documentation-style question or a short
   hypothetical documentation passage before searching.
4. **Hybrid search** (dense + BM25): exact names such as `st.data_editor` and error strings.
5. **Section chunking for documentation:** one chunk per section, as the gold passages are sections.

Also noticed: some issue versions are stored with a leading "- " (for example `- 1.29.0`), which
should be cleaned in the corpus metadata.

## Reproduce

```bash
supportdesk eval-retrieval --pipeline baseline --split dev --mode live
supportdesk eval --pipeline baseline --split dev --mode live
supportdesk snapshot --pipeline baseline --mode live
```