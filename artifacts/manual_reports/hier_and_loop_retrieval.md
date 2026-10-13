# Retrieval comparison: hierarchy and pooled loops

Live `eval-retrieval` on the 15 development cases (v06 corpus, all with gold evidence).
Models: `gpt-4o-mini` + `text-embedding-3-small`. Split: `dev` only. Not a full-answer `eval`.

**Best retrieval so far: `advanced_hier`.**

It is `advanced` with hierarchical search inside each query (section summaries first, then chunks).
On this set it is the only pipeline with hit@6 = 1.00 (every case has at least one gold source in the top 6) and the highest recall@6 / MRR / packet recall.

## Results

| Pipeline | recall@6 | hit@6 | MRR | gold in prompt | live cost |
|---|---|---|---|---|---|
| baseline | 0.03 | 0.07 | 0.01 | 0.03 | $0.0001 |
| hier_baseline | 0.41 | 0.67 | 0.15 | 0.41 | $0.0003 |
| retrieve_loop_best | 0.03 | 0.07 | 0.01 | 0.03 | $0.0069 |
| retrieve_loop_mixed | 0.03 | 0.07 | 0.01 | 0.03 | $0 (cached) |
| advanced | 0.84 | 0.93 | 0.82 | 0.84 | $0.0060 |
| **advanced_hier** | **0.88** | **1.00** | **0.84** | **0.88** | **$0.0031** |

Run ids: `base`, `hier-base`, `loop-best`, `loop-mixed`, `adv`, `adv-hier`.

The HTML comparison is `artifacts/manual_reports/hier_and_loop_retrieval.html`.

## What this means

- Hierarchy on the baseline (`hier_baseline`) lifts recall@6 from 0.03 to 0.41. It does not beat `advanced`.
- Hierarchy on `advanced` (`advanced_hier`) adds a further gain: recall@6 0.84 → 0.88, hit@6 0.93 → 1.00.
- `retrieve_loop_best` and `retrieve_loop_mixed` match baseline. They still search with the same dense retriever; they only change which of those chunks are kept after the loop.

This is retrieval only. Outcome accuracy needs `supportdesk eval`.
