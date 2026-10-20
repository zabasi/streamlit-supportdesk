# Grounded generation on hier-class retrieval

Generated 2026-10-16T14:19:22+00:00. Same annotated dev gold. Retrieval packets reused; only the answer call is new.

A grounded assistant is not one that cites; it is one that can be checked, and refuses when the check fails.

| method | answer | request_info | escalate | validation_failed | outcome accuracy | answerable | ambiguous | escalation | claim support | citation precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `advanced` | 7 | 5 | 3 | 0 | 0.4 | 0.5 | 0.3333 | 0.3333 | 1.0 | 1.0 |
| `llm_llm` | 3 | 5 | 6 | 1 | 0.2667 | 0.1667 | 0.1667 | 0.6667 | 0.75 | 1.0 |
| `advanced_rule_kmeans` | 3 | 3 | 6 | 3 | 0.2667 | 0.1667 | 0.1667 | 0.6667 | 0.6 | 1.0 |
| `advanced_llmchunk_llmfilter` | 7 | 3 | 4 | 1 | 0.2667 | 0.6667 | 0.0 | 0.0 | 0.8667 | 1.0 |

## How to read the numbers

- **answer / request_info / escalate**: the official support decision. Gold is the eval case category (`answerable` → answer, `ambiguous` → request_info, `escalation` → escalate).
- **outcome accuracy**: share of cases whose delivered decision is in `acceptable_outcomes`. Split by category in the next three columns.
- **confusion matrix** (below): rows are gold, columns are what the model delivered. Diagonal is correct. `validation_failed` is a blocked ungrounded `answer`, not a gold label.
- **claim support**: share of claims whose words appear in the *cited* packet passages. The usual bar is ≥ 0.90.
- **citation precision**: share of cited ids that are in this turn's packet. The usual bar is ≥ 0.95; outside packet must be 0.
- **answered after failure** must be 0: a rejected payload is never shown.
- **abstention F1**: should refuse when the packet has no gold source; should answer when it does. Precision + recall, because refusing everything or answering everything both look good on one number alone.
- **completeness vs gold spans**: of the reviewer-selected words, how many appear in the delivered answer. Retrieval can be fine and this still low if the model hedges.
- **answer when packet had gold**: of the cases where retrieval actually put gold in the packet, how many produced a delivered answer. This is the generation layer, not recall@6.

Links: [Pinecone RAG evaluation](https://www.pinecone.io/learn/series/rag/rag-evaluation/) · [RARR, Gao et al.](https://arxiv.org/abs/2305.14627) · [F1](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

## Confusion matrices (gold \ predicted)

Each cell is a case count. Read a row to see where that gold label went.

### `advanced`

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 3 | 2 | 1 | 6 |
| request_info | 3 | 2 | 1 | 6 |
| escalate | 1 | 1 | 1 | 3 |
| **total** | 7 | 5 | 3 | 15 |

### `llm_llm`

| gold \ predicted | answer | request_info | escalate | validation_failed | total |
|---|---:|---:|---:|---:|---:|
| answer | 1 | 3 | 2 | 0 | 6 |
| request_info | 2 | 1 | 2 | 1 | 6 |
| escalate | 0 | 1 | 2 | 0 | 3 |
| **total** | 3 | 5 | 6 | 1 | 15 |

### `advanced_rule_kmeans`

| gold \ predicted | answer | request_info | escalate | validation_failed | total |
|---|---:|---:|---:|---:|---:|
| answer | 1 | 2 | 1 | 2 | 6 |
| request_info | 2 | 1 | 3 | 0 | 6 |
| escalate | 0 | 0 | 2 | 1 | 3 |
| **total** | 3 | 3 | 6 | 3 | 15 |

### `advanced_llmchunk_llmfilter`

| gold \ predicted | answer | request_info | escalate | validation_failed | total |
|---|---:|---:|---:|---:|---:|
| answer | 4 | 1 | 1 | 0 | 6 |
| request_info | 3 | 0 | 3 | 0 | 6 |
| escalate | 0 | 2 | 0 | 1 | 3 |
| **total** | 7 | 3 | 4 | 1 | 15 |

