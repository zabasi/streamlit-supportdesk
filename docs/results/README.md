# Results

The final results of the three main pipelines on the 15 development and 15 test cases, and how to
reproduce them. How each metric is defined is in `docs/evaluation.md`; every experiment and its result
is in `docs/pipelines.md`.

**How to read these numbers.** Each split has 15 cases (6 answerable, 6 that need a question, 3 that need
an escalation), so one decision is about 7 points. The designs were chosen on the development cases; the
test split was run once, at the end, on the design chosen beforehand. The development runs had the scope
check off; the test runs had it on (it let every test case through).

## Summary

| | `baseline` | `advanced` | `advanced-gpt5mini` |
|---|---:|---:|---:|
| Gold passages in the prompt, dev / test | 5.7% / 0% | 73.2% / 45.0% | as `advanced` |
| Correct decisions of 15, dev / test | 5 / 6 | 12 / 6 | 11 / 8 |
| Answers citing gold evidence, dev / test | 20% / 0% | 100% / 100% | 100% / 87.5% |
| LLM judge, dev: pass / partial / fail | not judged | 9 / 3 / 3 | 10 / 2 / 3 |
| LLM judge, test: pass / partial / fail | 1 / 2 / 12 | 5 / 2 / 8 | 6 / 3 / 6 |
| Cost per case | ~$0.0008 | ~$0.024 | ~$0.009 |

The final design brings the right evidence and grounds its answers on both splits. Its decision gain on
the development cases did not carry over to the test cases; telling a question from an escalation stayed
the main weakness (`advanced` escalated none of the three test cases that needed it).

## Retrieval

Gold passages are the evidence spans annotated for each case (81 on dev, 65 on test). A passage counts as
reaching the model when at least 80% of its text is in the evidence actually sent.

**The ladder on the development cases:** each step adds one change to the one before.

| Step | Gold passages in the prompt |
|---|---:|
| `baseline` | 5.7% |
| + cleaned report and fixed places for documentation | 39.6% |
| + reranking with a model | 59.1% |
| + each kept chunk grown to its whole sections | 61.8% |
| + 8 chunks, at least 5 of them documentation | 70.1% |
| + doc2query sections (`advanced`) | 73.2% |

**Test split:**

| | `baseline` | `advanced` without doc2query | `advanced` |
|---|---:|---:|---:|
| Gold passages in the prompt | 0% | 52.6% | 45.0% |
| Cases with at least one gold passage | 0% | 93.3% | 93.3% |
| Gold passage at rank 1 / MRR (passages) | 0% / 0 | 46.7% / 0.624 | 33.3% / 0.576 |
| Gold pages in the prompt / MRR (pages) | 0% / 0 | 59.4% / 0.680 | 61.7% / 0.733 |

doc2query brought the right pages higher on the test cases but fewer gold passages: the extra candidates
competed with good ones, and the reranker dropped more of them. The funnel shows where each gold passage
ended up:

| Where each gold passage ended | dev, without doc2query | dev, `advanced` | test, without doc2query | test, `advanced` |
|---|---:|---:|---:|---:|
| In the prompt | 54 | 58 | 34 | 30 |
| Dropped by the reranker | 6 | 13 | 4 | 17 |
| Its page reached, through another chunk | 3 | 6 | 10 | 8 |
| Its page never found | 18 | 4 | 17 | 10 |
| Total | 81 | 81 | 65 | 65 |

The funnel comes from a separate diagnostic run (`scripts/diagnose_passages.py`), so its "in the prompt"
counts can differ by a passage or two from the percentages above.

## Decisions

Rows are the decision the case calls for; columns the decision made.

**Development**

| `baseline` (5 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **1** | 4 | 1 |
| ask (6) | 3 | **2** | 1 |
| escalate (3) | 1 | 0 | **2** |

| `advanced` (12 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **6** | 0 | 0 |
| ask (6) | 1 | **4** | 1 |
| escalate (3) | 0 | 1 | **2** |

| `advanced-gpt5mini` (11 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **5** | 1 | 0 |
| ask (6) | 0 | **6** | 0 |
| escalate (3) | 1 | 2 | **0** |

**Test**

| `baseline` (6 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **2** | 2 | 2 |
| ask (6) | 3 | **2** | 1 |
| escalate (3) | 0 | 1 | **2** |

| `advanced` (6 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **4** | 1 | 0 |
| ask (6) | 1 | **2** | 3 |
| escalate (3) | 2 | 1 | **0** |

One answerable test case is not in the `advanced` table: its assessment supported no action, and the
answer package reports an error rather than guessing.

| `advanced-gpt5mini` (8 of 15) | answer | ask | escalate |
|---|---:|---:|---:|
| answer (6) | **4** | 2 | 0 |
| ask (6) | 2 | **4** | 0 |
| escalate (3) | 2 | 1 | **0** |

`advanced-gpt5mini` did not escalate in any run, on either split.

## The LLM judge

The judge (Claude Opus 5.5, a different model family from the answer model) read each case separately:
the report, the decision, the exact evidence the model saw, and the case's rubric
(`configs/prompts/judge_case_system.txt`, criteria in `docs/evaluation/judge_prompt.md`). On the test
split, a second judge run separately on `advanced` agreed on 13 of 15 cases; both disagreements were
between partial and fail.

What the passes were: every pass of `advanced` and `advanced-gpt5mini` on the test split is an answer
or a question; no escalation case passed. The baseline's single test pass is an escalation.

## Cost and time

Per case, averaged over the development cases (the scope check, when on, adds one small `gpt-4o-mini` call):

| | `baseline` | `advanced` | `advanced-gpt5mini` |
|---|---:|---:|---:|
| Model calls | 1 | 3 | 3 |
| Input tokens | 4,448 | 12,598 | 12,598 |
| Output tokens | 180 | 3,808 | 3,357 |
| Time to answer, mean / 95th percentile | 5.2 s / 7.1 s | 20.5 s / 31.0 s | ~30 s / 52 s |
| Cost | ~$0.0008 | ~$0.024 | ~$0.009 |

Almost all of `advanced`'s cost is the answer model's output, most of which is its hidden reasoning.
One-time costs: embedding the index (under one cent) and building the doc2query questions (about $0.02).

## Reproducing

The test split runs only with `--final`. Calls made before replay from the local cache for free, so
repeating a run on the same machine costs nothing.

```bash
# development
supportdesk eval --pipeline baseline          --split dev --mode live --run-id dev-baseline
supportdesk eval --pipeline advanced          --split dev --mode live --run-id dev-advanced
supportdesk eval --pipeline advanced-gpt5mini --split dev --mode live --run-id dev-advanced-gpt5mini
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --run-id dev-retrieval-advanced

# test, once
supportdesk eval --pipeline baseline          --split test --final --mode live --run-id test-baseline
supportdesk eval --pipeline advanced          --split test --final --mode live --run-id test-advanced
supportdesk eval --pipeline advanced-gpt5mini --split test --final --mode live --run-id test-advanced-gpt5mini

# tables and reports, written under artifacts/ (not committed)
python scripts/confusion_matrix.py dev-baseline dev-advanced dev-advanced-gpt5mini
supportdesk report dev-baseline dev-advanced dev-advanced-gpt5mini --output artifacts/reports/dev.html
supportdesk report test-baseline test-advanced test-advanced-gpt5mini --output artifacts/reports/test.html
python scripts/diagnose_passages.py --pipeline advanced --mode live --output artifacts/reports/funnel-advanced.md
```

`supportdesk report` writes an HTML comparison of runs: which implementation filled each stage, the
settings that differ, every metric and the per-case outcomes. Open it in a browser. Model outputs vary a
little from run to run, so a new run can differ by a case or two from the numbers above.
