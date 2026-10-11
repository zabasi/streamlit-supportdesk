# Evaluating

How to measure a pipeline, what every number means, and how an evaluation can quietly mislead.
Commands assume `export SUPPORTDESK_CONFIG=configs/sample.toml` for the sample data; unset it for
real data.

## The loop

```bash
supportdesk index                                                            # 1. index without the evaluation issues
supportdesk check-cases                                                      # 2. validate the cases and the index
supportdesk eval-retrieval --pipeline baseline --split dev                   # 3. cheap: retrieval only
supportdesk eval --pipeline baseline --split dev --run-id baseline-dev       # 4. full run
supportdesk eval --pipeline advanced --split dev --run-id advanced-dev
supportdesk report baseline-dev advanced-dev                                 # 5. compare
supportdesk eval --pipeline advanced --split test --final                    # 6. once, at the very end
```

- **Pipeline:** every run names it explicitly, and the first line of its output repeats it.
- **Tune on dev only.** The test split is refused without `--final`.
- **Leakage guard:** `eval` and `eval-retrieval` refuse to run if the index contains an evaluation issue.

## Two kinds of evaluation

| Command | Runs | Measures | Live cost |
|---|---|---|---|
| `eval-retrieval` | query, search, rerank, prompt packet | recall@k, hit@k, MRR, packet recall | query embeddings only (cached) |
| `eval` | the whole pipeline | the above, plus outcome accuracy, citations, errors, cost | one model call per case, plus embeddings |

Retrieval decides what the model can possibly say. Tune it with `eval-retrieval` first, and run
`eval` once retrieval looks right. With a small API budget, that is the difference between many
experiments and a few.

## Evaluation cases

`eval.case_files` lists a folder of evaluation files (default `data/eval`; per split: inputs,
rubrics, history variants, source-usage tasks, multi-turn scenarios; read by `eval/dataset.py`), or
`.jsonl` files with one case per line in the format below (see `data/eval/sample_cases.jsonl`). Both
become the same `EvalCase` (`docs/schemas.md`).

| Field | Meaning |
|---|---|
| `case_id`, `family_id` | `family_id` groups duplicate reports; a family never spans dev and test (`supportdesk split` assigns splits by family) |
| `split` | `dev` or `test` |
| `category` | `answerable`, `ambiguous` (more information needed) or `escalation` (needs a maintainer) |
| `input` | what the assistant sees: `title`, `body`, `created_at`, `history`. Nothing from later comments |
| `gold_evidence` | source ids a good answer rests on; may be empty |
| `acceptable_outcomes` | which of `answer`, `request_info`, `escalate` are defensible |
| `source_issue_number`, `duplicate_issue_numbers` | the real issues the case came from; kept out of the index |
| `missing_info` | what a good reply should ask for; recorded for people, not scored yet |

### Labelling gold evidence

Read how the real issue was resolved: the documentation page the maintainer linked, the earlier issue
marked as a duplicate, the release that fixed it. Use those items' `source_id`s.

- **Comments count as their issue.** Finding the comment satisfies a label on its parent issue.
- **Labels are per source, not per chunk,** so you can change chunking without relabelling.
- **A case may have no gold evidence.** It is then left out of recall, never counted as zero.

`check-cases` and every evaluation flag labels that can never be retrieved, because they cap recall
for reasons unrelated to the system:
- **Not in the index:** a typo, or an evaluation issue that was correctly excluded.
- **Dated after the case:** the historical filter hides anything newer than the case. For example, the release that fixed a bug cannot be gold evidence for the report of that bug.

## What the numbers mean

| Metric | Definition |
|---|---|
| `recall@k` | per case, the share of its gold sources in the top k; averaged over cases with gold |
| `hit@k` | share of cases with at least one gold source in the top k |
| `MRR` | mean of 1 / rank of the first gold source (0 if none) |
| `packet recall` | recall over the chunks that actually reach the prompt, after the chunk and token limits |
| `outcome accuracy` | the outcome is one of the case's `acceptable_outcomes`, overall **and per category** |
| `answers cite gold evidence` | among answers, the share citing at least one gold source |
| `citation violations` | citing something not retrieved, answering without a citation, uncited facts |
| `errors`, `budget stops` | runs that failed or were stopped by the budget |
| `cost`, `calls`, `cache hits`, `tokens`, `time` | what the run used; replayed cost is shown apart from real spend |

Per category matters: a system that escalates everything scores well on `escalation` and badly on
`answerable`. Look at all three.

| You see | Likely cause | Try |
|---|---|---|
| low recall@k | retrieval cannot find the evidence | chunking, the query, `--retriever hybrid`, a larger `top_k` |
| high recall, lower packet recall | the prompt limits drop evidence | `context.max_tokens`, `retrieval.top_k` |
| high packet recall, low `answerable` accuracy | the model has the evidence but misuses it | the prompt, the generator |
| `answerable` fine, `ambiguous` poor | it answers when it should ask | missing-information reasoning |
| many citation violations | ungrounded answers | a validator stage |

## Small samples

With 15 cases per split, one case is about 6.7 percentage points. A one- or two-case difference is
noise. Report counts next to percentages, and read the per-case table in the report.

## Where results go

`artifacts/runs/<run_id>/` contains:
- **`results.jsonl`:** one record per case.
- **`summary.json`:** metrics, the pipeline's stages, the full configuration and its fingerprint, the index manifest and any warnings.
- **`cost_ledger.csv`:** every model call.

## Not built yet

- **LLM judge:** an LLM judge, and its comparison with human judgements.
- **Claim-level support:** today, citations are only checked to point at retrieved chunks.
- **`missing_info`:** scoring whether the reply asked for the right missing information.
- **Multi-turn scenarios:** running the staged scenarios against the case workflow.

## Where gold passages are lost

`scripts/diagnose_passages.py` traces every gold passage through a pipeline's retrieval and reports
where it ends up: in the prompt; among the candidates but dropped by the reranker; its page among
the candidates but through another chunk; or its page not among the candidates at all. For each
passage it also gives the rank of the chunk that holds it in a search over all documentation, which
shows whether a wider documentation pool could reach it.

```bash
python scripts/diagnose_passages.py --mode live --output docs/reports/passage-funnel.md
python scripts/diagnose_passages.py --pipeline clean_doc_slots --mode live
```

It changes nothing; after an `eval-retrieval` run of the same pipeline its embeddings and model calls
come from the cache.
