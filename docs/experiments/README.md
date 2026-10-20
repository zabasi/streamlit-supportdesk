# Experiments

How to try an option, measure it, and compare it with the main pipelines. The write-ups of earlier
experiments are listed at the end; their results, with every other experiment's, are in
`docs/pipelines.md`.

## Ground rules

- **Change one thing at a time,** on the development cases (`--split dev`). The test split runs only
  with `--final`; tuning on it would make its numbers meaningless.
- **Expect noise.** Each split has 15 cases, so one decision is about 7 points, and one gold passage
  about 1.2 points of retrieval. Model outputs also vary a little from run to run. Treat a difference of a
  case or two as noise unless it holds in several settings.
- **Measure retrieval first.** `eval-retrieval` costs only embeddings and the reranking calls; a full
  `eval` adds the answer model, about $0.024 per case for `advanced`.
- **Repeats are free.** Every model call goes through a local cache, so rerunning the same requests on the
  same machine costs nothing.

## Run an existing experiment

About a hundred experiment pipelines are kept, grouped by what they test in
`configs/pipelines/experiments/` (`search`, `query`, `chunking`, `loop`, `candidates`, `answer`, `scope`,
`hier_class`). Each runs by name:

```bash
supportdesk pipelines                                                    # every pipeline and its description
supportdesk eval-retrieval --pipeline clean_doc_slots_rerank --split dev --run-id try-rerank
```

An experiment with different chunking needs its own index; commands say so and how to build it:

```bash
supportdesk index --pipeline clean_doc_slots_rerank_docs_prompt_structure --mode live
```

## Try an option without a new pipeline

Any setting can be changed for one run with `--set`, on top of a pipeline. The run records every setting,
so the report shows what differed. Some options worth trying on `advanced`:

**Retrieval**

```bash
supportdesk eval-retrieval --pipeline advanced --split dev --set retrieval.top_k=6 --run-id adv-k6
supportdesk eval-retrieval --pipeline advanced --split dev --set retrieval.question_sections=10 --run-id adv-q10
supportdesk eval-retrieval --pipeline advanced --split dev --set retrieval.doc_slots=5 --run-id adv-docs5
supportdesk eval-retrieval --pipeline advanced --split dev --set retrieval.rerank_keep_docs=4 --run-id adv-keep4
supportdesk eval-retrieval --pipeline advanced --split dev --set retrieval.complete=neighbors --run-id adv-neighbors
```

| Setting | In `advanced` | What it changes |
|---|---|---|
| `retrieval.top_k` | 8 | pieces the model sees |
| `retrieval.candidate_k` | 15 | candidates the reranker orders |
| `retrieval.doc_slots` | 10 | places for documentation among the candidates |
| `retrieval.question_sections` | 15 | documentation sections added through their doc2query questions |
| `retrieval.rerank_keep_docs` | 5 | documentation kept among the final pieces |
| `retrieval.complete` | `section` | how a kept chunk grows: `none`, `neighbors`, `section`, `subsection`, `trim`, `section_trim` |
| `context.max_tokens` | 4,300 | the evidence budget; raise it with `top_k`, or pieces are dropped |

**Answer step**

```bash
supportdesk eval --pipeline advanced --split dev --set pipeline.answer_model=gpt-5-mini --run-id adv-5mini
supportdesk eval --pipeline advanced --split dev --set pipeline.version_notes=false --run-id adv-no-notes
supportdesk eval --pipeline advanced --split dev --set pipeline.controlled_version=v4 --run-id adv-v4
```

| Setting | In `advanced` | Options |
|---|---|---|
| `pipeline.answer_model` | `gpt-5.4-mini` | `gpt-5.4-mini`, `gpt-5-mini`, `gpt-4.1-mini` |
| `pipeline.version_notes` | `true` | version and usage notes on each evidence piece |
| `pipeline.complete_questions` | `true` | one question message for every material gap |
| `pipeline.controlled_version` | `v2` | `v2` (as delivered), `v3` (an investigation rule), `v4` (three rules) |
| `pipeline.scope_gate` | `llm` | `llm`, `rules`, `combined`, `off` |

**Search engine.** `--retriever` swaps the retriever for one run (any of the 13 in `docs/pipelines.md`,
under "Options for each stage"). The run is
marked as an override everywhere, so it is never mistaken for the pipeline itself:

```bash
supportdesk eval-retrieval --pipeline baseline --split dev --retriever hybrid --run-id base-hybrid
```

**A few settings at once.** Put them in a small file and pass it with `--config`. `configs/overlays/` has
examples:

```bash
supportdesk eval-retrieval --pipeline baseline --split dev --config configs/overlays/top_k_10.toml --run-id base-k10
```

**Chunking** changes the index, so give the run its own index folder and build it first:

```bash
supportdesk index --mode live --set chunking.method=token_structure --set paths.index_dir=index_token_structure
supportdesk eval-retrieval --pipeline advanced --split dev --set chunking.method=token_structure \
  --set paths.index_dir=index_token_structure --run-id adv-token-structure
```

`supportdesk config --pipeline advanced --set ...` shows every value a run would use, and where it came
from, before you spend anything.

## Compare

```bash
supportdesk report dev-advanced adv-k6 adv-q10 --output artifacts/reports/compare.html
python scripts/confusion_matrix.py dev-advanced adv-5mini
python scripts/diagnose_passages.py --pipeline advanced --mode live --output artifacts/reports/funnel.md
```

- **`supportdesk report`** (HTML) shows which implementation filled each stage, the settings that differ,
  every metric and the per-case outcomes.
- **`confusion_matrix.py`** shows where decisions went, per category; the overall count hides which
  kind of case changed.
- **`diagnose_passages.py`** traces every gold passage: into the prompt, dropped by the reranker, its page
  reached through another chunk, or its page never found. It shows which stage to work on.
- **For answer quality,** export the cases (`python scripts/export_cases.py RUN_ID`) and judge them with
  the judge prompt (`docs/evaluation.md`).

Reports are written under `artifacts/`, which is not committed.

## Make it a pipeline

When an option is worth keeping as a named experiment:

1. Add a profile, `configs/pipelines/experiments/<group>/<name>.toml`, with only the settings that
   differ, and a comment saying what it tests. A profile may set only certain keys
   (`docs/configuration.md`); anything else is an error.
2. Register the name in `PIPELINES` (`src/supportdesk/registry.py`), usually with `build_advanced_scoped`.
3. Run it by name like any other. Names must be unique across all folders.

A new stage (code, not settings) is described in `CLAUDE.md` and `docs/pipelines.md`.

## Write-ups of earlier experiments

| File | What was tried |
|---|---|
| `cluster_routing.md` | Grouping the corpus into topic clusters and routing each report to its cluster |
| `complete_chunks.md` | Growing kept chunks to their neighbours or sections after reranking |
| `hier_class/` | Hierarchical-class retrieval and its own answer-generation runs |
| `policy_v3/` | An answer step where the model chooses the action itself |
