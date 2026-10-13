# Configuration

Every tunable value (models, chunk size, `top_k`, prompts, limits, thresholds, paths, budget) is a
setting. Nothing tunable is hardcoded in the code.

## Where values come from

Lowest to highest precedence; a later layer overrides an earlier one:

1. built-in defaults (`src/supportdesk/config.py`, mirrored with comments in `configs/default.toml`)
2. `configs/default.toml`: shared by every pipeline; edit it for values that apply everywhere
3. the pipeline profile `configs/pipelines/<pipeline.name>.toml`, chosen automatically
4. `$SUPPORTDESK_CONFIG`: one more file, e.g. `configs/sample.toml`
5. `--config FILE` (repeatable): experiments
6. environment variables `SUPPORTDESK__SECTION__KEY=value`, e.g. `SUPPORTDESK__RETRIEVAL__TOP_K=8`
7. `--set section.key=value` and the shortcuts (`--pipeline`, `--retriever`, `--mode`, `--data-dir`,
   `--index-dir`, `--artifacts-dir`, `--cases`, `--port`)

Relative paths resolve against the directory you run from, so run commands from the project root.

## See what is in effect

```bash
supportdesk config                        # every value and where it came from, absolute paths
supportdesk config --pipeline advanced    # the same for another pipeline
supportdesk config --describe             # plus what each setting does
supportdesk pipelines                     # the pipelines side by side
```

## Per-pipeline profiles

Each pipeline loads its own profile, so pipelines can differ while sharing everything else:

```
configs/pipelines/baseline.toml   pinned reference values (these equal the defaults the golden test pins)
configs/pipelines/advanced.toml   6 chunks like the baseline; the report plus up to 3 concept queries,
                                  15 candidates each (10 documentation), reranked by a model call that
                                  sees focused snippets and the concepts; overlapping kept chunks merged
```

- **What a profile may set:** `chunking`, `retrieval`, `context`, `prompts`, `pipeline.retriever`, `paths.index_dir`. Anything else in a profile is an error.
- **Why the rest is shared:** models, budget and generation settings stay in `default.toml` for every pipeline. That keeps comparisons fair and lets one client and cost ledger serve all pipelines.
- **Different chunking needs its own index.** If a pipeline uses different chunking or a different embedding model, give it its own `paths.index_dir` in its profile and build it with `supportdesk index --pipeline NAME`. Every command refuses an index built with other chunking or another embedding model, and says how to fix it.

## One knob for "how many chunks"

`retrieval.top_k` is how many chunks retrieval returns, and the prompt follows it. Leave
`context.max_chunks` unset unless you want the prompt to hold fewer; setting it above `top_k` is an
error. `context.max_tokens` is a separate budget: raise it along with `top_k`, or it caps the prompt.

## Experiments are config files

```toml
# configs/experiments/top_k_10.toml
[retrieval]
top_k = 10
[context]
max_tokens = 3000
```
```bash
supportdesk eval --pipeline baseline --config configs/experiments/top_k_10.toml --split dev --run-id baseline-k10
```

Commit the file and the experiment can be reproduced from the repository. Each run's `summary.json`
stores the full resolved configuration, where each value came from, and a fingerprint.
`supportdesk report` lists the settings that differ between runs.

## Models, keys and prices

- **Models:** `models.chat` and `models.embedding` name models your gateway offers. Changing the embedding model requires `supportdesk index`.
- **Endpoint:** `models.base_url` is the gateway URL. If it is unset, the OpenAI SDK reads `$OPENAI_BASE_URL`.
- **Key:** `models.api_key_env` names the environment variable that holds the key (`OPENAI_API_KEY` by default). The key itself never goes in a file, and a config file that contains something key-shaped is rejected.
- **Prices:** `configs/pricing.json` (`paths.pricing_path`) holds the price per model. Live mode refuses to start until `"verified": true`.

## Prompts

The answer prompt lives in files: `prompts.system` (`configs/prompts/answer_system.txt`) and
`prompts.output_format` (`configs/prompts/answer_output_format.txt`). To give a pipeline its own
prompt, copy the file, edit it, and point that pipeline's profile at it:

```toml
# configs/pipelines/advanced.toml
[prompts]
system = "configs/prompts/advanced_system.txt"
```

## Validation

Loading fails with a message that names the key, and all problems are reported together:

- **Typos:** `retrieval.top_K (cli): unknown key. Did you mean 'top_k'?`
- **Wrong types:** `retrieval.top_k (cli): cannot read 'abc' as int`
- **Contradictions:** `chunking.overlap_tokens (600) must be >= 0 and < chunking.max_tokens (512)`

## Adding a setting

1. Add a field, with a default and a help text, to its section in `src/supportdesk/config.py`.
2. Add it to `configs/default.toml` and run `python scripts/gen_config_docs.py`. Tests fail until both are done.
3. Read it from `settings` and pass it into the function or stage that needs it. Do not add a default argument that repeats it.
4. If it can contradict another setting, add a check in `Settings._problems`.
5. Add a test showing that changing it changes behavior.

## Reference

Defaults shown are the built-in ones. `*unset*` means the key is optional and normally left out.

### `[run]`

| Key | Default | What it does |
|---|---|---|
| `mode` | `offline` | offline: free fake models; replay: recorded responses only; live: real model calls, which cost money |

### `[models]`

| Key | Default | What it does |
|---|---|---|
| `chat` | `gpt-4o-mini` | the chat model's name at the provider; it must have a price in pricing.json |
| `embedding` | `text-embedding-3-small` | the embedding model's name; changing it means building the index again |
| `base_url` | *unset* | the provider's URL; if unset, it is read from the OPENAI_BASE_URL environment variable |
| `api_key_env` | `OPENAI_API_KEY` | the name of the environment variable that holds the API key (never the key itself) |

### `[prompts]`

| Key | Default | What it does |
|---|---|---|
| `system` | `configs/prompts/answer_system.txt` | system prompt of the answer call |
| `output_format` | `configs/prompts/answer_output_format.txt` | JSON shape the answer must follow |
| `rewrite` | `configs/prompts/rewrite_system.txt` | system prompt of the query-rewriting call (QueryRewriter) |
| `rerank` | `configs/prompts/rerank_system.txt` | system prompt of the reranking call (LLMReranker) |
| `scope` | `configs/prompts/scope_system.txt` | system prompt of the pre-retrieval scope classifier (scope gate on advanced) |
| `summarize_section` | `configs/prompts/summarize_section_system.txt` | system prompt that summarizes one documentation section (scripts/summarize_sections.py) |
| `judge_coverage` | `configs/prompts/judge_coverage_system.txt` | system prompt of the coverage judge (CorrectiveReranker) |
| `rewrite_missing` | `configs/prompts/rewrite_missing_system.txt` | system prompt of the query for a missing mechanism (CorrectiveReranker) |
| `extract` | `configs/prompts/extract_system.txt` | system prompt of the call that keeps the relevant parts of the chosen chunks (ExtractingReranker) |
| `judge` | `configs/prompts/judge_system.txt` | system prompt of the retrieve_loop sufficiency check (yes or no) |
| `loop_rewrite` | `configs/prompts/loop_rewrite_system.txt` | system prompt of the retrieve_loop query rewrite |
| `judge_scored` | `configs/prompts/judge_scored_system.txt` | system prompt of the retrieve_loop_best and retrieve_loop_mixed judge: yes or no, then a 0-10 score for every passage |

### `[paths]`

| Key | Default | What it does |
|---|---|---|
| `data_dir` | `data/processed` | the folder that holds corpus.jsonl (its format is described in data/corpus.py) |
| `index_dir` | `index` | where the index is written (qdrant/ vector store, chunks.jsonl, manifest.json) |
| `artifacts_dir` | `artifacts` | where runs, reports, the tracker and the cache are written |
| `cache_path` | `artifacts/cache.sqlite` | SQLite cache of model responses (makes re-runs free, enables replay) |
| `ledger_path` | `artifacts/cost_ledger.jsonl` | record of every live call's cost; it counts toward the budget across runs |
| `pricing_path` | `configs/pricing.json` | per-model prices used by the budget guard |
| `summaries` | `data/summaries/sections.jsonl` | section summaries written by scripts/summarize_sections.py (SummaryReranker) |

### `[budget]`

| Key | Default | What it does |
|---|---|---|
| `ceiling_usd` | `5.0` | hard stop for live spending on this machine, all runs together |
| `reserve_usd` | `0.25` | kept unspent: a call is refused if it could eat into this |
| `max_live_calls` | `5000` | the most real model calls one command may make |

### `[chunking]`

| Key | Default | What it does |
|---|---|---|
| `method` | `fixed` | which chunker the indexing graph's chunk node runs: fixed \| structure \| token_structure |
| `max_tokens` | `512` | chunk size in tokens (the project's tokenizer); changing it means building the index again |
| `overlap_tokens` | `128` | tokens shared by consecutive chunks; must be smaller than max_tokens |
| `context_header` | `False` | embed each documentation chunk with its page title and heading path in front (the stored text is unchanged); needs its own index |
| `context_header_scope` | `start` | which headings the context header names: start (the path where the chunk starts) or all (that path plus every section heading inside the chunk) |
| `structure_min_chars` | `150` | structure chunker: sections shorter than this many characters are merged into the next |
| `structure_max_chars` | `2000` | structure chunker: sections longer than this many characters are split at block boundaries |
| `token_target` | `500` | token_structure chunker: the chunk size it aims for, in tokens |
| `token_soft_max` | `700` | token_structure chunker: the most tokens a chunk may grow to before it is cut |
| `token_min_section` | `150` | token_structure chunker: a section of at least this many tokens starts a new chunk; smaller ones are packed together |

### `[retrieval]`

| Key | Default | What it does |
|---|---|---|
| `top_k` | `6` | how many chunks the final retrieval step returns |
| `candidate_k` | *unset* | candidates fetched before reranking; if unset, the same as top_k |
| `historical_filter` | `True` | hide material dated after a case was filed |
| `embed_batch_size` | `64` | texts per embedding request |
| `bm25_k1` | `1.5` | BM25: how quickly repeating a word stops raising the score |
| `bm25_b` | `0.75` | BM25: how much longer chunks are penalised, from 0 (not at all) to 1 |
| `hybrid_pool` | `30` | in hybrid search, how many candidates dense and BM25 search each contribute |
| `doc_slots` | `0` | places among the candidates kept for documentation chunks, used by the dense_doc_slots retriever; 0 keeps none |
| `max_queries` | `1` | most search queries a query stage may return; their results are merged by rank fusion |
| `keep_report_query` | `False` | query stages that write several queries (QueryRewriter, SentenceSplitQuery) also search with the cleaned report itself |
| `rerank_snippet_chars` | `600` | characters of each candidate shown to the model by the LLMReranker |
| `rerank_snippet` | `start` | which part of each candidate the LLMReranker shows: start (its first rerank_snippet_chars) \| focused (the window that best matches the report, with its section heading) |
| `rerank_keep_docs` | *unset* | documentation chunks the LLMReranker keeps in the final top_k; unset: retrieval.doc_slots |
| `min_chunks` | `2` | fewest chunks the GradedReranker keeps, even when fewer are graded relevant |
| `extract_mode` | `keep` | ExtractingReranker: the model lists the parts to keep, or the parts to drop |
| `extract_show_concepts` | `False` | ExtractingReranker: show the model the concept queries the query stage wrote |
| `extract_protect_top` | `0` | ExtractingReranker: never drop parts of this many top-ranked chunks |
| `rerank_show_concepts` | `False` | the LLMReranker also sees the concept queries the query stage wrote ("Mechanisms involved") |
| `corrective_turns` | `0` | CorrectiveReranker: after reranking, a judge checks the kept chunks; if they are not enough, rewrite the query, search again and rerank old and new candidates together, at most this many times (0 = off) |
| `corrective_judge` | `relevance` | CorrectiveReranker: relevance (yes or no: is there relevant evidence) or coverage (which concept queries no passage explains; one new query per missing one) |
| `corrective_max_missing` | `2` | CorrectiveReranker with the coverage judge: most missing mechanisms searched for in one turn |
| `summary_sections` | `0` | SummaryReranker: before reranking, add the chunks of this many documentation sections whose summaries best match the queries (0 = off) |
| `hier_sections` | `3` | hierarchical retrievers: how many documentation sections, chosen by comparing the query with the section summaries, the chunk search looks inside |
| `hier_places` | `3` | hierarchical retrievers: how many of the returned chunks come from inside those sections; the rest come from the ordinary search |
| `expand` | `none` | ExpandingReranker: add more chunks of the documentation pages among the candidates before reranking: none \| page (all their other chunks) \| section (only chunks sharing a section with a found chunk) \| branch (for structure chunks: the found chunk's ## parent and all its children, from the chunks' heading paths) |
| `expand_max_candidates` | `30` | ExpandingReranker: the most candidates after expansion |
| `section_candidates` | `30` | most sections and other candidates the SectionReranker ranks |
| `rerank_doc_places` | `0` | documentation places filled by the model's choice among the documentation candidates (DocPlacesReranker) |
| `bm25_slots` | `0` | places in the results filled by BM25 (keyword) search, the rest by dense search; used by the hybrid_split retriever |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant; higher values make rank differences count less |
| `max_loops` | `3` | retrieve-and-judge tries in the retrieve_loop pipeline; the last chunks are used once this many tries have run |

### `[context]`

| Key | Default | What it does |
|---|---|---|
| `max_chunks` | *unset* | most chunks shown to the model; if unset, retrieval.top_k (it cannot be more) |
| `max_tokens` | `3200` | token budget for all evidence in the prompt |
| `max_query_chars` | `4000` | longest search query built from a report |
| `max_body_chars` | `4000` | longest report body shown to the model |

### `[generation]`

| Key | Default | What it does |
|---|---|---|
| `max_output_tokens` | `900` | cap on tokens the model may generate per call |
| `max_retries` | `1` | retries after a provider error or a reply that is not valid JSON, and nothing else |
| `temperature` | `0.0` | sampling temperature; 0 gives the most repeatable answers |
| `timeout_seconds` | `60.0` | per-request timeout to the gateway |

### `[limits]`

| Key | Default | What it does |
|---|---|---|
| `max_input_chars` | `60000` | a request larger than this is refused before it is sent |

### `[pipeline]`

| Key | Default | What it does |
|---|---|---|
| `name` | `baseline` | registered pipeline to run (see registry.py) |
| `retriever` | `dense` | registered retriever: dense \| bm25 \| hybrid |
| `scope_gate` | `llm` | advanced only: the scope check before retrieval: llm (a model call), rules (free keyword rules) or off (no check) |

### `[eval]`

| Key | Default | What it does |
|---|---|---|
| `case_files` | `[PosixPath('data/eval')]` | a folder of evaluation files, or case .jsonl files; their source issues are kept out of the index |
| `test_fraction` | `0.5` | share of families assigned to the test split by `supportdesk split` |
| `split_seed` | `7` | seed for `supportdesk split` |
| `recall_ks` | `[1, 3, 6, 10]` | k values reported for Recall@k |
| `passage_found_fraction` | `0.8` | a gold passage counts as found when the retrieved chunks cover at least this fraction of it |
| `leakage_overlap_threshold` | `0.6` | warn when a chunk repeats this fraction of an evaluation case's text |
| `leakage_shingle_tokens` | `8` | length, in words, of the word sequences the leakage check compares |

### `[web]`

| Key | Default | What it does |
|---|---|---|
| `port` | `8765` | port of the local web page |
| `max_request_bytes` | `64000` | largest accepted request body |
| `max_title_chars` | `300` | longest title the UI accepts |
| `max_body_chars` | `8000` | longest description the UI accepts |
| `max_created_at_chars` | `40` | longest date the web page accepts |
| `max_history_items` | `20` | most earlier messages the UI accepts |
| `max_history_chars` | `2000` | longest earlier message |

### `[case]`

| Key | Default | What it does |
|---|---|---|
| `review` | `escalations` | which proposals wait for a maintainer before they are posted: escalations (answers and questions are posted directly) or all |
