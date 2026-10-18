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
| `rerank` | `` | the chat model for the reranking call only (it must have a price in pricing.json); empty uses models.chat |
| `embedding` | `text-embedding-3-small` | the embedding model's name; changing it means building the index again |
| `second_embedding` | `text-embedding-3-large` | the second embedding model of the dual_doc_slots retriever (its own index: paths.second_index_dir) |
| `base_url` | *unset* | the provider's URL; if unset, it is read from the OPENAI_BASE_URL environment variable |
| `api_key_env` | `OPENAI_API_KEY` | the name of the environment variable that holds the API key (never the key itself) |

### `[prompts]`

| Key | Default | What it does |
|---|---|---|
| `system` | `configs/prompts/answer_system.txt` | system prompt of the answer call |
| `output_format` | `configs/prompts/answer_output_format.txt` | JSON shape the answer must follow |
| `rewrite` | `configs/prompts/rewrite_system.txt` | system prompt of the query-rewriting call (QueryRewriter) |
| `rerank` | `configs/prompts/rerank_system.txt` | system prompt of the reranking call (LLMReranker) |
| `policy` | `configs/prompts/policy_v3_system.txt` | pipeline.generator = policy_v3: its system prompt (configs/prompts/policy_v3_system.txt, policy_v3b_system.txt or policy_v3c_system.txt) |
| `rerank_second` | `` | a second reranking prompt (for example configs/prompts/rerank_explain_system.txt): both rankings are made and merged by retrieval.rerank_merge; empty: one ranking |
| `scope` | `configs/prompts/scope_system.txt` | system prompt of the pre-retrieval scope classifier (scope gate on advanced) |
| `section_questions` | `configs/prompts/section_questions_system.txt` | system prompt that writes user-style problem descriptions for one documentation section |
| `topic_discover` | `configs/prompts/topic_discover_system.txt` | system prompt that grows a list of routing topics from the corpus, batch by batch (scripts/discover_topics.py) |
| `topic_refine` | `configs/prompts/topic_refine_system.txt` | system prompt that merges the discovered topics into the target range |
| `topic_classify` | `configs/prompts/topic_classify_system.txt` | system prompt that labels chunks and reports with a fixed list of Streamlit topics (class_doc_slots) |
| `label_cluster` | `configs/prompts/label_cluster_system.txt` | system prompt that names a topic cluster and lists its keywords (scripts/build_clusters.py) |
| `scope_rewrite` | `configs/prompts/scope_rewrite_system.txt` | system prompt of the combined scope check and concept queries, one call (pipeline.scope_gate = combined) |
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
| `second_index_dir` | `index/large` | the dual_doc_slots retriever's second index, built with models.second_embedding |
| `section_questions` | `data/questions` | symptom questions per documentation section, with their vectors, for the question_doc_slots retriever (scripts/build_section_questions.py) |
| `class_labels` | `data/classes/llm.jsonl` | topic labels per chunk for the class_doc_slots retriever (scripts/build_class_labels.py) |
| `clusters` | `data/clusters/clusters.jsonl` | topic clusters written by scripts/build_clusters.py (cluster_doc_slots retriever) |

### `[budget]`

| Key | Default | What it does |
|---|---|---|
| `ceiling_usd` | `5.0` | hard stop for live spending on this machine, all runs together |
| `reserve_usd` | `0.25` | kept unspent: a call is refused if it could eat into this |
| `max_live_calls` | `5000` | the most real model calls one command may make |

### `[chunking]`

| Key | Default | What it does |
|---|---|---|
| `method` | `fixed` | which chunker the indexing graph's chunk node runs: fixed \| structure \| token_structure \| langchain_rule |
| `max_tokens` | `512` | chunk size in tokens (the project's tokenizer); changing it means building the index again |
| `overlap_tokens` | `128` | tokens shared by consecutive chunks; must be smaller than max_tokens |
| `context_header` | `False` | embed each documentation chunk with its page title and heading path in front (the stored text is unchanged); needs its own index |
| `context_header_scope` | `start` | which headings the context header names: start (the path where the chunk starts) or all (that path plus every section heading inside the chunk) |
| `structure_min_chars` | `150` | structure chunker: sections shorter than this many characters are merged into the next |
| `structure_max_chars` | `2000` | structure chunker: sections longer than this many characters are split at block boundaries |
| `token_target` | `500` | token_structure chunker: the chunk size it aims for, in tokens |
| `token_soft_max` | `700` | token_structure chunker: the most tokens a chunk may grow to before it is cut |
| `token_min_section` | `150` | token_structure chunker: a section of at least this many tokens starts a new chunk; smaller ones are packed together |
| `langchain_chunk_chars` | `1600` | langchain_rule chunker: the most characters in a chunk (LangChain's recursive splitter, inside each heading section) |
| `langchain_overlap_chars` | `200` | langchain_rule chunker: characters shared by neighbouring chunks |

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
| `second_weight` | `0.5` | dual_doc_slots: how much the second embedding model's ranking counts against the main model's in the fusion |
| `report_views` | `none` | advanced: extra searches besides the queries, comma-separated: title (the report's title alone), code (its st.* commands, their parameter names and its error lines), or none; fused with the report's ranking at view_weight |
| `view_weight` | `0.5` | report_views: how much each view's ranking counts against the report's |
| `rerank_mode` | `llm` | llm (one reranking call) \| score (no call: the candidates' similarity to the report, for a question-found one its question's) \| rrf (no call: each candidate's rank among the candidates of its own source) |
| `rerank_merge` | `rrf` | prompts.rerank_second: merge the two rankings by rrf (a candidate's ranks in both) \| alternate (the best not yet taken of each in turn) \| split (the first ranking's best rerank_split, then the second's best not yet taken up to top_k) |
| `rerank_split` | `4` | rerank_merge = split: pieces taken from the first ranking; the rest of top_k comes from the second |
| `rerank_merge_k` | `10` | rerank_merge = rrf: the RRF constant |
| `bm25_extra` | `0` | question_doc_slots: add the best BM25 documentation chunks for the report that are not already candidates (exact identifiers); 0 adds none |
| `rerank_fuse_weight` | `0.0` | blend the reranker's order with each candidate's own search rank (among the candidates of its source) by weighted RRF; 0 uses the reranker's order alone |
| `rerank_fuse_k` | `10` | rerank_fuse_weight: the RRF constant (smaller makes the top ranks count more) |
| `rerank_versions` | `False` | reranking: state the reporter's Streamlit version and tag each candidate with the version it describes, and add a version rule to the prompt (configs/prompts/rerank_version_rule.txt) |
| `release_slots` | `0` | retrieval: add up to this many release notes for versions newer than the reported one, best matching the query, to the candidates (retrieval/release_slots.py); 0 = off |
| `rerank_show_question` | `False` | the reranker also sees, for a candidate found through a documentation section's symptom question (question_doc_slots), the question that matched the report |
| `rerank_show_sections` | `False` | the reranker also sees the headings of the sections each candidate contains, which its window may not show |
| `pool_mode` | `fuse` | advanced: how the reranker's candidates are chosen: fuse (every query's ranking merged with RRF, the best candidate_k kept) \| quota (fixed shares: the report's best dense and BM25 chunks, and the concept queries' best dense chunks) \| union (no fusion: the report's own candidates all kept, each concept query adding its best new ones) |
| `union_report` | `15` | pool_mode = union: the cleaned report's own candidates, all kept |
| `union_per_concept` | `12` | pool_mode = union: new candidates each concept query adds (its best ones not already in the pool) |
| `union_max` | `40` | pool_mode = union: the most candidates the reranker gets |
| `quota_report_dense` | `5` | pool_mode = quota: dense chunks from the cleaned report |
| `quota_report_bm25` | `5` | pool_mode = quota: keyword (BM25) chunks from the cleaned report |
| `quota_concepts_dense` | `10` | pool_mode = quota: dense chunks from the concept queries, taken in turns |
| `concept_weight` | `1.0` | with concept queries after the cleaned report (keep_report_query), how much each concept query's ranking counts in the fusion against the report's (1 = equal, plain RRF; below 1, the report's matches keep the lead) |
| `decompose` | `True` | advanced: write concept queries (the decompose step) besides the cleaned report; false searches with the cleaned report alone and reranks without the mechanisms list |
| `complete` | `none` | after reranking, repair each kept documentation chunk's edges: none \| neighbors (add the chunk before and after it) \| section (grow to the whole top-level sections it touches) \| subsection (grow each edge only to the innermost section around it) \| trim (drop a short leading fragment, so it starts at a heading) \| section_trim (trim the start, complete the last section) |
| `complete_max_tokens` | `900` | retrieval.complete: the most tokens a repaired chunk may grow to |
| `cluster_size` | `100` | cluster routing: about how many chunks scripts/build_clusters.py puts in each topic cluster (k-means on the chunk embeddings) |
| `cluster_method` | `kmeans` | cluster routing: how scripts/build_clusters.py groups the chunks: kmeans (the embeddings as they are) \| centered (each source type's average removed first, so clusters follow topic rather than writing style) \| doc_anchored (the documentation clustered alone, centered; every other chunk joins its nearest documentation cluster) |
| `cluster_top` | `2` | cluster routing: how many clusters each query is routed to |
| `cluster_route` | `keywords` | cluster routing: how a query is routed: keywords (its words among each cluster's keywords) \| embedding (its meaning against each cluster's description) \| centroid (its meaning against the average of each cluster's own chunks) |
| `cluster_mode` | `filter` | cluster routing: filter (search only the routed clusters) \| boost (search everything, the routed clusters' chunks ranked higher by retrieval.cluster_boost) |
| `question_filter_top` | `0` | question_doc_slots, Doc2Query--: keep a symptom question only if one of its this many nearest chunks in the index belongs to its own section; 0 keeps all |
| `question_chunk` | `overlap` | question_doc_slots: the chunk added for a matched section: overlap (the one holding most of the section) \| question (the one most similar to the matched question) |
| `question_snippet` | `report` | question_doc_slots: the window the reranker sees for a candidate found through a symptom question: report (chosen by the report's words) \| question (by the matched question's words) \| both |
| `question_order` | `append` | question_doc_slots: where the question-found candidates appear in the reranking list: append (after the base's) \| interleave (one of each in turn) \| shuffle (an order that says nothing about their source) |
| `question_keep` | `0` | question_doc_slots: at least this many of the kept pieces are question-found candidates (like rerank_keep_docs); 0 leaves it to the reranker |
| `question_sections` | `5` | question_doc_slots: sections found through their symptom questions that add a candidate beside the base retriever's |
| `class_route` | `llm` | class_doc_slots: how a query picks its topic classes: llm (one model call with prompts.topic_classify) \| centroid (nearest class centres, no call) |
| `class_top` | `2` | class_doc_slots: topic classes each query is routed to |
| `class_mode` | `filter` | class_doc_slots: filter (search only the routed classes) \| boost (search everything, the routed classes' chunks ranked higher by retrieval.class_boost) |
| `class_boost` | `0.05` | class_doc_slots with class_mode = boost: added to the similarity of chunks in the routed classes |
| `cluster_boost` | `0.05` | cluster routing with cluster_mode = boost: added to the similarity of chunks in the routed clusters |
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
| `scope_gate` | `off` | advanced only: the scope check before retrieval: llm (a model call), rules (free keyword rules), combined (the same call that writes the concept queries) or off (no check); advanced's profile turns it on (llm) |
| `answer_model` | `gpt-5.4-mini` | controlled_v2 answer step: the model the package's request is sent to (the package builds it for gpt-5.4-mini; gpt-5-mini and gpt-4.1-mini are cheaper) |
| `controlled_version` | `v2` | controlled answer step's package: v2 (as delivered), v3 (v2 + an investigation rule) or v4 (v2 + already-supplied facts, escalation with complete controls and length limits); generation/controlled_v3 and _v4 |
| `complete_questions` | `False` | controlled_v2 answer step: a question asks for every material gap the user can supply in one message, not only the first; the decision is unchanged (pipelines/stages/complete_questions.py) |
| `version_notes` | `False` | controlled_v2 answer step: add version and usage notes to each evidence piece (what version it describes, compared with the reporter's; how it may be used); pipelines/stages/version_notes.py |
| `generator` | `single_call` | advanced-based pipelines only: the answer step: single_call (the baseline's prompt and parser) \| controlled_v2 (one structured assessment by gpt-5.4-mini, the action routed by Python; supportdesk/generation/controlled_v2) \| policy_v3 (gpt-5.4-mini chooses the action itself, guided by prompts.policy) |

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
