# Building and comparing pipelines

## Terms

| Term | Meaning | Where |
|---|---|---|
| **pipeline** | a complete recipe that answers a report; what `--pipeline` selects | `src/supportdesk/pipelines/baseline.py`, `advanced.py`, listed in `registry.py` |
| **stage** | one replaceable step of a pipeline | `src/supportdesk/pipelines/stages/` |
| **retriever** | the search stage: `dense`, `bm25` or `hybrid`; `--retriever` swaps it for one run | `src/supportdesk/retrieval/` |
| **profile** | a pipeline's own settings | `configs/pipelines/<name>.toml` for `baseline`, `advanced`, `advanced-gpt5mini`; `configs/pipelines/experiments/<group>/<name>.toml` for experiments |

## The stages

`StagedPipeline` (`pipelines/staged.py`) builds the support graph in LangGraph: an `intake` node,
then one node per stage below, in order. It owns everything that must not differ between pipelines:
timing, per-case usage, failure outcomes, secret redaction and the trace. Graph state is plain data
(see `docs/schemas.md`), converted to and from the stage objects at each node.

| Stage | Signature | Baseline | `advanced` (the final design) |
|---|---|---|---|
| `query` | `(case, env) -> list[str]` | `PlainQuery` | `CleanQuery`: the report without template sections, links and debug info |
| `retriever` | `.search(query, *, k, as_of) -> list[Hit]` | dense | `question_doc_slots`: dense with 10 of 15 places for documentation, plus the chunks of the 15 sections whose doc2query questions match the report |
| `reranker` | `(case, query, hits) -> list[Hit]` | `NoRerank` | one `gpt-4o-mini` call ranks the candidates; 8 kept (at least 5 documentation), each grown to its whole sections |
| `packet` | `(hits) -> Packet` | `DefaultPacketBuilder` | `DefaultPacketBuilder` (8 pieces, 4,300 tokens) |
| `generator` | `(case, env, packet) -> ParsedAnswer` | `SingleCallGenerator` | `ControlledGenerator`: a structured assessment by `gpt-5.4-mini`; Python chooses the action; version notes and complete questions |
| `validator` | `(case, packet, answer) -> ParsedAnswer` | `NoValidator` | `NoValidator` |
| `summarizer` | `(case, env, packet, answer) -> dict` | `build_summary` | `build_summary` |

Before these stages, `advanced` runs a scope check (`docs/scope_classifier.md`).

A stage is any callable with that signature; the contracts are in `pipelines/stages/protocols.py`.

## Options for each stage

Every option below is a setting, so it can be tried on any pipeline with `--set` (how to measure it is in
`docs/experiments/README.md`; every setting, with its default, is in `docs/configuration.md`).
**B** marks what `baseline` uses, **A** what `advanced` (and `advanced-gpt5mini`) uses.

**Chunking** (`chunking.method`; changes the index, so a different value needs its own index)

| Option | What it does | |
|---|---|---|
| `fixed` | windows of `chunking.max_tokens` tokens (512) overlapping by `chunking.overlap_tokens` (128) | B, A |
| `structure` | chunks follow headings and sections, never splitting a code block | |
| `token_structure` | sections packed into chunks of about a target size, still following headings | |
| `langchain_rule` | LangChain's Markdown header and recursive splitters | |

**Query** (the query stage)

| Option | What it does | |
|---|---|---|
| the report as written | title and body, up to `retrieval.max_query_chars` | B |
| the cleaned report | without template sections, checkboxes, links and debug info | A |
| `retrieval.decompose = true` | also up to `retrieval.max_queries` concept queries written by a model | |

**Retriever** (`pipeline.retriever`)

| Option | What it does | |
|---|---|---|
| `dense` | embedding similarity between the query and each chunk | B |
| `bm25` | keyword match; finds exact identifiers and error strings | |
| `hybrid` | dense and BM25 merged with Reciprocal Rank Fusion | |
| `hybrid_split` | dense and BM25 side by side: a fixed number of results from each | |
| `dense_doc_slots` | dense, with `retrieval.doc_slots` places kept for documentation | |
| `hybrid_doc_slots` | hybrid, with places kept for documentation | |
| `split_doc_slots` | dense and BM25 side by side, with places kept for documentation | |
| `question_doc_slots` | dense with places for documentation, plus the chunks of the `retrieval.question_sections` sections whose doc2query questions best match the report | A |
| `dual_doc_slots` | two indexes built with two embedding models, merged | |
| `cluster_doc_slots` | routes the query to its best topic clusters, then searches | |
| `class_doc_slots` | routes the query to topic classes (fixed or discovered), then searches | |
| `hierarchical`, `hierarchical_doc_slots` | some results taken from inside the best-matching sections | |

**Reranking**

| Option | What it does | |
|---|---|---|
| none | the search order is kept | B |
| one model call | orders `retrieval.candidate_k` candidates (15) and keeps `retrieval.top_k` (8), at least `retrieval.rerank_keep_docs` (5) of them documentation; it sees `retrieval.rerank_snippet_chars` characters of each, either the start or the best-matching window (`retrieval.rerank_snippet`: `start` or `focused`) | A (`focused`) |

**Completion after reranking** (`retrieval.complete`)

| Option | What it does | |
|---|---|---|
| `none` | each kept chunk as it is | B |
| `neighbors` | add the chunk before and after it | |
| `section` | grow it to the whole top-level sections it touches | A |
| `subsection` | grow each edge only to the innermost section around it | |
| `trim` | drop a short leading fragment, so it starts at a heading | |
| `section_trim` | trim the start and complete the last section | |

**Answer step** (`pipeline.generator`)

| Option | What it does | |
|---|---|---|
| `single_call` | one `gpt-4o-mini` call writes the answer and chooses the action | B |
| `controlled_v2` | one structured assessment by `pipeline.answer_model`; Python rules choose the action | A |
| `policy_v3` | one call where the model chooses the action from a policy prompt | |

With `controlled_v2`: `pipeline.answer_model` (`gpt-5.4-mini` in `advanced`, `gpt-5-mini` in
`advanced-gpt5mini`, or `gpt-4.1-mini`), `pipeline.controlled_version` (`v2`, `v3`, `v4`),
`pipeline.version_notes` and `pipeline.complete_questions` (both on in `advanced`).

**Scope check** (`pipeline.scope_gate`)

| Option | What it does | |
|---|---|---|
| `off` | no check | B |
| `llm` | one small model call decides whether the report is about Streamlit | A |
| `rules` | free keyword rules | |
| `combined` | the same call that writes the concept queries also decides | |

## Improving `advanced`

1. **Write the stage in its own file,** for example `src/supportdesk/pipelines/stages/rerank.py`. One file per stage keeps changes independent.

```python
from ...config import Settings
from ...llm.factory import Clients


class ModelReranker:
    def __init__(self, clients: Clients, settings: Settings):
        self.clients, self.settings = clients, settings   # every number comes from settings

    def __call__(self, case, query, hits):
        text, _ = self.clients.chat.complete([...], stage="rerank")   # counted on the ledger
        ...                                                            # parse defensively
        return reordered_hits
```

2. **Use it in `src/supportdesk/pipelines/advanced.py`:**

```python
stages = default_stages(retriever, clients, settings)
stages["reranker"] = ModelReranker(clients, settings)
```

3. **Give it settings in `configs/pipelines/advanced.toml`,** for example `retrieval.candidate_k`.
4. **Test it offline** (see `tests/test_stages.py`). The fake chat client can fail on purpose: `clients.chat.script = {1: "bad_json"}`.
5. **Compare it against the baseline** (below).

A new pipeline is one factory function like `build_advanced`, plus one line in
`registry.PIPELINES`, plus optionally `configs/pipelines/<name>.toml`. Its docstring's first line is the
description `supportdesk pipelines` shows.

## Seeing what a pipeline does

```bash
supportdesk snapshot --pipeline advanced                    # artifacts/snapshots/advanced.md
supportdesk snapshot --pipeline advanced --title "App stuck on loading" --body "Works locally only"
```

The snapshot lists each stage's implementation, described by the first line of its docstring, so
write one for every stage you add. It then shows what each stage produced on one case, and the
pipeline's latest evaluation results. In code, `pipeline.run(case, observe=callback)` reports each
step's output as it happens; the snapshot is built on that.

`docs/snapshots/` holds committed snapshots of the three main pipelines (`baseline.md`, `advanced.md`,
`advanced-gpt5mini.md`), each with real outputs for the same development case. `--all` snapshots the
main pipelines; an experiment is snapshotted by name. Regenerate them after a change (almost free: calls
already made by an evaluation run replay from the cache):

```bash
supportdesk snapshot --all --mode live --case-id DEV-16388 --output-dir docs/snapshots
```

## Rules for stages

- **Model calls go through `clients`,** with their own `stage="..."` name. That way the budget guard, the cache and the cost ledger see them. Never call a provider SDK directly.
- **Every number comes from `settings`.** If a value is missing, add a setting (see `docs/configuration.md`).
- **Do not catch gateway errors.** Provider and budget errors propagate, and the pipeline records them as `error` or `budget_stopped`. Generators can reuse `call_and_parse`, which retries only provider errors and non-JSON output.
- **Bugs crash the run on purpose,** so they are never mistaken for model failures.
- **Retrieved text is data, never instructions.**
- **Use only `case.input`,** never evaluation labels.

## Comparing

Change one thing per run, on the same dev cases:

```bash
supportdesk eval-retrieval --pipeline baseline --split dev --run-id base-dense
supportdesk eval-retrieval --pipeline baseline --retriever hybrid --split dev --run-id base-hybrid
supportdesk eval --pipeline advanced --split dev --run-id advanced-dev
supportdesk report base-dense base-hybrid advanced-dev
```

`base-hybrid` is an experiment, not the baseline: its retriever was overridden. The run says so
everywhere: `retriever: hybrid (OVERRIDE: baseline's own is dense)` on its first output line,
`retriever_override_of` in its `summary.json`, and a "retriever override" note under its name in
the report. The web page never overrides: it always runs each pipeline with its own retriever.

The report shows:
- which implementation filled each stage,
- the settings that differ between runs,
- every metric,
- the per-case outcomes.

`StagedPipeline.variant(reranker=...)` makes a one-stage variant in code, for quick experiments.

**The baseline is pinned.** `tests/test_golden_baseline.py` replays 15 scenarios and requires
identical output. If you change the baseline on purpose, regenerate with
`PYTHONPATH=src:. python -m tests.golden.generate` and review the diff. Prefer adding stages to
`advanced` over editing baseline stages.

## End-to-end designs

Each is a complete design (retrieval and answer step), run with `supportdesk eval` on the development
cases with the scope gate off. "Correct" counts decisions matching the label, out of 15.

| Pipeline | Retrieval | Answer step | Correct |
| --- | --- | --- | --- |
| `baseline` | dense, 6 chunks | one `gpt-4o-mini` call | 5 (6 with 8 chunks) |
| `candidate-1-dense-rerank-section` … `candidate-4-…` | reranked, whole sections; with symptom questions (2), two ranking prompts (3) or 6 chunks (4) | one `gpt-4o-mini` call | 5 to 7 |
| `candidate-5-dense-rerank-section-controlled-v2` | as candidate 1 | controlled routing v2 (`gpt-5.4-mini`) | 11 |
| `candidate-6-doc2query15-rerank-section-controlled-v2` | as candidate 1, plus symptom questions | controlled routing v2 | 12 (the previous `advanced`) |
| `candidate-7-…`, `candidate-8-…` | two ranking prompts; 6 chunks | controlled routing v2 | 10; 9 |
| `candidate-9-dense-rerank-section-policy-v3b` | as candidate 1 | the model chooses the action (policy v3b) | 10 |
| `advanced-version-notes` | as candidate 6 | controlled v2 + version notes | 11 with `gpt-5-mini` (a screening run) |
| `previous-design-gpt5mini` | as candidate 6 | controlled v2, `gpt-5-mini` | 10 |
| **`advanced`** | as candidate 6 | controlled v2 + version notes + complete questions | **12** (escalation 2 of 3) |
| `advanced-gpt5mini` | as candidate 6 | the same with `gpt-5-mini` | 11 (never escalates) |
| `advanced-v4-gpt5mini` | as candidate 6 | the same with the v4 prompt rules | 11 (replies about 75% shorter) |

On the test split (run once, with the scope gate on): `baseline` 6 of 15, `advanced` 6 (and one error),
`advanced-gpt5mini` 8. The decision gain did not carry over; grounding did: every `advanced` answer cites
gold evidence, against none for the baseline (gold passages in the prompt: 0% for the baseline, 45.0% for
`advanced`).

The answer-step settings are described in `docs/generation/README.md`.

## Experiments

Every experiment's profile is in `configs/pipelines/experiments/`, grouped by what it tested; the name is
the same wherever the file sits (`--pipeline NAME`):

| Folder | What its pipelines test |
|---|---|
| `search/` | the retrieval ladder: query cleaning, documentation slots, hybrid search, reranking prompts and budgets |
| `query/` | query reformulation: rewriting, splitting, decomposing the report into concept queries |
| `chunking/` | chunking: structure-aware and token-structure chunkers, fixed sizes, contextual headers, section summaries |
| `loop/` | retrieve again after a model judges the evidence |
| `candidates/` | the nine end-to-end designs compared before the final choice (table above) |
| `answer/` | answer-step variants: version notes alone, version-aware retrieval, the v4 rules, the previous design with `gpt-5-mini` |
| `scope/` | the scope gate on its own, with a model or with rules |
| `hier_class/` | hierarchical-class retrieval (`docs/experiments/hier_class/`) |

The early experiments below are each the baseline with exactly one change, in `pipelines/experiments.py`. Each shares the baseline's index, so measuring it costs only query
embeddings:

| Pipeline | The one change |
|---|---|
| `clean_query` | Query stage `CleanQuery`: the report without its template sections, checkboxes, links and debug info; from code, only the lines that call Streamlit or name an error |
| `doc_slots` | Retriever `dense_doc_slots` with `retrieval.doc_slots = 3`: at least 3 of the 6 results are documentation, still ordered by relevance |
| `hybrid_rrf` | Retriever `hybrid`: dense and BM25 each return their top `retrieval.hybrid_pool` results, merged by Reciprocal Rank Fusion; chunks both find rise to the top |
| `hybrid_split` | Retriever `hybrid_split` with `retrieval.bm25_slots = 3`: 3 results from dense, 3 different ones from BM25, without merging scores; the two lists take turns by rank, and if one runs short the other fills in |
| `rewrite_llm` | Query stage `QueryRewriter`: one model call turns the cleaned report into one documentation-style search query |
| `split_llm` | `QueryRewriter` with `retrieval.max_queries = 3`: the model writes up to 3 focused queries; each is searched and the results are merged by rank fusion. Compare with `rewrite_llm` |
| `split_sentences` | Query stage `SentenceSplitQuery` with `retrieval.max_queries = 4`: the cleaned report as the title, its Streamlit calls and its sentences, without a model; results merged by rank fusion. Compare with `clean_query` |
| `retrieve_loop` | After each search of 6 chunks, one model call answers yes or no. A no rewrites the query and searches again, at most `retrieval.max_loops` searches (its profile sets 2: the first search plus one repeat; the setting allows 1 to 3). Yes, or the last try, sends that search's 6 chunks to the generator. Same index and dense retriever as the baseline. No web search. |
| `retrieve_loop_best` | `retrieve_loop` that remembers. The judge (`prompts.judge_scored`) answers yes or no *and* scores every chunk it sees from 0 to 10, in the same call (no extra model call). Every chunk of every try goes into a pool with its best score. When the loop ends, the generator gets the 6 highest-scoring chunks of the whole pool, so good evidence from the first search survives a worse second search. A chunk the judge did not score gets its retrieval similarity × 10. Marked `found_by: "loop_best"`; the choice is saved in `trace.loop.selection` |
| `retrieve_loop_mixed` | The same pool, but the 6 returned are a third of the best, a third from the middle and a third of the worst-scoring chunks (2, 2, 2; what does not divide by three goes to the best first, then the middle: 7 gives 3, 2, 2). The middle ones are those around the median of the pool. Marked `found_by: "loop_best"`, `"loop_middle"` or `"loop_worst"`. Weak chunks are included on purpose, to test whether contrast helps or hurts |
| `hier_baseline` | Hierarchical retrieval on its own, on the baseline: the query is compared with the section summaries (`retrieval.hier_sections`, 3 best sections); the chunks inside them are ranked by their own similarity and 3 of the 6 results come from them (`retrieval.hier_places`), marked `found_by: "hierarchy"`; the other 3 come from the baseline's dense search, so issues and release notes can still be found. No model call besides the generator. Needs `data/summaries/sections.jsonl` |
| `advanced_hier` | `advanced` with the same hierarchy inside every one of its searches: of the 15 candidates of each query, 4 come from inside the 3 best-matching sections (retriever `hierarchical_doc_slots`, `hier_places = 4`); everything after retrieval is as in `advanced`. Compare it with `advanced`, and with the additive `…_summaries` variant below. A separate class-routing 3×3 on top of `advanced` is in `docs/experiments/hier_class/README.md` (`llmchunk` = LLM splitter, `llmfilter` = LLM topic filter; `advanced_llmchunk` is chunking only) |

Combinations put the strongest single change, the cleaned query, together with another:

| Pipeline | Built from |
|---|---|
| `clean_doc_slots` | `CleanQuery` + retriever `dense_doc_slots` with `retrieval.doc_slots = 3` |
| `clean_hybrid_split` | `CleanQuery` + retriever `hybrid_split` with `retrieval.bm25_slots = 3` |
| `clean_hybrid_doc_slots` | `CleanQuery` + retriever `hybrid_doc_slots` with `retrieval.doc_slots = 3`: like `clean_doc_slots`, but both lists come from hybrid search (dense and BM25 merged by rank fusion), the reserved places from documentation only |
| `clean_split_doc_slots` | `CleanQuery` + retriever `split_doc_slots` with `retrieval.doc_slots = 3` and `retrieval.bm25_slots = 3`: half the results from dense, half from BM25, at least 3 documentation; the reserved places are filled first, the two methods taking turns, then the rest the same way; scores are not merged. Compare with `clean_hybrid_doc_slots` (merged by rank fusion) |

Reranking on the two best retrieval combinations:

| Pipeline | Built from |
|---|---|
| `clean_doc_slots_rerank` | `clean_doc_slots` + `LLMReranker`: the retriever returns 15 candidates (`candidate_k = 15`), one model call orders them (`rerank_snippet_chars = 400`), the best 6 are kept |
| `clean_hybrid_doc_slots_rerank` | `clean_hybrid_doc_slots` + the same reranking |
| `clean_doc_slots_rerank_docs` | `clean_doc_slots` where the model chooses the documentation: 10 documentation candidates and the 3 best other chunks (`candidate_k = 13`, `doc_slots = 10`); `DocPlacesReranker` orders the documentation candidates with one model call and the best 3 fill the documentation places (`rerank_doc_places = 3`); the other 3 places are the best remaining by similarity; the 6 keep the similarity order |
| `clean_doc_slots_rerank_prompt` | `clean_doc_slots_rerank` with the reranking prompt `configs/prompts/rerank_docs_system.txt`, which ranks explanations first, earlier issues only when they state a cause or fix that applies, and every candidate |
| `clean_doc_slots_rerank_docs_prompt` | Both: 10 documentation candidates and the 5 best other chunks (`candidate_k = 15`, `doc_slots = 10`); the fixed prompt ranks all 15; the best 6 are kept in the model's order with at least 3 documentation chunks (`rerank_keep_docs = 3`) |
| `clean_hybrid_doc_slots_rerank_docs_prompt` | `clean_doc_slots_rerank_docs_prompt` (the earlier `advanced` design) with hybrid candidates: retriever `hybrid_doc_slots` (dense and BM25 merged by rank fusion) instead of dense |
| `clean_doc_slots_rerank_docs_prompt_wide` | `clean_doc_slots_rerank_docs_prompt` with a wider pool: 20 candidates, 15 of them documentation (`candidate_k = 20`, `doc_slots = 15`) |
| `…rerank_docs_prompt_focused` (dense and hybrid) | The reranker sees each candidate's best-matching 400-character window, with its section heading, instead of its first 400 characters (`retrieval.rerank_snippet = "focused"`). Report words are weighted by how rare they are among the candidates, so a word every candidate contains does not count |
| `…rerank_docs_prompt_long` (dense and hybrid) | The reranker sees the first 1,200 characters of each candidate instead of 400 (`retrieval.rerank_snippet_chars = 1200`) |
| `clean_doc_slots_rerank_docs_prompt_keep4`, `…_keep5` | Keep at least 4 or 5 documentation chunks in the final 6 after reranking (`rerank_keep_docs`), instead of 3. The reranker's call is unchanged |
| `clean_doc_slots_rerank_docs_prompt_top8` | Show the model 8 chunks instead of 6 (`top_k = 8`, `context.max_tokens = 4300`). The reranker's call is unchanged |
| `clean_doc_slots_rerank_docs_prompt_rewrite` | Search with the cleaned report **and** one model-written query naming the Streamlit concept behind the problem (`QueryRewriter`, `keep_report_query = true`, prompt `configs/prompts/rewrite_concepts_system.txt`); results merged by rank fusion, then reranked |
| `clean_doc_slots_rerank_docs_prompt_decompose` | The same with up to 3 model-written queries, one per concept involved (`max_queries = 3`) |
| `clean_doc_slots_rerank_docs_prompt_sentences` | Search with the cleaned report and up to 3 of its pieces, without a model (`SentenceSplitQuery`: the title, the Streamlit calls, the sentences) |
| `…_decompose_top8`, `…_decompose_focused`, `…_decompose_top8_focused` | `decompose` with 8 chunks (`top_k = 8`, budget 4,300 tokens, at least 3 documentation), with focused snippets for the reranker, or with both |
| `…_structure`, `…_token_structure` (on `clean_doc_slots_rerank_docs_prompt` and on `…_decompose`) | The same design on structure-aware chunks: `chunking.method = "structure"` (cut at headings of levels 1-3, never inside code blocks) or `"token_structure"` (structure boundaries packed to about 500 tokens). Other chunking needs its own index (`paths.index_dir = "index/structure"` or `"index/token_structure"`); build it once with `supportdesk index --pipeline <name>` |
| `…_structure_top8_keep4`, `…_token_structure_top8_keep4` (on both designs) | The chunking experiments showing the model 8 chunks (`top_k = 8`, budget 4,300 tokens), at least 4 of them documentation (`rerank_keep_docs = 4`); same index and reranking call |
| `…_ctx`, `…_structure_ctx`, `…_token_structure_ctx` and their `_ctx_top8_keep4` versions (on `clean_doc_slots_rerank_docs_prompt` and on `…_decompose`) | Contextual chunk headers (`chunking.context_header = true`): each documentation chunk is embedded with its page title and heading path in front ("Session State › Caveats and limitations"); the stored chunk text, its offsets and what the model reads are unchanged. Each has its own index (`index/<chunking>_ctx`); an index records whether it was built with headers and is refused by a pipeline configured otherwise |
| `…_decompose_structure_large`, `…_decompose_token_structure_large`, each also `_ctx` and `_ctx_all` | `decompose` on large structured chunks that pack small neighbouring sections: `structure` with `structure_min_chars = 2000`, `structure_max_chars = 4000` (median about 400 tokens), `token_structure` with `token_target = 600`, `token_soft_max = 900`, `token_min_section = 300` (median about 320); without a context header, with the usual one (`context_header_scope = "start"`), or with one that also names every section inside the chunk (`"all"`). Each has its own index |
| `…_decompose_structure_top12`, `…_top15`, and the same on `…_decompose_structure_ctx` | Small structure chunks, more of them within the same 3,200-token budget: 12 chunks from 25 candidates (18 documentation, at least 6 kept) or 15 from 30 (20 documentation, at least 8 kept). Same index as the parent |
| `…_decompose_sections` | Search exactly as `decompose` (fixed chunks), then the reranker ranks **whole page sections**: each candidate documentation chunk is replaced by the sections it overlaps (a `##` section with its `###` subsections; a section bigger than its chunk is cut to the chunk), at most 30 units (`retrieval.section_candidates`); the best 8 go to the model within 3,200 tokens |
| `…_decompose_trimmed` | `decompose` unchanged up to the reranker (same reranking call); each kept documentation chunk is then cut down to its one section that best matches the report |
| `…_decompose_merged` | `decompose` unchanged up to the reranker (same reranking call); kept chunks of one page that overlap or touch are merged into one span, so shared text appears once. Nothing is cut |
| `…_decompose_graded` | The reranker also grades each candidate as relevant, partly or not (prompt `configs/prompts/rerank_graded_system.txt`); chunks graded "not" are dropped, so the model gets as many chunks as are relevant, up to 6 and at least `retrieval.min_chunks` (2). The documentation minimum applies only to chunks not graded "not"; ungraded or unusable output behaves like `decompose` |
| `…_decompose_merged_extract` | `…_decompose_merged`, then one model call (prompt `configs/prompts/extract_system.txt`, stage `extract`) sees the full text of each part of the chosen chunks (documentation chunks split into their sections, clipped to the chunk; issues and comments whole) and keeps only the parts that help with the report; when unsure it keeps a part. Kept neighbouring parts are joined again; unusable output, or nothing kept, keeps everything |
| `…_decompose_merged_extract_drop` | The extraction revised: the model sees the concept queries `decompose` wrote (`extract_show_concepts`), is asked only which parts are clearly unrelated and may be removed (`extract_mode = "drop"`, prompt `configs/prompts/extract_drop_system.txt`), and parts of the top-ranked chunk are never removed (`extract_protect_top = 1`) |
| `…_decompose_merged_fixed320`, `…_fixed320_top9`, `…_fixed384`, `…_fixed384_top8` | `decompose_merged` on smaller fixed chunks (320/80 or 384/96 tokens), with 6 chunks (less text) or with 9 / 8 chunks (about the same text as 6 of 512) from 20 candidates (14 documentation); 3,200-token budget. Each size has its own index |
| `…_decompose_merged_expand_page`, `…_merged_expand_section` | `decompose_merged` where, before reranking, the documentation pages among the candidates add more of their chunks: all of them (`retrieval.expand = "page"`) or only those sharing a section with a found chunk (`"section"`), nearest first, at most 30 candidates (`retrieval.expand_max_candidates`). Added chunks are marked `found_by: "expand"`. Works on any index, for example the `3-large` one with `--set` |
| `…_decompose_merged_focused`, `…_merged_concepts`, `…_merged_focused_concepts` | `decompose_merged` where the reranker sees each candidate's best-matching window (`retrieval.rerank_snippet = "focused"`), the concept queries `decompose` wrote as a "Mechanisms involved" list (`retrieval.rerank_show_concepts`, prompt `configs/prompts/rerank_docs_concepts_system.txt`), or both |
| `…_decompose_structure_advanced`, `…_structure_advanced_branch` | The `advanced` design on structure chunks with context headers (`index/structure_ctx`): 12 sections from 25 candidates (18 documentation, at least 6 kept), within 3,200 tokens; the second adds, for each found section, its `##` parent and all its children before reranking (`retrieval.expand = "branch"`, from the chunks' heading paths; at most 40 candidates). On the development cases every gold passage lies within one `##` branch |
| `advanced --set retrieval.corrective_turns=1` | Corrective retrieval on top of `advanced` (the `CorrectiveReranker`): after reranking, a judge (`prompts.judge`) reads the 6 kept chunks; on "no", a model writes one new query (`prompts.loop_rewrite`), the retriever searches with it, and old and new candidates are reranked together. Chunks from the corrective search are marked `found_by: "corrective"`; each case's decisions are saved as `rerank_trace`, and reports count the cases with a corrective turn. `advanced` itself uses 0 turns (unchanged). The `retrieve_loop` pipeline is a separate loop on the baseline |
| `advanced --set retrieval.corrective_turns=1 --set retrieval.corrective_judge=coverage` | The corrective turn with a coverage judge: instead of "is there relevant evidence?", the judge (`prompts.judge_coverage`) lists which of `advanced`'s concept queries (the mechanisms involved) no kept passage explains; for each (at most `retrieval.corrective_max_missing`, 2) a model writes a new query in the documentation's words (`prompts.rewrite_missing`, told not to repeat the concept queries), then old and new candidates are reranked together |
| `…_decompose_merged_focused_concepts_summaries` | `advanced` with a section-summary level (hierarchical retrieval): a model writes a one- or two-sentence summary of each documentation section once (`python scripts/summarize_sections.py --mode live`, 146 sections, about $0.01, written to `data/summaries/sections.jsonl` with a manifest). Before reranking, the queries (the report and the concept queries) are compared with the summaries, and the fixed chunks of the 3 best-matching sections are added to the candidates (`retrieval.summary_sections`), marked `found_by: "summary"` |

With documentation places, the reranker keeps them (`retrieval.rerank_keep_docs`, or
`retrieval.doc_slots` when it is unset): after the model's order,
the best-ranked documentation below the cut replaces the lowest-ranked other chunks until the top
`top_k` hold at least `doc_slots` documentation chunks. The same applies when the model's output is
unusable. The stage is `LLMReranker` in `pipelines/stages/llm_rerank.py`; add it to any pipeline with
`stages["reranker"] = LLMReranker(clients, settings)` and `retrieval.candidate_k` above `top_k`.

The model-based stages fall back safely: unusable model output means the cleaned report (rewriting)
or the retriever's order (reranking), never a failed case. Their prompts are
`configs/prompts/rewrite_system.txt` and `configs/prompts/rerank_system.txt`, and their calls are
recorded under the stages `rewrite` and `rerank`.

```bash
supportdesk eval-retrieval --pipeline clean_query --split dev
supportdesk report <baseline run> <experiment run>
```

A change that helps moves into `advanced`. Retrieval runs also report the evidence the model
receives: its median size in tokens, the mean number of chunks, and its gold share (how much of the
evidence text lies inside a gold passage), so variants that show the model more or noisier text can be
compared fairly.

## The case workflow

`casework.py` extends the support graph for `supportdesk case`: after `summarize`, an answer or a
question is posted directly and an escalation waits for a maintainer (`case.review = escalations`),
or every proposal waits (`case.review = all`). Approvals, staleness and posting exactly once are
enforced there, in code; a new stage needs no changes to it.

## LangChain adapters

`src/supportdesk/integrations/langchain.py` exposes project parts as LangChain objects, for stages
that want a LangChain component:

| Adapter | Is a LangChain | Wraps |
|---|---|---|
| `ChatModel(client=clients.chat)` | chat model | the budget-guarded chat client |
| `EmbeddingModel(clients.embedding)` | embeddings | the embedding client |
| `Retriever(retriever=..., k=6)` | retriever (returns `Document`s with source metadata) | a project retriever |
| `pipeline_node(pipeline)` | LangGraph node: `state["case"]` to `state["result"]` | a pipeline |

Use these instead of LangChain's own model classes, so every model call stays inside the budget
guard and on the ledger.
