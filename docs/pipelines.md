# Building and comparing pipelines

## Terms

| Term | Meaning | Where |
|---|---|---|
| **pipeline** | a complete recipe that answers a report; what `--pipeline` selects | `src/supportdesk/pipelines/baseline.py`, `advanced.py`, listed in `registry.py` |
| **stage** | one replaceable step of a pipeline | `src/supportdesk/pipelines/stages/` |
| **retriever** | the search stage: `dense`, `bm25` or `hybrid`; `--retriever` swaps it for one run | `src/supportdesk/retrieval/` |
| **profile** | a pipeline's own settings | `configs/pipelines/<name>.toml` |

## The stages

`StagedPipeline` (`pipelines/staged.py`) builds the support graph in LangGraph: an `intake` node,
then one node per stage below, in order. It owns everything that must not differ between pipelines:
timing, per-case usage, failure outcomes, secret redaction and the trace. Graph state is plain data
(see `docs/schemas.md`), converted to and from the stage objects at each node.

| Stage | Signature | Baseline | Ideas for `advanced` |
|---|---|---|---|
| `query` | `(case, env) -> list[str]` | `PlainQuery` | rewrite with a model, extract the error message, several queries (fused with RRF) |
| `retriever` | `.search(query, *, k, as_of) -> list[Hit]` | dense | `hybrid`, version-aware filters, graph retrieval |
| `reranker` | `(case, query, hits) -> list[Hit]` | `NoRerank` | cross-encoder or model rerank (fetch `retrieval.candidate_k`, keep `top_k`) |
| `packet` | `(hits) -> Packet` | `DefaultPacketBuilder` | diversity, drop superseded sources |
| `generator` | `(case, env, packet) -> ParsedAnswer` | `SingleCallGenerator` | separate "ask" and "answer" decisions |
| `validator` | `(case, packet, answer) -> ParsedAnswer` | `NoValidator` | claim-support check, downgrade to `escalate` |
| `summarizer` | `(case, env, packet, answer) -> dict` | `build_summary` | richer maintainer summary |

A stage is any callable with that signature; the contracts are in `pipelines/stages/protocols.py`.

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

## Experiments

An experiment is the baseline with exactly one change, in `pipelines/experiments.py` with its own
profile in `configs/pipelines/`. Each shares the baseline's index, so measuring it costs only query
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
| `retrieve_loop` | After each search of 6 chunks, one model call answers yes or no. A no rewrites the query and searches again, at most `retrieval.max_loops` (3) times. Yes, or the third try, sends those 6 chunks to the generator. Same index and dense retriever as the baseline. No web search. |

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

A change that helps moves into `advanced`.

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
