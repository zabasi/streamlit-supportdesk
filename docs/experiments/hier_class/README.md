# Hierarchical-class RAG (LangChain experiment)

Two ways to split the corpus, two ways to group the chunks into topic classes, retrieval only
inside the classes a query is about. Compared with `advanced` and `advanced_hier` on the **new
annotated development gold**. Then every chunker × classifier is also attached to the current
`advanced` retrieve path (the 3×3 ablation) so each factor can be read alone and together.

Stay on `--split dev`. Do not commit `.env`.

- What each method does: `docs/experiments/hier_class/methods.md`
- Worked numbers: `docs/experiments/hier_class/examples.md`
- Grounded generation (HW03 layer) + metrics: `docs/experiments/hier_class/generation.md`
- Generation policy v3 (support record; best prompt v3b is 12/15 on gold passages): `docs/experiments/policy_v3/README.md`
- HTML reports (open in a browser):
  - one change vs `advanced`: `artifacts/hier_class_rag/report_one_change.html`
  - two changes vs `advanced`: `artifacts/hier_class_rag/report_two_changes.html`
  - LLM filter model vs `advanced`: `artifacts/hier_class_rag/report_llmfilter.html`
  - everything in one table: `artifacts/hier_class_rag/report.html`

## Better design than “take the best of the four and rebuild advanced”

The four new methods change two things at once (chunker and classifier). The current systems
also add query rewrite and reranking. A raw scoreboard would confound those.

So the experiment isolates factors:

1. **The 2×2** uses the same simple retrieve path (class-routed dense search, documentation
   places kept, no rewrite, no rerank). The only differences are the chunker and the classifier.
   The report’s one-factor view averages the pair that share a chunker, then the pair that share
   a classifier.
2. **`advanced` and `advanced_hier`** run unchanged on the existing fixed-token index, with the
   new gold. They are the current systems, not members of the 2×2.
3. **The combo** does **not** rebuild `advanced` with the winning chunker. It takes the winning
   *classifier* (k-means or LLM topics) and routes `advanced`’s existing chunks. If recall
   moves, the cause is routing, not a new chunker.

4. **`advanced_llmchunk_llmfilter`** is the mixed test: full `advanced` stages on the
   LLM-chunked index with the LLM topic filter. Both factors change, so a gain or loss
   cannot be blamed on one alone.
5. **The advanced 3×3** then fills the rest of the grid: `{existing, rule, llmchunk}` ×
   `{none, kmeans, llmfilter}`, each on the same `advanced` retrieve path. Finished cells
   were reused under the old ids, then renamed.

## Name key

`llmchunk` is the **splitter**. `llmfilter` is the **topic router**. They are different
LLM calls (`llm_chunk_system.txt` vs `topic_classify_system.txt`).

| id | means |
|---|---|
| `advanced_llmchunk` | LLM **chunker only** (no class filter). Was `advanced_llm`. |
| `advanced_exist_llmfilter` | LLM **filter only** on the existing index. Was `advanced+llm`. |
| `advanced_llmchunk_llmfilter` | LLM chunker **and** LLM filter, both `gpt-4o-mini`. Was `advanced_llm_llm`. |
| `advanced_llmchunk_llmfilter_gpt5mini` | Same, but the **filter** is `gpt-5-mini`. Chunking stays `gpt-4o-mini`. |
| `advanced_exist_kmeans` | k-means filter on the existing index. Was `advanced+kmeans`. |
| `advanced_rule_llmfilter` | rule chunker + LLM filter. Was `advanced_rule_llm`. |

LLM chunking is applied to **documentation only**. Issues and comments stay on the rule chunker
(they are short and rarely heading-structured). A failed LLM split falls back to embedding
breakpoints, then to the rule chunker.

## Methods

| Name | Chunker | Classifier | Index | Retrieve |
|---|---|---|---|---|
| `rule_kmeans` | LangChain Markdown headers + recursive character split | spherical k-means (`doc_anchored`, 8 clusters) | new | class-routed dense, top 6 |
| `rule_llm` | same | LLM, 8 fixed Streamlit topics | new | same; query routed by one LLM call |
| `llm_kmeans` | LLM split points on documentation; rule on the rest | k-means as above | new | class-routed dense, top 6 |
| `llm_llm` | same LLM chunker | LLM topics | new | same |
| `advanced` | existing fixed 512/128 | none (current design) | existing | rewrite + `dense_doc_slots` + rerank |
| `advanced_hier` | existing fixed 512/128 | section-summary hierarchy | existing | same as `advanced`, searches inside the best sections |
| `advanced_exist_kmeans` | existing fixed 512/128 | k-means filter | existing | `advanced` stages, class-routed |
| `advanced_exist_llmfilter` | existing fixed 512/128 | LLM topic filter | existing | `advanced` stages, class-routed |
| `advanced_rule` | LangChain rule (705) | none | new | full `advanced` on the rule index |
| `advanced_rule_kmeans` | LangChain rule (705) | k-means filter | new | full `advanced`, class-routed |
| `advanced_rule_llmfilter` | LangChain rule (705) | LLM topic filter | new | full `advanced`, class-routed |
| `advanced_llmchunk` | LLM documentation splits (806) | none | new | full `advanced` on the LLM-chunk index |
| `advanced_llmchunk_kmeans` | LLM documentation splits (806) | k-means filter | new | full `advanced`, class-routed |
| `advanced_llmchunk_llmfilter` | LLM documentation splits (806), `gpt-4o-mini` | LLM topic filter, `gpt-4o-mini` | new | full `advanced`, class-routed |
| `advanced_llmchunk_llmfilter_gpt5mini` | same 806 chunks, `gpt-4o-mini` | LLM topic filter, `gpt-5-mini` | same | full `advanced`; only routing uses the stronger model |

`advanced_hier` is a **section-summary** hierarchy (summaries first, then chunks inside those
sections). The four new methods are a **class/cluster** hierarchy (label chunks, then search
only the matching classes). They are different ideas.

### Code

| Piece | Where |
|---|---|
| Gold from `selected_spans` | `src/supportdesk/eval/annotated.py` |
| LangChain rule + LLM chunkers | `src/supportdesk/data/langchain_chunkers.py` |
| Class router, LLM topics, k-means labels | `src/supportdesk/retrieval/class_routing.py` |
| Existing cluster k-means / `ClusterRoutedRetriever` | `src/supportdesk/retrieval/clusters.py` |
| `advanced` | `src/supportdesk/pipelines/advanced.py` |
| `advanced_hier` | `src/supportdesk/pipelines/hierarchical.py`, `src/supportdesk/retrieval/hierarchy.py` |
| Runner | `scripts/hier_class_rag.py` |
| Topic prompt | `configs/prompts/topic_classify_system.txt` |
| LLM-split prompt | `configs/prompts/llm_chunk_system.txt` |
| Metrics | `src/supportdesk/eval/metrics.py` (`evaluate_retrieval`) |

LLM topic labels: `session_state`, `widgets_keys`, `callbacks_forms`, `data_editor`, `caching`,
`multipage_navigation`, `expanders_layout`, `other`.

## Gold

The annotated development dump is not shipped with the repository. Put it at
`data/eval/development_source_annotated_v1/development.jsonl`, or pass its path with `--gold PATH`.

15 development cases. For each case, gold sources are the `sources[].source.source_id` values;
passage metrics use `selected_spans` (`source_id`, `start_char`, `end_char`). Official
`data/eval` category and acceptable outcomes are kept. The runner writes
`artifacts/hier_class_rag/annotated_dev_cases.jsonl`.

## How to run

From the project root, with the venv active and `SUPPORTDESK_CONFIG` **unset** (real corpus, not
the sample). `.env` must have `OPENAI_API_KEY` and `OPENAI_BASE_URL` (an OpenAI-compatible endpoint). Models stay
`gpt-4o-mini` and `text-embedding-3-small`.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
# langchain-text-splitters is in pyproject.toml; the venv already has it if you installed -e .

python scripts/hier_class_rag.py --mode live
```

Useful flags:

```powershell
python scripts/hier_class_rag.py --mode live --ablation --no-combo
python scripts/hier_class_rag.py --mode live --methods advanced_llmchunk_llmfilter --no-combo
python scripts/hier_class_rag.py --mode live --methods rule_kmeans,rule_llm,advanced,advanced_hier
python scripts/hier_class_rag.py --mode live --no-combo
python scripts/hier_class_rag.py --mode live --rebuild
python scripts/hier_class_rag.py --mode live --limit 3
python scripts/hier_class_rag.py --mode offline   # fake models; not a real score
```

`--ablation` runs the 3×3 of `advanced` × chunker × classifier and **reuses** any finished
`artifacts/hier_class_rag/runs/<name>/summary.json`. Use `--rebuild` to force a new live run.

Outputs (gitignored under `artifacts/`):

| File | What |
|---|---|
| `artifacts/hier_class_rag/report_one_change.html` | `advanced` vs each single change |
| `artifacts/hier_class_rag/report_two_changes.html` | `advanced` vs each chunker+filter pair |
| `artifacts/hier_class_rag/report_llmfilter.html` | `advanced` vs `gpt-4o-mini` / `gpt-5-mini` topic filters |
| `artifacts/hier_class_rag/report.html` | all runs in one table |
| `artifacts/hier_class_rag/comparison.md` | scoreboard + one-factor view |
| `artifacts/hier_class_rag/results.json` | full metrics |
| `artifacts/hier_class_rag/runs/<method>/` | the usual `summary.json` / `results.jsonl` |
| `artifacts/hier_class_rag/indexes/{rule,llm}/` | new indexes (reuse on the next run) |
| `artifacts/hier_class_rag/labels/*.jsonl` | chunk → class |

Cost order of magnitude on this corpus: a few cents for the new indexes and LLM labels; the
`advanced` / `advanced_hier` retrieval calls reuse the cache when the query rewrite and rerank
prompts already ran.

## Results

Live run 2026-10-15 on the 15 annotated development cases (`gpt-4o-mini`,
`text-embedding-3-small`). Full numbers: `artifacts/hier_class_rag/comparison.md`.

| method | recall@1 | recall@3 | recall@6 | hit@6 | MRR | passage_recall@6 | chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_kmeans` | 0.00 | 0.00 | 0.22 | 0.47 | 0.10 | 0.00 | 705 |
| `rule_llm` | 0.02 | 0.02 | 0.33 | 0.60 | 0.18 | 0.03 | 705 |
| `llm_kmeans` | 0.00 | 0.00 | 0.38 | 0.80 | 0.19 | 0.31 | 806 |
| `llm_llm` | 0.02 | 0.02 | 0.46 | 0.87 | 0.26 | 0.37 | 806 |
| `advanced` | 0.27 | 0.52 | 0.63 | **1.00** | 0.79 | 0.50 | 636 |
| `advanced_hier` | 0.27 | 0.49 | 0.61 | **1.00** | 0.79 | 0.51 | 636 |
| `advanced_exist_kmeans` | 0.30 | 0.52 | 0.62 | 0.93 | 0.82 | 0.60 | 636 |
| `advanced_exist_llmfilter` | 0.28 | 0.46 | 0.60 | 0.93 | 0.75 | 0.53 | 636 |
| `advanced_rule` | 0.19 | 0.40 | 0.61 | **1.00** | 0.69 | 0.00* | 705 |
| `advanced_rule_kmeans` | 0.30 | **0.56** | **0.66** | **1.00** | 0.87 | 0.02* | 705 |
| `advanced_rule_llmfilter` | 0.26 | 0.44 | 0.58 | **1.00** | 0.76 | 0.03* | 705 |
| `advanced_llmchunk` | 0.30 | 0.44 | 0.59 | 0.93 | 0.81 | 0.44 | 806 |
| `advanced_llmchunk_kmeans` | **0.32** | 0.47 | 0.54 | 0.87 | 0.83 | 0.39 | 806 |
| `advanced_llmchunk_llmfilter` | 0.31 | 0.48 | 0.62 | **1.00** | **0.88** | 0.53 | 806 |
| `advanced_llmchunk_llmfilter_gpt5mini` | 0.26 | 0.49 | 0.59 | **1.00** | 0.82 | 0.41 | 806 |

\* Passage recall is **not** comparable across chunkers. Gold spans were annotated on the
original corpus windows; remapped rule/LLM chunk ids rarely overlap those spans even when
the source id is correct. Compare cells with source recall / hit / MRR.

One-factor view on the **2×2 only** (mean recall@6 of the two simple methods that share the
factor; no rewrite, no rerank):

- LLM chunker 0.42 vs rule chunker 0.28
- LLM classifier 0.39 vs k-means 0.30

**Winner among the four:** `llm_llm`. It is still well below `advanced`, because the 2×2 has
no query rewrite and no rerank — that is expected.

### Advanced 3×3 (Δ recall@6 vs `advanced` 0.63)

| method | chunker | classifier | recall@6 | Δ | reused? |
|---|---|---|---:|---:|---|
| `advanced` | existing | none | 0.63 | 0 | yes |
| `advanced_exist_kmeans` | existing | kmeans filter | 0.62 | −0.01 | no (new) |
| `advanced_exist_llmfilter` | existing | LLM filter | 0.60 | −0.03 | yes |
| `advanced_rule` | rule chunker | none | 0.61 | −0.02 | no (new) |
| `advanced_rule_kmeans` | rule chunker | kmeans filter | **0.66** | **+0.02** | no (new) |
| `advanced_rule_llmfilter` | rule chunker | LLM filter | 0.58 | −0.06 | no (new) |
| `advanced_llmchunk` | LLM chunker | none | 0.59 | −0.04 | no (new) |
| `advanced_llmchunk_kmeans` | LLM chunker | kmeans filter | 0.54 | −0.09 | no (new) |
| `advanced_llmchunk_llmfilter` | LLM chunker | LLM filter | 0.62 | −0.01 | yes |

Mean recall@6 inside the 3×3 (each factor averaged over the other two):

- existing chunks 0.62 · rule chunks 0.61 · LLM chunks 0.59
- no classifier 0.61 · k-means 0.61 · LLM classifier 0.60

**What each factor does to `advanced`:**

- **Filter only, existing index.** Both filters lose source recall (`advanced_exist_kmeans`
  0.62, `advanced_exist_llmfilter` 0.60 vs 0.63) and drop hit@6 from 1.00 to 0.93.
- **Rule chunker only** (`advanced_rule`). Almost tied on recall@6 (0.61) and still hit@6
  1.00, but first-hit is worse (MRR 0.69 vs 0.79).
- **LLM chunker only** (`advanced_llmchunk`). Worse recall@6 (0.59) and hit@6 0.93. Finer
  docs do not help the full pipeline the way they helped the simple 2×2.
- **Together.** The only cell that **beats** `advanced` on recall@6 is
  `advanced_rule_kmeans` (0.66). `advanced_llmchunk_llmfilter` is the best first-hit
  (MRR 0.88) but not more complete. `advanced_llmchunk_kmeans` is the worst (−0.09).

`advanced_hier` stays essentially tied with unfiltered `advanced` (0.61 vs 0.63). On this
gold, rewrite + rerank still matter more than swapping the chunker or adding a class
filter. The one useful interaction is **rule chunks + k-means** under `advanced`.

### Stronger classifier (`gpt-5-mini`)

`gpt-5o-mini` is not a real model. The cheap GPT-5 tier is `gpt-5-mini`. This cell keeps
the **same 806 `gpt-4o-mini` chunks** and the same advanced rewrite/rerank (`gpt-4o-mini`).
Only chunk labels and query routing use `gpt-5-mini`.

| method | classifier | recall@6 | Δ vs `advanced_llmchunk_llmfilter` | hit@6 | MRR |
|---|---|---:|---:|---:|---:|
| `advanced_llmchunk_llmfilter` | `gpt-4o-mini` | 0.62 | 0 | 1.00 | 0.88 |
| `advanced_llmchunk_llmfilter_gpt5mini` | `gpt-5-mini` | 0.59 | −0.03 | 1.00 | 0.82 |

The stronger filter did **not** help. It is a little worse on completeness and first-hit.
A better topic model still applies a hard class cut; on this corpus that cut is the risk.

```powershell
python scripts/hier_class_rag.py --mode live --methods advanced_llmchunk_llmfilter_gpt5mini --no-combo
```

## Links (methods this experiment draws on)

- LangChain recursive splitter: <https://python.langchain.com/docs/how_to/recursive_text_splitter/>
- LangChain markdown header splitter: <https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/>
- LangChain semantic chunker (the fallback when the LLM split is unusable): <https://python.langchain.com/docs/how_to/semantic-chunker/>
- RAPTOR / tree (hierarchical) RAG: <https://arxiv.org/abs/2401.18059>
- Hierarchical index + recursive retrieval (LangChain blog): <https://blog.langchain.com/hierarchial-indices-and-retrieval-augmented-generation/>
- Cluster / topic routing as metadata pre-filtering: <https://www.pinecone.io/learn/chunking-strategies/>
- HyDE and query-side routing context: <https://arxiv.org/abs/2212.10496>
- Recall@k / MRR (the metrics already in `eval/metrics.py`): <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)>

Class notebooks (local, not links): HW03 LangChain contents and `Week03_ClassLab_student` — text
splitters and RAG routing are the pieces reused here.
