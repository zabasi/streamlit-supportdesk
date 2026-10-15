# Hierarchical-class RAG (LangChain experiment)

Two ways to split the corpus, two ways to group the chunks into topic classes, retrieval only
inside the classes a query is about. Compared with `advanced` and `advanced_hier` on the **new
annotated development gold**. The combo then attaches only the winning **classifier** to
`advanced` on the **existing** index.

Stay on `--split dev`. Do not commit `.env`.

- What each method does: `docs/hier_class_methods.md`
- Worked numbers: `docs/hier_class_examples.md`
- Official HTML report: `artifacts/hier_class_rag/report.html`

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

4. **`advanced_llm_llm`** is the mixed test: full `advanced` stages on the LLM-chunked index
   with LLM topic routing. Chunking and routing both change, so a gain or loss cannot be
   blamed on one factor alone.

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
| `advanced+kmeans` or `advanced+llm` | existing fixed 512/128 | winner’s classifier | existing | `advanced` stages, class-routed retriever |
| `advanced_llm_llm` | LLM documentation splits | LLM topics | new (806 chunks) | full `advanced` (rewrite + rerank) on that index |

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

Path (outside the repo):

`D:\QBC 13\Project\Data - new Dev\supportdesk-development-annotated-sources-2026-10-15\data\eval\development_source_annotated_v1\development.jsonl`

15 development cases. For each case, gold sources are the `sources[].source.source_id` values;
passage metrics use `selected_spans` (`source_id`, `start_char`, `end_char`). Official
`data/eval` category and acceptable outcomes are kept. The runner writes
`artifacts/hier_class_rag/annotated_dev_cases.jsonl`.

## How to run

From the project root, with the venv active and `SUPPORTDESK_CONFIG` **unset** (real corpus, not
the sample). `.env` must have `OPENAI_API_KEY` and `OPENAI_BASE_URL` (Metis). Models stay
`gpt-4o-mini` and `text-embedding-3-small`.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
# langchain-text-splitters is in pyproject.toml; the venv already has it if you installed -e .

python scripts/hier_class_rag.py --mode live
```

Useful flags:

```powershell
python scripts/hier_class_rag.py --mode live --methods advanced_llm_llm --no-combo
python scripts/hier_class_rag.py --mode live --methods rule_kmeans,rule_llm,advanced,advanced_hier
python scripts/hier_class_rag.py --mode live --no-combo
python scripts/hier_class_rag.py --mode live --rebuild
python scripts/hier_class_rag.py --mode live --limit 3
python scripts/hier_class_rag.py --mode offline   # fake models; not a real score
```

Outputs (gitignored under `artifacts/`):

| File | What |
|---|---|
| `artifacts/hier_class_rag/report.html` | official `supportdesk report` HTML |
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
| `llm_llm` | 0.02 | 0.02 | **0.46** | 0.87 | 0.26 | 0.37 | 806 |
| `advanced` | **0.27** | **0.52** | **0.63** | **1.00** | **0.79** | 0.50 | 636 |
| `advanced_hier` | 0.27 | 0.49 | 0.61 | **1.00** | **0.79** | 0.51 | 636 |
| `advanced+llm` | 0.28 | 0.46 | 0.60 | 0.93 | 0.75 | **0.53** | 636 |
| `advanced_llm_llm` | **0.31** | 0.48 | 0.62 | **1.00** | **0.88** | **0.53** | 806 |

One-factor view (mean recall@6 of the two methods that share the factor):

- LLM chunker 0.42 vs rule chunker 0.28
- LLM classifier 0.39 vs k-means 0.30

**Winner among the four:** `llm_llm`. It is still well below `advanced`, because the 2×2 has
no query rewrite and no rerank — that is expected.

**Combo:** routing `advanced`’s existing index with the LLM classifier did **not** improve
source recall@6 (0.60 vs 0.63) or hit@6 (0.93 vs 1.00). Passage recall@6 rose slightly
(0.53 vs 0.50). A wrong topic filter drops a gold source that `advanced` used to find.
Hard class filters are risky on this corpus: most reports are about Session State, so
classes overlap.

**`advanced_llm_llm`** (advanced rewrite/rerank + LLM chunks + LLM topics) did **not**
beat `advanced` on recall@6 (0.62 vs 0.63). It did find gold **earlier** (MRR 0.88 vs
0.79, recall@1 0.31 vs 0.27) and matched the best passage recall@6 (0.53). Hit@6 is
again 1.00. So it is not a better complete-evidence system; it is a slightly better
first-hit system.

`advanced` remains the best on recall@6. `advanced_hier` is essentially tied with it.

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
