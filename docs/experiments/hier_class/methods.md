# What each compared method does

This is the method guide for the hierarchical-class RAG experiment. Scores and how to run
the experiment are in `docs/experiments/hier_class/README.md`. Worked numbers are in
`docs/experiments/hier_class/examples.md`. HTML reports: `artifacts/hier_class_rag/report_one_change.html` (one factor),
`report_two_changes.html` (chunker + filter), `report_llmfilter.html` (filter model),
and `report.html` (everything).

Fourteen systems were measured with **retrieval only** (no answers) on the **same 15 annotated
development cases**. Gold for a case is the source ids in that case’s `selected_spans`. Split =
`dev`.

**Name key:** `llmchunk` is the LLM **splitter** (`llm_chunk_system.txt`). `llmfilter` is
the LLM **topic router** (`topic_classify_system.txt`). `advanced_llmchunk` = chunking
only. `advanced_exist_llmfilter` = filter only. `advanced_llmchunk_llmfilter` = both.

There are three families:

| Family | What it tests | Index | After search |
|---|---|---|---|
| The four new methods | chunker × classifier | newly built | nothing: take the top 6 |
| `advanced`, `advanced_hier` | the current support pipelines | existing fixed 512/128 | rewrite + fusion + rerank + keep 6 |
| Advanced 3×3 | same `advanced` retrieve, every chunker × classifier | existing / rule 705 / llm 806 | same `advanced` stages; class filter only when the cell has one |

The four new methods are **not** a weaker `advanced`. They deliberately omit rewrite and
rerank so the only difference among them is chunking and class routing.

In this comparison `pipeline.scope_gate` was **off**, so the pre-retrieval “is this Streamlit?”
check did not run.

---

## Shared pieces (read these first)

### Chunking

A **chunk** is an exact slice of one corpus record: `source.text[start_char:end_char]`. The
index stores that text plus an embedding (`text-embedding-3-small`). Changing the chunker
requires a new index.

### Classification / routing

After chunks exist, each chunk gets a **class** (a topic bucket). At search time the query is
routed to the best **2** classes. Only chunks in those classes can be returned (`mode = filter`).
A few places are reserved for documentation (`doc_slots`), then the rest are filled from
everything still allowed.

Two classifiers:

- **k-means:** 8 clusters, `doc_anchored` (cluster the documentation first, attach other chunks
  to the nearest documentation cluster). The query is routed by cosine similarity to each
  cluster’s **centroid** (the mean of its chunk vectors).
- **LLM topics:** 8 fixed names (`session_state`, `widgets_keys`, `callbacks_forms`,
  `data_editor`, `caching`, `multipage_navigation`, `expanders_layout`, `other`). Chunks are
  labelled in batches. The query is labelled with one `gpt-4o-mini` call that returns the top
  1–2 topics.

### How `advanced` searches (the current system)

This is the pipeline in `src/supportdesk/pipelines/advanced.py` and
`configs/pipelines/advanced.toml`.

1. **Clean the report** (drop GitHub template chrome; keep Streamlit code).
2. **Rewrite:** one model call writes up to **3 concept queries** (for example “Session State
   and expander widgets”). The cleaned report is also searched (`keep_report_query = true`),
   so there are up to **4 queries**.
3. **Each query** asks the retriever for **15** candidates (`candidate_k`), of which **10**
   places are reserved for documentation (`dense_doc_slots`).
4. **Rank fusion (RRF)** merges the four lists of 15 into one list of 15.
5. **Rerank:** one model call sees a 400-character focused window of each candidate, plus the
   concept queries as “Mechanisms involved”, and orders them.
6. **Keep 6**, with at least **3 documentation** chunks. Overlapping kept spans on the same
   page are merged.

That is why `advanced` can find gold that a single raw embedding search misses: it asks
several questions, forces documentation into the pool, then lets a model pick.

### How `advanced_hier` differs

Same pipeline as `advanced`, except the retriever is `hierarchical_doc_slots`
(`src/supportdesk/retrieval/hierarchy.py`, `configs/pipelines/advanced_hier.toml`).

For **every** query (report + concept queries):

1. Compare the query embedding with **section summaries** (`data/summaries/sections.jsonl`).
2. Take the **3** best-matching documentation sections (`hier_sections = 3`).
3. Of the 15 candidates, **4 places** (`hier_places = 4`) are filled from chunks that sit
   **inside those sections**. The other 11 still come from ordinary `dense_doc_slots`.

This is a **section-summary hierarchy** (summaries → chunks inside those sections). It is
not the same idea as the four new methods, which are a **class hierarchy** (label every
chunk → search only matching labels).

---

## The four new methods

Code: `src/supportdesk/data/langchain_chunkers.py`,
`src/supportdesk/retrieval/class_routing.py`, `scripts/hier_class_rag.py`.

### 1. `rule_kmeans`

**Chunker (rule, LangChain).**

- Documentation / release notes / linked pages: `MarkdownHeaderTextSplitter` on `#` / `##` /
  `###`, then `RecursiveCharacterTextSplitter` (about 1600 characters, 200 overlap) on a
  section that is still too long.
- Issues and comments: recursive splitter only (they are rarely heading-structured).
- Result on this corpus: **705 chunks**.

**Classifier (k-means).** Eight `doc_anchored` clusters. Query → two nearest centroids →
search only those clusters.

**Retrieve.** One query: title + body. No rewrite, no rerank. Top 6.

**What it is good at / weak at.** Cheap and deterministic after embeddings exist. On this
corpus plain clusters still mix “Session State-ish” text with nearby issues, and a wrong
centroid route is a hard miss. Recall@6 was **0.22**.

### 2. `rule_llm`

**Chunker.** Same 705 LangChain-rule chunks.

**Classifier (LLM topics).** Each chunk is assigned one of the eight named topics. The user
report is classified into the top two topics with one model call. Search only those topics.

**Retrieve.** Same simple top-6 path.

**What it is good at / weak at.** Named topics match how people talk about Streamlit better
than anonymous clusters. Still no rewrite/rerank, and a wrong topic still hides gold.
Recall@6 was **0.33**.

### 3. `llm_kmeans`

**Chunker (LLM on documentation only).**

- For each documentation page, `gpt-4o-mini` returns character offsets (`cuts`) where a new
  chunk should start. Cuts are snapped to a heading or blank line. Code fences are not to be
  split.
- A bad or missing answer falls back to embedding-breakpoint packing, then to the rule
  chunker.
- Issues and comments stay on the rule chunker.
- Result: **806 chunks** (documentation is split more finely than the rule chunker).

**Classifier.** Same k-means as `rule_kmeans`, but on the new chunks.

**Retrieve.** Same simple top-6 path.

**What it is good at / weak at.** Smaller documentation pieces help passage overlap (gold
spans are more likely to sit inside one chunk). K-means routing can still send the query to
the wrong cluster. Recall@6 was **0.38**; passage recall@6 jumped to **0.31**.

### 4. `llm_llm` (winner of the four)

**Chunker.** Same 806 LLM-split chunks.

**Classifier.** Same LLM topics as `rule_llm`.

**Retrieve.** Same simple top-6 path.

**What it is good at / weak at.** Best of the 2×2 (recall@6 **0.46**, hit@6 **0.87**). Still
far below `advanced`, because it never rewrites the report and never reranks. This win is
“best chunker × best classifier”, not “better than the current system”.

---

## The current systems and the advanced 3×3

The nine `advanced_*` cells keep **the same retrieve path** as `advanced` (clean → up to 4
queries → 15 candidates with documentation places → RRF → rerank → keep 6). Only the index
and whether a class filter is on change. That is how each chunker and each classifier can
be read **alone** (the `none` / existing rows) and **together** (the four mixed cells).

### 5. `advanced`

Described above. Existing index: **636** fixed windows of 512 tokens with 128 overlap.

On the new gold: recall@6 **0.63**, hit@6 **1.00**, MRR **0.79**. Every case had at least one
gold source in the top 6.

Code: `src/supportdesk/pipelines/advanced.py`, retriever
`src/supportdesk/retrieval/retriever.py` (`DocSlotsRetriever`), reranker
`src/supportdesk/pipelines/stages/evidence.py` (`CorrectiveReranker` with no extra turns in
this run).

### 6. `advanced_hier`

`advanced` with section-summary hierarchy **inside every search**. Same 636 chunks. Needs
`data/summaries/sections.jsonl` (already built).

On the new gold: recall@6 **0.61**, hit@6 **1.00**, MRR **0.79**, passage recall@6 **0.51**
(slightly above `advanced`’s 0.50). Effectively tied with `advanced`. The reserved section
places help some pages and crowd others out of the 15-candidate list.

Code: `src/supportdesk/pipelines/hierarchical.py`,
`src/supportdesk/retrieval/hierarchy.py`.

### 7. `advanced_exist_kmeans` (existing chunks + k-means only)

Same 636 existing chunks as `advanced`. Every search is filtered to the two nearest
`doc_anchored` clusters.

On the new gold: recall@6 **0.62** (−0.01), hit@6 **0.93**, MRR **0.82**. Completeness is
almost unchanged; one case loses its only gold (hit drops). First-hit is a little better
than unfiltered `advanced`.

### 8. `advanced_exist_llmfilter` (existing chunks + LLM topic filter only)

Same 636 chunks. Every search is filtered to the two predicted Streamlit topics.

On the new gold: recall@6 **0.60**, hit@6 **0.93**, MRR **0.75**. Source retrieval got
**worse**. A hard filter can drop a gold source that unfiltered `advanced` already found
(see `DEV-5604` in the examples file: 3/3 → 1/3).

### 9. `advanced_rule` (rule chunks only)

Full `advanced` on the 705 LangChain-rule chunks. No class filter.

On the new gold: recall@6 **0.61** (−0.02), hit@6 **1.00**, MRR **0.69**. Completeness is
almost tied; first-hit is worse. Passage recall is ~0 because gold spans were annotated
on the original windows, not these remapped chunks.

### 10. `advanced_rule_kmeans` (rule chunks + k-means)

The only cell that **beats** `advanced` on source recall@6: **0.66** (+0.02), hit@6
**1.00**, MRR **0.87**. Rule sections plus anonymous clusters happen to keep gold in the
candidate pool while still shrinking the search space. On `DEV-5604` this cell finds 2/3
(misses `issue:4651`).

### 11. `advanced_rule_llmfilter` (rule chunker + LLM topic filter)

Recall@6 **0.58** (−0.06), hit@6 **1.00**, MRR **0.76**. Named topics on heading-split
chunks hide more gold than k-means does on the same index. On `DEV-5604` it finds 2/3
(misses `widget-behavior.md`).

### 12. `advanced_llmchunk` (LLM chunker only, no filter)

Full `advanced` on the 806 LLM-split chunks. No class filter.

Recall@6 **0.59** (−0.04), hit@6 **0.93**, MRR **0.81**. Finer documentation helped the
simple 2×2; under rewrite + rerank it does not. On `DEV-5604` it still finds all 3.

### 13. `advanced_llmchunk_kmeans` (LLM chunker + k-means filter)

Worst cell: recall@6 **0.54** (−0.09), hit@6 **0.87**, MRR **0.83**. Fine chunks plus a
hard cluster filter drop gold that both `advanced` and `advanced_llmchunk` kept. On `DEV-5604`
it still finds all 3 — the mean loss is on other cases.

### 14. `advanced_llmchunk_llmfilter` (LLM chunker + LLM topic filter)

Recall@6 **0.62** (−0.01), hit@6 **1.00**, MRR **0.88** (best first-hit in the grid),
recall@1 **0.31**. Not better as a complete-evidence system; better at putting the first
gold near the top. On `DEV-5604` it finds all 3 (unlike `advanced_exist_llmfilter`).

### 15. `advanced_llmchunk_llmfilter_gpt5mini` (same chunks, stronger filter)

Same 806 `gpt-4o-mini` chunks and the same advanced rewrite/rerank. Chunk labels and
query routing use `gpt-5-mini` (`gpt-5o-mini` does not exist).

Recall@6 **0.59** (worse than `gpt-4o-mini` filter 0.62), hit@6 **1.00**, MRR **0.82**
(was 0.88). A stronger topic model did not fix the hard-filter miss.

---



## How to read the scoreboard

| Metric | Meaning |
|---|---|
| **recall@k** | For one case: (gold sources found in the top k) / (number of gold sources). Then average over the 15 cases. A comment counts as its parent issue. |
| **hit@k** | Share of cases with **at least one** gold source in the top k. |
| **MRR** | Mean of 1 / rank of the first gold source (0 if none). |
| **passage recall@k** | Share of gold *spans* of which the top k chunks cover at least 80%. |

`advanced` can have recall@6 = 0.63 and hit@6 = 1.00 at the same time: every case found
*something* gold, but not every gold source.

The **2×2 one-factor** numbers average the two *simple* methods that share a factor, so
they do not mix `advanced` in:

- LLM chunker (0.42) vs rule chunker (0.28)
- LLM classifier (0.39) vs k-means (0.30)

The **advanced 3×3** averages (each factor over the other two) go the other way: existing
chunks 0.62, rule 0.61, LLM chunks 0.59; no classifier 0.61, k-means 0.61, LLM classifier
0.60. Under rewrite + rerank the extra pieces usually cost recall. The exception is
`advanced_rule_kmeans`.

---

## Code map

| Method | Chunker | Router | Pipeline after search |
|---|---|---|---|
| `rule_kmeans` | `LangChainRuleChunker` | `ClassRoutedRetriever` (centroid) | baseline stages (top 6) |
| `rule_llm` | `LangChainRuleChunker` | `ClassRoutedRetriever` (LLM query) | baseline stages |
| `llm_kmeans` | `LangChainLLMChunker` | centroid | baseline stages |
| `llm_llm` | `LangChainLLMChunker` | LLM query | baseline stages |
| `advanced` | existing fixed 512/128 | none | rewrite + `dense_doc_slots` + rerank |
| `advanced_hier` | existing fixed 512/128 | section summaries | same as `advanced` |
| `advanced_exist_kmeans` | existing fixed 512/128 | centroid on those chunks | same as `advanced` |
| `advanced_exist_llmfilter` | existing fixed 512/128 | LLM topics on those chunks | same as `advanced` |
| `advanced_rule` | `LangChainRuleChunker` | none | full `advanced` on the 705-chunk index |
| `advanced_rule_kmeans` | `LangChainRuleChunker` | centroid | full `advanced` on the 705-chunk index |
| `advanced_rule_llmfilter` | `LangChainRuleChunker` | LLM topic filter | full `advanced` on the 705-chunk index |
| `advanced_llmchunk` | `LangChainLLMChunker` | none | full `advanced` on the 806-chunk index |
| `advanced_llmchunk_kmeans` | `LangChainLLMChunker` | centroid | full `advanced` on the 806-chunk index |
| `advanced_llmchunk_llmfilter` | `LangChainLLMChunker` (`gpt-4o-mini`) | LLM query (`gpt-4o-mini`) | full `advanced` on the 806-chunk index |
| `advanced_llmchunk_llmfilter_gpt5mini` | same `gpt-4o-mini` chunks | LLM query (`gpt-5-mini`) | full `advanced`; rewrite/rerank stay `gpt-4o-mini` |
