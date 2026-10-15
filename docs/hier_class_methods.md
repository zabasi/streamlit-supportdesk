# What each compared method does

This is the method guide for the hierarchical-class RAG experiment. Scores and how to run
the experiment are in `docs/hier_class_rag.md`. Worked numbers are in
`docs/hier_class_examples.md`. The official HTML comparison is
`artifacts/hier_class_rag/report.html`.

Eight systems were measured with **retrieval only** (no answers) on the **same 15 annotated
development cases**. Gold for a case is the source ids in that case’s `selected_spans`. Split =
`dev`.

There are two families:

| Family | What it tests | Index | After search |
|---|---|---|---|
| The four new methods | chunker × classifier | newly built | nothing: take the top 6 |
| `advanced`, `advanced_hier`, `advanced+llm` | the current support pipeline, and routing on top of it | existing fixed 512/128 | rewrite + fusion + rerank + keep 6 |
| `advanced_llm_llm` | those two LLM pieces **inside** `advanced` | new LLM index (806) | same `advanced` stages |

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

## The current systems and the combo

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

### 7. `advanced+llm` (the combo)

**Not** `advanced` rebuilt with LLM chunks. The winning **classifier only** (LLM topics) is
applied to `advanced`’s existing 636 chunks. Then the full `advanced` pipeline runs, but each
search is filtered to the two predicted topics.

On the new gold: recall@6 **0.60**, hit@6 **0.93**, MRR **0.75**. Source retrieval got
**worse**. Passage recall@6 rose slightly to **0.53**. A hard filter can drop a gold source
that unfiltered `advanced` already found (see `DEV-5604` in the examples file).

### 8. `advanced_llm_llm` (advanced + LLM chunker + LLM classifier)

This is `advanced`’s full retrieve path (clean → up to 4 queries → 15 candidates with
documentation places → RRF → rerank → keep 6) running on the **806 LLM-split chunks**,
and every search is filtered to the two LLM topics.

It mixes two changes at once (new chunks *and* topic filter), so a move versus `advanced`
cannot be attributed to one of them.

On the new gold: recall@6 **0.62** (just under `advanced` 0.63), hit@6 **1.00**, MRR
**0.88** (above `advanced` 0.79), recall@1 **0.31** (above 0.27), passage recall@6
**0.53**. Not better as a complete-evidence system; better at putting the first gold
near the top.

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

The **one-factor** numbers average the two methods that share a factor, so they do not mix
`advanced` in:

- LLM chunker (0.42) vs rule chunker (0.28)
- LLM classifier (0.39) vs k-means (0.30)

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
| `advanced+llm` | existing fixed 512/128 | LLM topics on those chunks | same as `advanced` |
| `advanced_llm_llm` | `LangChainLLMChunker` | LLM query | full `advanced` on the 806-chunk index |
