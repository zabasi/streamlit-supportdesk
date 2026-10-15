# Numerical examples for the hierarchical-class comparison

Companion to `docs/hier_class_methods.md`. Formulas are the ones in
`src/supportdesk/eval/metrics.py`. The live case `DEV-5604` is taken from
`artifacts/hier_class_rag/runs/*/results.jsonl`.

---

## 1. The metrics, with a tiny made-up case

Gold sources (what a good retrieve must find):

```
G = {A, B, C}     # 3 gold sources
```

Retrieved top 6, in rank order:

```
rank 1: X
rank 2: A          ← first gold
rank 3: Y
rank 4: B          ← second gold
rank 5: Z
rank 6: W
```

`C` is missing. A retrieved **comment** whose `parent_source_id` is a gold issue also
counts as that issue (`hit_source_ids` in `metrics.py`).

### recall@k

```
recall@k = |gold sources found in top k| / |gold|
```

```
recall@1 = |{X} ∩ {A,B,C}| / 3 = 0 / 3 = 0.00
recall@3 = |{X,A,Y} ∩ {A,B,C}| / 3 = |{A}| / 3 = 0.33
recall@6 = |{X,A,Y,B,Z,W} ∩ {A,B,C}| / 3 = |{A,B}| / 3 = 0.67
```

### hit@k

```
hit@k = 1 if any gold is in the top k, else 0
```

```
hit@1 = 0
hit@3 = 1    # A is at rank 2
hit@6 = 1
```

On a **set** of cases, hit@k is the mean of those 0/1 values. If 14 of 15 cases find at
least one gold in the top 6, hit@6 = 14/15 = 0.933.

### MRR (mean reciprocal rank)

For one case:

```
MRR_case = 1 / rank of the first gold    (0 if none)
```

Here the first gold is at rank 2:

```
MRR_case = 1 / 2 = 0.50
```

The number on the scoreboard is the mean over the 15 cases.

### Why recall can be 0.63 while hit is 1.00

`advanced` on this gold: hit@6 = 1.00 and recall@6 = 0.63.

That means **every** case found at least one gold source, but a typical case has about 3
gold sources and the top 6 only recovered about 1.9 of them:

```
0.63 × 3 ≈ 1.9 gold sources per case in the top 6
```

Hit asks “did we find anything useful?” Recall asks “how complete was the evidence?”

---

## 2. How the published averages were computed

From `artifacts/hier_class_rag/comparison.md`, recall@6:

| method | recall@6 |
|---|---:|
| `rule_kmeans` | 0.2222 |
| `rule_llm` | 0.3333 |
| `llm_kmeans` | 0.3778 |
| `llm_llm` | 0.4556 |
| `advanced` | 0.6333 |
| `advanced_hier` | 0.6111 |
| `advanced+llm` | 0.6000 |
| `advanced_llm_llm` | 0.6222 |

One-factor view (mean of the two methods that share the factor; `advanced` is **not**
included):

```
rule chunker   = (0.2222 + 0.3333) / 2 = 0.27775  →  0.2777
llm chunker    = (0.3778 + 0.4556) / 2 = 0.4167
k-means        = (0.2222 + 0.3778) / 2 = 0.3000
LLM classifier = (0.3333 + 0.4556) / 2 = 0.39445 →  0.3944
```

So the LLM chunker is about **0.14** recall@6 better than the rule chunker, and the LLM
classifier is about **0.09** better than k-means. Those two gains stacked in `llm_llm`.

---

## 3. Real case `DEV-5604`

Report (short): a `st.text_area` inside an expander does not load from Session State after
a button rerun.

**Gold** (3 sources, 7 selected spans):

1. `doc:…/session_state.md`
2. `doc:…/widget-behavior.md`
3. `issue:4651`

A retrieved **comment** on issue 4651 counts as gold source 3.

### `advanced` (found all 3)

Top 6 source ids:

| rank | source | gold? |
|---:|---|---|
| 1 | `session_state.md` | yes (1) |
| 2 | `multipage-apps/widgets.md` | no |
| 3 | `widget-updating-session-state.md` | no |
| 4 | `widget-behavior.md` | yes (2) |
| 5 | `widget-behavior.md` (another chunk) | already counted |
| 6 | `comment:1274881298` (parent `issue:4651`) | yes (3) |

```
found in top 1 = {session_state.md}                         → recall@1 = 1/3 = 0.333
found in top 3 = {session_state.md}                         → recall@3 = 1/3 = 0.333
found in top 6 = {session_state.md, widget-behavior.md, issue:4651}
                                                            → recall@6 = 3/3 = 1.000
first gold at rank 1                                        → MRR     = 1/1 = 1.000
hit@6 = 1
```

`n_queries = 4` on this case: cleaned report + 3 concept queries. Each query produced 15
candidates; RRF fused them; the reranker kept these 6.

### `advanced_hier` (also found all 3)

Same recall@1/@3/@6 = 0.333 / 0.333 / 1.000 and MRR = 1.000. The **order** differs:
`widget-behavior.md` is at rank 6 instead of 4, and a comment on `issue:6370` sits at
rank 5. The section-summary places changed which documentation chunks entered the pool;
for this case it did not change completeness.

### `advanced+llm` (hard filter dropped two golds)

Top 6:

| rank | source | gold? |
|---:|---|---|
| 1 | `session_state.md` | yes (1) |
| 2 | `issue:6370` | no |
| 3–6 | comments on 6939 / 6370 | no |

`widget-behavior.md` and `issue:4651` are absent.

```
recall@1 = 1/3 = 0.333
recall@3 = 1/3 = 0.333
recall@6 = 1/3 = 0.333     # was 1.000 on unfiltered advanced
MRR      = 1/1 = 1.000     # first hit is still rank 1
hit@6    = 1
```

The LLM router labelled the report (and its concept queries) as Session State topics. The
filter then **hid** the widget-behavior page and issue 4651 even though `advanced` knew
how to rank them. That is why the combo’s **mean** recall@6 fell 0.633 → 0.600 and hit@6
fell 1.00 → 0.93 (one or more cases lost their only gold).

### `rule_llm` and `llm_llm` (simple search, no rewrite)

Both returned, among the top 6, `issue:4651` at rank 1 and `session_state.md` at rank 6.
Neither returned `widget-behavior.md`.

```
recall@1 = 1/3 = 0.333
recall@6 = 2/3 = 0.667
MRR      = 1/1 = 1.000
```

They beat k-means on this case (k-means found **zero** gold: recall@6 = 0, MRR = 0) but
still miss the third gold that `advanced` recovered after rewrite + rerank.

### `advanced_llm_llm` on this case (found all 3 again)

Full `advanced` stages on the 806 LLM chunks, filtered to LLM topics. Top 6:

| rank | source | gold? |
|---:|---|---|
| 1 | `issue:4651` | yes (3) |
| 2 | `issue:4651` (another chunk) | already counted |
| 3 | `session-state.md` (concepts page) | no |
| 4 | `session_state.md` (API page) | yes (1) |
| 5 | `session_state.md` (another chunk) | already counted |
| 6 | `widget-behavior.md` | yes (2) |

```
recall@1 = 1/3 = 0.333
recall@3 = 1/3 = 0.333
recall@6 = 3/3 = 1.000
MRR      = 1/1 = 1.000
```

Unlike `advanced+llm` (classifier only, 1/3), putting the LLM **chunks** under `advanced`
brought `widget-behavior.md` and `issue:4651` back. Across all 15 cases this still does
not beat `advanced` on mean recall@6 (0.62 vs 0.63), but the first gold appears earlier
on average (MRR 0.88 vs 0.79).

### `rule_kmeans` / `llm_kmeans` on this case

Top 6 were other issues (`10007`, `9194`, `7862`) and unrelated rendered docs.

```
recall@1 = recall@3 = recall@6 = 0
MRR = 0
hit@6 = 0
```

The query embedding sat nearer the wrong cluster centroids, and `filter` then made that
miss certain.

---

## 4. Class routing with numbers

Suppose the index has 636 chunks and the LLM assigns:

```
session_state          274
callbacks_forms         93
widgets_keys            69
other                  110
… (the rest)
```

(These counts are from the live `advanced+llm` labelling.)

The query is routed to the top 2 topics, say `{session_state, widgets_keys}`:

```
allowed = 274 + 69 = 343 chunks
blocked = 636 − 343 = 293 chunks   # including, on DEV-5604, widget-behavior.md
```

Dense search then ranks **only the 343**. If a gold source lives in a blocked class, its
recall contribution for that case is 0, no matter how similar its embedding is.

k-means uses the same filter, but the buckets are anonymous clusters `0`…`7` and the query
picks the 2 nearest **centroids**:

```
score(cluster c) = cosine(query_vector, mean(vectors of chunks in c))
route = two largest scores
```

---

## 5. `advanced` candidate pool and RRF

One case, 4 queries, 15 candidates each, 10 of them documentation.

Upper bound on unique candidates before fusion (usually less, because lists overlap):

```
4 × 15 = 60 rows
```

Reciprocal Rank Fusion (`src/supportdesk/retrieval/fusion.py`), default `rrf_k = 60`:

```
RRF(chunk) = Σ  1 / (60 + rank_in_that_list)
```

Toy: chunk D is rank 1 on the report query and rank 4 on concept-query 2; unseen on the
other two lists.

```
RRF(D) = 1/(60+1) + 1/(60+4) = 1/61 + 1/64
       = 0.01639 + 0.01563
       = 0.03202
```

Chunk E is rank 2 on only one list:

```
RRF(E) = 1/(60+2) = 1/62 = 0.01613
```

D ranks above E even if E’s raw cosine was higher on that one list. After fusion the
reranker sees 15 chunks and keeps 6 (at least 3 documentation).

---

## 6. `advanced_hier` reserved places

For **each** of the 4 queries:

```
15 candidates
  4 from chunks that overlap the 3 best section summaries   (hier_places)
 11 from ordinary dense_doc_slots
     of which 10 places prefer documentation                (doc_slots)
```

If a useful issue is crowded out of those 11, it never reaches the reranker. If a useful
paragraph sits inside a well-summarised section, it is guaranteed a look. That is why
`advanced_hier` can move **passage** recall a little (0.50 → 0.51) while **source** recall
moves a little the other way (0.63 → 0.61).

---

## 7. Passage recall (why LLM chunks helped the 2×2)

A gold span counts as found when the retrieved chunks cover at least **80%** of its
characters (`eval.passage_found_fraction = 0.8`).

Toy span of 1000 characters. Two retrieved chunks overlap it by 300 and 400 characters
(no overlap with each other):

```
covered = 300 + 400 = 700
coverage = 700 / 1000 = 0.70   < 0.80  →  not found
```

If a third chunk adds 150 more unique characters:

```
covered = 850
coverage = 0.85  ≥ 0.80  →  found
```

Finer documentation chunks (`llm_*`, 806 vs 705 or 636) make it easier for a retrieved
piece to sit **inside** the span instead of around it. That is why `llm_kmeans` /
`llm_llm` passage recall@6 (0.31 / 0.37) beat the rule methods (0.00 / 0.03) even when
source recall was only a bit higher.

---

## 8. What to remember

1. The 2×2 compares **chunking and routing only**.
2. `advanced` / `advanced_hier` add **rewrite + doc slots + fusion + rerank**.
3. The combo asks: “does LLM topic filtering help that full pipeline?” On this gold,
   **no** for source recall (0.63 → 0.60), because a wrong class is a hard miss
   (`DEV-5604` dropped from 3/3 to 1/3).
4. `advanced_llm_llm` asks the same with LLM chunks as well. Mean recall@6 is 0.62
   (still under 0.63). MRR rises to 0.88. On `DEV-5604` completeness comes back to 3/3.
5. Hit@6 = 1.00 on `advanced` means “never empty”; recall@6 = 0.63 means “still missing
   about one gold source per case.”
