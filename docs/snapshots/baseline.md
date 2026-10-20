# Pipeline snapshot: baseline

| | |
|---|---|
| Taken | 2026-10-20T12:13:04+00:00 |
| Profile | configs/pipelines/baseline.toml |
| Mode | live |
| Retriever | `dense` |
| Models | chat `gpt-4o-mini`, embedding `text-embedding-3-small` |
| Config fingerprint | `067712fc4369138d` |

## Stages

| # | Stage | Implementation | What it does |
|---|---|---|---|
| 1 | query | `PlainQuery` | One query: the report as written, cut to ``context.max_query_chars``. |
| 2 | retriever | `DenseRetriever` | Embedding similarity between the query and each chunk. |
| 3 | reranker | `NoRerank` | Keep the retriever's order. |
| 4 | packet | `DefaultPacketBuilder` | Keep chunks in rank order until the chunk or token limit is reached. |
| 5 | generator | `SingleCallGenerator` | One prompt with the evidence, one model call (retried only for provider errors or non-JSON). |
| 6 | validator | `NoValidator` | No checks beyond the citation-membership flags the parser already sets. |
| 7 | summarizer | `build_summary` | The maintainer summary, built in code from the validated answer. |

## Settings

Values marked ★ come from this pipeline's own profile.

| Setting | Value | Source |
|---|---|---|
| `models.chat` | gpt-4o-mini | file:configs/default.toml |
| `models.rerank` |  | file:configs/default.toml |
| `models.embedding` | text-embedding-3-small | file:configs/default.toml |
| `models.second_embedding` | text-embedding-3-large | file:configs/default.toml |
| `models.base_url` | (unset) | default |
| `models.api_key_env` | OPENAI_API_KEY | file:configs/default.toml |
| `prompts.system` | configs/prompts/answer_system.txt | file:configs/default.toml |
| `prompts.output_format` | configs/prompts/answer_output_format.txt | file:configs/default.toml |
| `prompts.rewrite` | configs/prompts/rewrite_system.txt | file:configs/default.toml |
| `prompts.rerank` | configs/prompts/rerank_system.txt | file:configs/default.toml |
| `prompts.policy` | configs/prompts/policy_v3_system.txt | file:configs/default.toml |
| `prompts.rerank_second` |  | file:configs/default.toml |
| `prompts.scope` | configs/prompts/scope_system.txt | file:configs/default.toml |
| `prompts.section_questions` | configs/prompts/section_questions_system.txt | file:configs/default.toml |
| `prompts.topic_discover` | configs/prompts/topic_discover_system.txt | file:configs/default.toml |
| `prompts.topic_refine` | configs/prompts/topic_refine_system.txt | file:configs/default.toml |
| `prompts.topic_classify` | configs/prompts/topic_classify_system.txt | file:configs/default.toml |
| `prompts.label_cluster` | configs/prompts/label_cluster_system.txt | file:configs/default.toml |
| `prompts.scope_rewrite` | configs/prompts/scope_rewrite_system.txt | file:configs/default.toml |
| `prompts.summarize_section` | configs/prompts/summarize_section_system.txt | file:configs/default.toml |
| `prompts.judge_coverage` | configs/prompts/judge_coverage_system.txt | file:configs/default.toml |
| `prompts.rewrite_missing` | configs/prompts/rewrite_missing_system.txt | file:configs/default.toml |
| `prompts.extract` | configs/prompts/extract_system.txt | file:configs/default.toml |
| `prompts.judge` | configs/prompts/judge_system.txt | file:configs/default.toml |
| `prompts.loop_rewrite` | configs/prompts/loop_rewrite_system.txt | file:configs/default.toml |
| `prompts.judge_scored` | configs/prompts/judge_scored_system.txt | file:configs/default.toml |
| `chunking.method` | fixed | file:configs/pipelines/baseline.toml ★ |
| `chunking.max_tokens` | 512 | file:configs/pipelines/baseline.toml ★ |
| `chunking.overlap_tokens` | 128 | file:configs/pipelines/baseline.toml ★ |
| `chunking.context_header` | False | file:configs/default.toml |
| `chunking.context_header_scope` | start | file:configs/default.toml |
| `chunking.structure_min_chars` | 150 | file:configs/default.toml |
| `chunking.structure_max_chars` | 2000 | file:configs/default.toml |
| `chunking.token_target` | 500 | file:configs/default.toml |
| `chunking.token_soft_max` | 700 | file:configs/default.toml |
| `chunking.token_min_section` | 150 | file:configs/default.toml |
| `chunking.langchain_chunk_chars` | 1600 | file:configs/default.toml |
| `chunking.langchain_overlap_chars` | 200 | file:configs/default.toml |
| `retrieval.top_k` | 6 | file:configs/pipelines/baseline.toml ★ |
| `retrieval.candidate_k` | (unset) | default |
| `retrieval.historical_filter` | True | file:configs/pipelines/baseline.toml ★ |
| `retrieval.embed_batch_size` | 64 | file:configs/default.toml |
| `retrieval.bm25_k1` | 1.5 | file:configs/default.toml |
| `retrieval.bm25_b` | 0.75 | file:configs/default.toml |
| `retrieval.hybrid_pool` | 30 | file:configs/default.toml |
| `retrieval.doc_slots` | 0 | file:configs/default.toml |
| `retrieval.max_queries` | 1 | file:configs/default.toml |
| `retrieval.keep_report_query` | False | file:configs/default.toml |
| `retrieval.rerank_snippet_chars` | 600 | file:configs/default.toml |
| `retrieval.rerank_snippet` | start | file:configs/default.toml |
| `retrieval.rerank_keep_docs` | (unset) | default |
| `retrieval.min_chunks` | 2 | file:configs/default.toml |
| `retrieval.extract_mode` | keep | file:configs/default.toml |
| `retrieval.extract_show_concepts` | False | file:configs/default.toml |
| `retrieval.extract_protect_top` | 0 | file:configs/default.toml |
| `retrieval.rerank_show_concepts` | False | file:configs/default.toml |
| `retrieval.corrective_turns` | 0 | file:configs/default.toml |
| `retrieval.corrective_judge` | relevance | file:configs/default.toml |
| `retrieval.corrective_max_missing` | 2 | file:configs/default.toml |
| `retrieval.summary_sections` | 0 | file:configs/default.toml |
| `retrieval.hier_sections` | 3 | file:configs/default.toml |
| `retrieval.hier_places` | 3 | file:configs/default.toml |
| `retrieval.second_weight` | 0.5 | file:configs/default.toml |
| `retrieval.report_views` | none | file:configs/default.toml |
| `retrieval.view_weight` | 0.5 | file:configs/default.toml |
| `retrieval.rerank_mode` | llm | file:configs/default.toml |
| `retrieval.rerank_merge` | rrf | file:configs/default.toml |
| `retrieval.rerank_split` | 4 | file:configs/default.toml |
| `retrieval.rerank_merge_k` | 10 | file:configs/default.toml |
| `retrieval.bm25_extra` | 0 | file:configs/default.toml |
| `retrieval.rerank_fuse_weight` | 0.0 | file:configs/default.toml |
| `retrieval.rerank_fuse_k` | 10 | file:configs/default.toml |
| `retrieval.rerank_versions` | False | file:configs/default.toml |
| `retrieval.release_slots` | 0 | file:configs/default.toml |
| `retrieval.rerank_show_question` | False | file:configs/default.toml |
| `retrieval.rerank_show_sections` | False | file:configs/default.toml |
| `retrieval.pool_mode` | fuse | file:configs/default.toml |
| `retrieval.union_report` | 15 | file:configs/default.toml |
| `retrieval.union_per_concept` | 12 | file:configs/default.toml |
| `retrieval.union_max` | 40 | file:configs/default.toml |
| `retrieval.quota_report_dense` | 5 | file:configs/default.toml |
| `retrieval.quota_report_bm25` | 5 | file:configs/default.toml |
| `retrieval.quota_concepts_dense` | 10 | file:configs/default.toml |
| `retrieval.concept_weight` | 1.0 | file:configs/default.toml |
| `retrieval.decompose` | True | file:configs/default.toml |
| `retrieval.complete` | none | file:configs/default.toml |
| `retrieval.complete_max_tokens` | 900 | file:configs/default.toml |
| `retrieval.cluster_size` | 100 | file:configs/default.toml |
| `retrieval.cluster_method` | kmeans | file:configs/default.toml |
| `retrieval.cluster_top` | 2 | file:configs/default.toml |
| `retrieval.cluster_route` | keywords | file:configs/default.toml |
| `retrieval.cluster_mode` | filter | file:configs/default.toml |
| `retrieval.question_filter_top` | 0 | file:configs/default.toml |
| `retrieval.question_chunk` | overlap | file:configs/default.toml |
| `retrieval.question_snippet` | report | file:configs/default.toml |
| `retrieval.question_order` | append | file:configs/default.toml |
| `retrieval.question_keep` | 0 | file:configs/default.toml |
| `retrieval.question_sections` | 5 | file:configs/default.toml |
| `retrieval.class_route` | llm | file:configs/default.toml |
| `retrieval.class_top` | 2 | file:configs/default.toml |
| `retrieval.class_mode` | filter | file:configs/default.toml |
| `retrieval.class_boost` | 0.05 | file:configs/default.toml |
| `retrieval.cluster_boost` | 0.05 | file:configs/default.toml |
| `retrieval.expand` | none | file:configs/default.toml |
| `retrieval.expand_max_candidates` | 30 | file:configs/default.toml |
| `retrieval.section_candidates` | 30 | file:configs/default.toml |
| `retrieval.rerank_doc_places` | 0 | file:configs/default.toml |
| `retrieval.bm25_slots` | 0 | file:configs/default.toml |
| `retrieval.rrf_k` | 60 | file:configs/default.toml |
| `retrieval.max_loops` | 3 | file:configs/default.toml |
| `context.max_chunks` | (unset) | default |
| `context.max_tokens` | 3200 | file:configs/pipelines/baseline.toml ★ |
| `context.max_query_chars` | 4000 | file:configs/pipelines/baseline.toml ★ |
| `context.max_body_chars` | 4000 | file:configs/pipelines/baseline.toml ★ |
| `generation.max_output_tokens` | 900 | file:configs/default.toml |
| `generation.max_retries` | 1 | file:configs/default.toml |
| `generation.temperature` | 0.0 | file:configs/default.toml |
| `generation.timeout_seconds` | 60.0 | file:configs/default.toml |
| `pipeline.name` | baseline | cli |
| `pipeline.retriever` | dense | file:configs/pipelines/baseline.toml ★ |
| `pipeline.scope_gate` | off | file:configs/default.toml |
| `pipeline.answer_model` | gpt-5.4-mini | file:configs/default.toml |
| `pipeline.controlled_version` | v2 | file:configs/default.toml |
| `pipeline.complete_questions` | False | file:configs/default.toml |
| `pipeline.version_notes` | False | file:configs/default.toml |
| `pipeline.generator` | single_call | file:configs/default.toml |

Derived: retrieval pool 6, chunks allowed in the prompt 6.

## Walkthrough of one case

**Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun)**

> ### Checklist ### Summary Deleting a keyed widget's entry from `st.session_state` (e.g. `del st.session_state["foo"]` inside a callback) does not reset the widget, but can lead to an inconsistency between frontend and backend. The Python side returns the widget's *default* value for that run, but the frontend keeps displaying the old value, so backend and frontend disagree. On the **next** rerun the backend silently re-adopts the stale frontend value, so the deletion is effectively undone. ### Reproducible Code Example ```Python import streamlit as st text = st.text_input("Foo", value="defaul…

Gold evidence for this case: `doc:pinned:content/develop/concepts/architecture/widget-behavior.md`, `doc:rendered:7890201016c75cb0`

### Environment read from the report

browser = `Chromium`, operating_system = `Ubuntu 22.04`, python_version = `3.14.6`, streamlit_version = `1.59.2`

### 1. query: `PlainQuery`

- Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun) ### Checklist ### Summary Deleting a keyed widget's entry from `st.session_state` (e.g. `del st.session_state["foo"]` inside a callback) does not reset the widget, but can l…

### 2. retriever: `DenseRetriever`

6 candidates (pool 6):

| rank | score | source | type | chunk | gold |
|---|---|---|---|---|---|
| 1 | 0.7494 | issue:4247 | issue | `issue:4247:000` |  |
| 2 | 0.7061 | issue:12782 | issue | `issue:12782:000` |  |
| 3 | 0.7042 | issue:9194 | issue | `issue:9194:000` |  |
| 4 | 0.6949 | issue:9171 | issue | `issue:9171:000` |  |
| 5 | 0.6902 | issue:12013 | issue | `issue:12013:000` |  |
| 6 | 0.6838 | issue:5537 | issue | `issue:5537:000` |  |

### 3. reranker: `NoRerank`

Order unchanged; kept the top 6.

### 4. packet: `DefaultPacketBuilder`

6 chunks, 1976 tokens shown to the model:

| label | chunk | tokens |
|---|---|---|
| S1 | `issue:4247:000` | 305 |
| S2 | `issue:12782:000` | 468 |
| S3 | `issue:9194:000` | 288 |
| S4 | `issue:9171:000` | 512 |
| S5 | `issue:12013:000` | 175 |
| S6 | `issue:5537:000` | 228 |

### 5. generator: `SingleCallGenerator`

Attempts: ok. Proposed outcome: **escalate**. Raw model output:

```json
{
  "outcome": "escalate",
  "confidence": "low",
  "user_reply": "Thank you for your detailed report. It appears that you are experiencing a divergence between the frontend and backend values of a widget after modifying the session state. This issue may be related to known problems with widget state management in Streamlit. However, the evidence does not provide a clear resolution or confirm whether this is a regression or a bug in your specific version. I recommend escalating this issue to a maintainer for further investigation.",
  "known_facts": [],
  "open_questions": [],
  "next_step": "Escalate the issue to a maintainer for further investigation.",
  "citations": []
}
```

### 6. validator: `NoValidator`

Outcome kept: **escalate**.
Violations: none.

### 7. summarizer: `build_summary`

```json
{
  "problem": "Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun)",
  "outcome": "escalate",
  "environment": {
    "known": {
      "streamlit_version": "1.59.2",
      "python_version": "3.14.6",
      "operating_system": "Ubuntu 22.04",
      "browser": "Chromium"
    },
    "unknown": []
  },
  "known_facts": [],
  "open_questions": [],
  "next_step": "Escalate the issue to a maintainer for further investigation.",
  "sources": []
}
```

### Result

Outcome: **escalate**

Reply to the reporter:

> Thank you for your detailed report. It appears that you are experiencing a divergence between the frontend and backend values of a widget after modifying the session state. This issue may be related to known problems with widget state management in Streamlit. However, the evidence does not provide a clear resolution or confirm whether this is a regression or a bug in your specific version. I recommend escalating this issue to a maintainer for further investigation.

Citations: none

Model calls during this run:

| stage | type | mode | tokens in | tokens out | cost (USD) | status |
|---|---|---|---|---|---|---|
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| answer | chat | cache | 5245 | 147 | 0.000000 | ok |

Time: 0.037s

## Latest evaluation results

**full** run `final-dev-baseline-6` (2026-10-18T14:52:02+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
evidence   recall @1 0.02  @3 0.02  @6 0.04  @10 0.04   MRR 0.08
           answers cite gold evidence 0.20
decisions  outcome accuracy 0.33   (ambiguous 0.33  answerable 0.17  escalation 0.67)
grounding  citation violations 0.00   answers without citation 0.00   errors 0.00   budget stops 0
cost       $0.0000 (+$0.0116 replayed, not spent)   0 live calls, 30 cache hits   tokens in/out 66716/2693
time       mean 0.020s   p95 0.044s
```

**retrieval** run `retrieval-test-baseline-8` (2026-10-18T09:14:11+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
       15 have gold evidence; 0 do not (excluded from recall)
evidence   recall @1 0.00  @3 0.00  @6 0.00  @10 0.00   MRR 0.00
           hit    @1 0.00  @3 0.00  @6 0.00  @10 0.00   gold in the prompt packet 0.00
passages   recall @1 0.00  @3 0.00  @6 0.00  @10 0.00   MRR 0.00
           hit    @1 0.00  @3 0.00  @6 0.00  @10 0.00   coverage@10 0.00   in the prompt packet 0.00
           ambiguous   recall@10 0.00  (n=6)
           answerable  recall@10 0.00  (n=6)
           escalation  recall@10 0.00  (n=3)
cost       $0.0000   0 live calls, 15 cache hits   tokens in/out 0/0
time       mean 0.008s   p95 0.014s
```
