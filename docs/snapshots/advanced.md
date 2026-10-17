# Pipeline snapshot: advanced

| | |
|---|---|
| Taken | 2026-10-17T20:57:33+00:00 |
| Profile | configs/pipelines/advanced.toml |
| Mode | live |
| Retriever | `question_doc_slots` |
| Models | chat `gpt-4o-mini`, embedding `text-embedding-3-small`, answer `gpt-5.4-mini` (controlled package v2) |
| Config fingerprint | `56bf02464d1e5f9e` |

## Stages

| # | Stage | Implementation | What it does |
|---|---|---|---|
| 1 | query | `CleanQuery` | One query: the title and the meaningful parts of the report, without template, links or debug info. |
| 2 | retriever | `QuestionExpandedRetriever` | The base retriever's candidates, all kept, plus the chunks of up to ``extra`` more sections whose |
| 3 | reranker | `CorrectiveReranker` | Corrective retrieval on top of another reranker (by default advanced's ExpandingReranker). |
| 4 | packet | `DefaultPacketBuilder` | Keep chunks in rank order until the chunk or token limit is reached. |
| 5 | generator | `ControlledGenerator` | One structured assessment by the model; Python owns the action (controlled routing v2). |
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
| `prompts.rewrite` | configs/prompts/rewrite_concepts_system.txt | file:configs/pipelines/advanced.toml ★ |
| `prompts.rerank` | configs/prompts/rerank_docs_concepts_system.txt | file:configs/pipelines/advanced.toml ★ |
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
| `chunking.method` | fixed | file:configs/pipelines/advanced.toml ★ |
| `chunking.max_tokens` | 512 | file:configs/pipelines/advanced.toml ★ |
| `chunking.overlap_tokens` | 128 | file:configs/pipelines/advanced.toml ★ |
| `chunking.context_header` | False | file:configs/default.toml |
| `chunking.context_header_scope` | start | file:configs/default.toml |
| `chunking.structure_min_chars` | 150 | file:configs/default.toml |
| `chunking.structure_max_chars` | 2000 | file:configs/default.toml |
| `chunking.token_target` | 500 | file:configs/default.toml |
| `chunking.token_soft_max` | 700 | file:configs/default.toml |
| `chunking.token_min_section` | 150 | file:configs/default.toml |
| `chunking.langchain_chunk_chars` | 1600 | file:configs/default.toml |
| `chunking.langchain_overlap_chars` | 200 | file:configs/default.toml |
| `retrieval.top_k` | 8 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.candidate_k` | 15 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.historical_filter` | True | file:configs/pipelines/advanced.toml ★ |
| `retrieval.embed_batch_size` | 64 | file:configs/default.toml |
| `retrieval.bm25_k1` | 1.5 | file:configs/default.toml |
| `retrieval.bm25_b` | 0.75 | file:configs/default.toml |
| `retrieval.hybrid_pool` | 30 | file:configs/default.toml |
| `retrieval.doc_slots` | 10 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.max_queries` | 3 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.keep_report_query` | True | file:configs/pipelines/advanced.toml ★ |
| `retrieval.rerank_snippet_chars` | 400 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.rerank_snippet` | focused | file:configs/pipelines/advanced.toml ★ |
| `retrieval.rerank_keep_docs` | 5 | file:configs/pipelines/advanced.toml ★ |
| `retrieval.min_chunks` | 2 | file:configs/default.toml |
| `retrieval.extract_mode` | keep | file:configs/default.toml |
| `retrieval.extract_show_concepts` | False | file:configs/default.toml |
| `retrieval.extract_protect_top` | 0 | file:configs/default.toml |
| `retrieval.rerank_show_concepts` | True | file:configs/pipelines/advanced.toml ★ |
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
| `retrieval.decompose` | False | file:configs/pipelines/advanced.toml ★ |
| `retrieval.complete` | section | file:configs/pipelines/advanced.toml ★ |
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
| `retrieval.question_sections` | 15 | file:configs/pipelines/advanced.toml ★ |
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
| `context.max_tokens` | 4300 | file:configs/pipelines/advanced.toml ★ |
| `context.max_query_chars` | 4000 | file:configs/pipelines/advanced.toml ★ |
| `context.max_body_chars` | 4000 | file:configs/pipelines/advanced.toml ★ |
| `generation.max_output_tokens` | 900 | file:configs/default.toml |
| `generation.max_retries` | 1 | file:configs/default.toml |
| `generation.temperature` | 0.0 | file:configs/default.toml |
| `generation.timeout_seconds` | 60.0 | file:configs/default.toml |
| `pipeline.name` | advanced | cli |
| `pipeline.retriever` | question_doc_slots | file:configs/pipelines/advanced.toml ★ |
| `pipeline.scope_gate` | llm | file:configs/pipelines/advanced.toml ★ |
| `pipeline.answer_model` | gpt-5.4-mini | file:configs/default.toml |
| `pipeline.controlled_version` | v2 | file:configs/default.toml |
| `pipeline.complete_questions` | True | file:configs/pipelines/advanced.toml ★ |
| `pipeline.version_notes` | True | file:configs/pipelines/advanced.toml ★ |
| `pipeline.generator` | controlled_v2 | file:configs/pipelines/advanced.toml ★ |

Derived: retrieval pool 15, chunks allowed in the prompt 8.

## Walkthrough of one case

**Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun)**

> ### Checklist ### Summary Deleting a keyed widget's entry from `st.session_state` (e.g. `del st.session_state["foo"]` inside a callback) does not reset the widget, but can lead to an inconsistency between frontend and backend. The Python side returns the widget's *default* value for that run, but the frontend keeps displaying the old value, so backend and frontend disagree. On the **next** rerun the backend silently re-adopts the stale frontend value, so the deletion is effectively undone. ### Reproducible Code Example ```Python import streamlit as st text = st.text_input("Foo", value="defaul…

Gold evidence for this case: `doc:pinned:content/develop/concepts/architecture/widget-behavior.md`, `doc:rendered:7890201016c75cb0`

### Environment read from the report

browser = `Chromium`, operating_system = `Ubuntu 22.04`, python_version = `3.14.6`, streamlit_version = `1.59.2`

### 1. query: `CleanQuery`

- Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun) Deleting a keyed widget's entry from `st.session_state` (e.g. `del st.session_state["foo"]` inside a callback) does not reset the widget, but can lead to an inconsistency be…

### 2. retriever: `QuestionExpandedRetriever`

30 candidates (pool 15):

| rank | score | source | type | chunk | gold |
|---|---|---|---|---|---|
| 1 | 0.6666 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:005` | ✓ |
| 2 | 0.6566 | comment:1007820799 | comment | `comment:1007820799:000` |  |
| 3 | 0.6530 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:000` | ✓ |
| 4 | 0.6434 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:003` | ✓ |
| 5 | 0.6294 | issue:4247 | issue | `issue:4247:000` |  |
| 6 | 0.6280 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:004` | ✓ |
| 7 | 0.6247 | doc:pinned:content/develop/concepts/multipage-apps/widgets.md | documentation | `doc:pinned:content/develop/concepts/multipage-apps/widgets.md:001` |  |
| 8 | 0.6237 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:001` | ✓ |
| 9 | 0.6134 | doc:pinned:content/kb/FAQ/widget-updating-session-state.md | documentation | `doc:pinned:content/kb/FAQ/widget-updating-session-state.md:000` |  |
| 10 | 0.6125 | doc:pinned:content/develop/concepts/multipage-apps/widgets.md | documentation | `doc:pinned:content/develop/concepts/multipage-apps/widgets.md:000` |  |
| 11 | 0.6108 | doc:rendered:7890201016c75cb0 | documentation | `doc:rendered:7890201016c75cb0:002` | ✓ |
| 12 | 0.6100 | comment:2495181293 | comment | `comment:2495181293:000` |  |
| 13 | 0.6084 | doc:pinned:content/develop/api-reference/caching-and-state/session_state.md | documentation | `doc:pinned:content/develop/api-reference/caching-and-state/session_state.md:000` |  |
| 14 | 0.6070 | comment:924405437 | comment | `comment:924405437:000` |  |
| 15 | 0.6009 | linked:efcf225600e4fbeb1fda | linked_reference | `linked:efcf225600e4fbeb1fda:000` |  |
| 16 | 0.5268 | doc:pinned:content/develop/api-reference/caching-and-state/session_state.md | documentation | `doc:pinned:content/develop/api-reference/caching-and-state/session_state.md:002` |  |
| 17 | 0.4932 | doc:linked:4d90122c5d83ea8a2e4d | documentation | `doc:linked:4d90122c5d83ea8a2e4d:002` |  |
| 18 | 0.4679 | doc:linked:4d90122c5d83ea8a2e4d | documentation | `doc:linked:4d90122c5d83ea8a2e4d:001` |  |
| 19 | 0.4617 | doc:linked:78cb80651cf78a0964fe | documentation | `doc:linked:78cb80651cf78a0964fe:000` |  |
| 20 | 0.4603 | doc:rendered:4e76630f08712af2 | documentation | `doc:rendered:4e76630f08712af2:000` |  |
| 21 | 0.4489 | doc:linked:4d90122c5d83ea8a2e4d | documentation | `doc:linked:4d90122c5d83ea8a2e4d:003` |  |
| 22 | 0.4440 | doc:linked:ed65d8f03d4d9b9aeb73 | documentation | `doc:linked:ed65d8f03d4d9b9aeb73:000` |  |
| 23 | 0.4436 | doc:linked:4bf3a73927a3d7d51d6f | documentation | `doc:linked:4bf3a73927a3d7d51d6f:002` |  |
| 24 | 0.4414 | doc:linked:b965c9a9da74f5676ffe | documentation | `doc:linked:b965c9a9da74f5676ffe:000` |  |
| 25 | 0.4344 | doc:pinned:content/develop/concepts/architecture/session-state.md | documentation | `doc:pinned:content/develop/concepts/architecture/session-state.md:000` |  |
| 26 | 0.4304 | doc:linked:f62d2bc4926afc5516ff | documentation | `doc:linked:f62d2bc4926afc5516ff:000` |  |
| 27 | 0.4225 | doc:linked:4bf3a73927a3d7d51d6f | documentation | `doc:linked:4bf3a73927a3d7d51d6f:001` |  |
| 28 | 0.4200 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:002` | ✓ |
| 29 | 0.4166 | doc:linked:a80b997108d279d99857 | documentation | `doc:linked:a80b997108d279d99857:001` |  |
| 30 | 0.4157 | doc:rendered:7890201016c75cb0 | documentation | `doc:rendered:7890201016c75cb0:000` | ✓ |

### 3. reranker: `CorrectiveReranker`

New order:

| rank | score | source | type | chunk | gold |
|---|---|---|---|---|---|
| 1 | 0.6666 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#12836-17083` | ✓ |
| 2 | 0.6530 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#0-9535` | ✓ |
| 3 | 0.6434 | doc:pinned:content/develop/concepts/architecture/widget-behavior.md | documentation | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:003` | ✓ |
| 4 | 0.6566 | comment:1007820799 | comment | `comment:1007820799:000` |  |
| 5 | 0.6294 | issue:4247 | issue | `issue:4247:000` |  |
| 6 | 0.6100 | comment:2495181293 | comment | `comment:2495181293:000` |  |

### 4. packet: `DefaultPacketBuilder`

6 chunks, 3048 tokens shown to the model:

| label | chunk | tokens |
|---|---|---|
| S1 | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#12836-17083` | 671 |
| S2 | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#0-9535` | 1406 |
| S3 | `doc:pinned:content/develop/concepts/architecture/widget-behavior.md:003` | 512 |
| S4 | `comment:1007820799:000` | 94 |
| S5 | `issue:4247:000` | 305 |
| S6 | `comment:2495181293:000` | 60 |

### 5. generator: `ControlledGenerator`

Attempts: assessment by `gpt-5.4-mini` (controlled_v2), routed by `internal_investigation_no_useful_question`. Proposed outcome: **escalate**. Raw model output:

```json
{
  "case_id": "DEV-16388",
  "main_need": "Determine whether deleting a keyed widget's `st.session_state` entry should reset the widget or whether the stale frontend value is expected, and identify a code pattern that avoids the backend/frontend divergence.",
  "reported_observation_refs": [
    "R0",
    "R3",
    "R4",
    "R5",
    "R7",
    "R13"
  ],
  "required_human_review": null,
  "information_gaps": [],
  "answer_candidate": null,
  "internal_investigation": {
    "unresolved_behavior": "In the minimal example, `del st.session_state['foo']` inside the Delete callback leaves the text input visibly at 'hello' while the backend briefly reports the default value on that rerun, then re-adopts the stale frontend value on the next rerun.",
    "why_internal_review_needed": "This behavior is not explained by the shown app code and appears inconsistent with the documented keyed-widget lifecycle, which says a deleted widget key is recreated from the current frontend value rather than reset to default. A maintainer review is needed to determine whether Streamlit 1.59.2 has a regression or an undocumented interaction here.",
    "basis_refs": [
      "R3",
      "R4",
      "R5",
      "R13",
      "S1",
      "S2"
    ]
  },
  "workflow_findings": [
    {
      "report_ref": "R7",
      "problem": "The `text_input` and the callbacks share the same widget-bound key (`'foo'`), and `delete()` removes that same key from `st.session_state`.",
      "correction": "Do not use direct deletion of the widget's own key as a reset mechanism; use a separate permanent app-state key plus a temporary widget key, copying between them as shown in the widget-behavior guide.",
      "evidence_refs": [
        "S1",
        "S2"
      ]
    },
    {
      "report_ref": "R9",
      "problem": "The Delete button triggers a callback before the page reruns, so the removal happens as a prefix to the rerun that reconstructs the same keyed widget.",
      "correction": "Because the widget is still rendered with the same identity on that rerun, Streamlit reconnects to the existing widget rather than creating a fresh defaulted one; the separate-key pattern is the documented way to decouple stored data from widget state.",
      "evidence_refs": [
        "S2",
        "S3",
        "S1"
      ]
    }
  ]
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
  "next_step": "Propose maintainer investigation within the reported environment and supplied controls; no ticket has been created.",
  "sources": [
    {
      "chunk_id": "doc:pinned:content/develop/concepts/architecture/widget-behavior.md#0-9535",
      "type": "documentation",
      "url": "https://docs.streamlit.io/develop/concepts/architecture/widget-behavior",
      "version": "695e6ce2fb1b23afc6df66b40420ab0013ad72fd"
    },
    {
      "chunk_id": "doc:pinned:content/develop/concepts/architecture/widget-behavior.md#12836-17083",
      "type": "documentation",
      "url": "https://docs.streamlit.io/develop/concepts/architecture/widget-behavior",
      "version": "695e6ce2fb1b23afc6df66b40420ab0013ad72fd"
    }
  ]
}
```

### Result

Outcome: **escalate**

Reply to the reporter:

> Maintainer review is needed: In the minimal example, `del st.session_state['foo']` inside the Delete callback leaves the text input visibly at 'hello' while the backend briefly reports the default value on that rerun, then re-adopts the stale frontend value on the next rerun. This behavior is not explained by the shown app code and appears inconsistent with the documented keyed-widget lifecycle, which says a deleted widget key is recreated from the current frontend value rather than reset to default. A maintainer review is needed to determine whether Streamlit 1.59.2 has a regression or an undocumented interaction here. [R3][R4][R5][R13][S1][S2]

Citations: `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#12836-17083`, `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#0-9535`

Model calls during this run:

| stage | type | mode | tokens in | tokens out | cost (USD) | status |
|---|---|---|---|---|---|---|
| scope_classify | chat | cache | 874 | 25 | 0.000000 | ok |
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| rerank | chat | cache | 3528 | 44 | 0.000000 | ok |
| answer | chat | cache | 8432 | 4384 | 0.000000 | ok |

Time: 0.107s

## Latest evaluation results

**full** run `e2e-advanced-final` (2026-10-17T20:25:57+00:00, mode replay):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
evidence   recall @1 0.32  @3 0.60  @6 0.71  @10 0.73   MRR 0.89
           answers cite gold evidence 1.00
decisions  outcome accuracy 0.80   (ambiguous 0.67  answerable 1.00  escalation 0.67)
grounding  citation violations 0.00   answers without citation 0.00   errors 0.00   budget stops 0
cost       $0.0000 (+$0.3633 replayed, not spent)   0 live calls, 60 cache hits   tokens in/out 188972/57115
time       mean 0.117s   p95 0.182s
```

**retrieval** run `q41-10` (2026-10-16T18:09:32+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
       15 have gold evidence; 0 do not (excluded from recall)
evidence   recall @1 0.32  @3 0.57  @6 0.66  @10 0.68   MRR 0.89
           hit    @1 0.80  @3 1.00  @6 1.00  @10 1.00   gold in the prompt packet 0.68
passages   recall @1 0.35  @3 0.56  @6 0.65  @10 0.68   MRR 0.89
           hit    @1 0.80  @3 1.00  @6 1.00  @10 1.00   coverage@10 0.68   in the prompt packet 0.68
           ambiguous   recall@10 0.81  (n=6)
           answerable  recall@10 0.58  (n=6)
           escalation  recall@10 0.61  (n=3)
cost       $0.0068   15 live calls, 30 cache hits   tokens in/out 45182/996
time       mean 3.560s   p95 5.193s
```
