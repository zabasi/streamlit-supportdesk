# Pipeline snapshot: advanced

| | |
|---|---|
| Taken | 2026-10-20T12:13:05+00:00 |
| Profile | configs/pipelines/advanced-gpt5mini.toml |
| Mode | live |
| Retriever | `question_doc_slots` |
| Models | chat `gpt-4o-mini`, embedding `text-embedding-3-small`, answer `gpt-5-mini` (controlled package v2) |
| Config fingerprint | `1f90e885374348f7` |

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
| `prompts.rewrite` | configs/prompts/rewrite_concepts_system.txt | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `prompts.rerank` | configs/prompts/rerank_docs_concepts_system.txt | file:configs/pipelines/advanced-gpt5mini.toml ★ |
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
| `chunking.method` | fixed | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `chunking.max_tokens` | 512 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `chunking.overlap_tokens` | 128 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `chunking.context_header` | False | file:configs/default.toml |
| `chunking.context_header_scope` | start | file:configs/default.toml |
| `chunking.structure_min_chars` | 150 | file:configs/default.toml |
| `chunking.structure_max_chars` | 2000 | file:configs/default.toml |
| `chunking.token_target` | 500 | file:configs/default.toml |
| `chunking.token_soft_max` | 700 | file:configs/default.toml |
| `chunking.token_min_section` | 150 | file:configs/default.toml |
| `chunking.langchain_chunk_chars` | 1600 | file:configs/default.toml |
| `chunking.langchain_overlap_chars` | 200 | file:configs/default.toml |
| `retrieval.top_k` | 8 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.candidate_k` | 15 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.historical_filter` | True | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.embed_batch_size` | 64 | file:configs/default.toml |
| `retrieval.bm25_k1` | 1.5 | file:configs/default.toml |
| `retrieval.bm25_b` | 0.75 | file:configs/default.toml |
| `retrieval.hybrid_pool` | 30 | file:configs/default.toml |
| `retrieval.doc_slots` | 10 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.max_queries` | 3 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.keep_report_query` | True | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.rerank_snippet_chars` | 400 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.rerank_snippet` | focused | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.rerank_keep_docs` | 5 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.min_chunks` | 2 | file:configs/default.toml |
| `retrieval.extract_mode` | keep | file:configs/default.toml |
| `retrieval.extract_show_concepts` | False | file:configs/default.toml |
| `retrieval.extract_protect_top` | 0 | file:configs/default.toml |
| `retrieval.rerank_show_concepts` | True | file:configs/pipelines/advanced-gpt5mini.toml ★ |
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
| `retrieval.decompose` | False | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `retrieval.complete` | section | file:configs/pipelines/advanced-gpt5mini.toml ★ |
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
| `retrieval.question_sections` | 15 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
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
| `context.max_tokens` | 4300 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `context.max_query_chars` | 4000 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `context.max_body_chars` | 4000 | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `generation.max_output_tokens` | 900 | file:configs/default.toml |
| `generation.max_retries` | 1 | file:configs/default.toml |
| `generation.temperature` | 0.0 | file:configs/default.toml |
| `generation.timeout_seconds` | 60.0 | file:configs/default.toml |
| `pipeline.name` | advanced-gpt5mini | cli |
| `pipeline.retriever` | question_doc_slots | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `pipeline.scope_gate` | llm | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `pipeline.answer_model` | gpt-5-mini | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `pipeline.controlled_version` | v2 | file:configs/default.toml |
| `pipeline.complete_questions` | True | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `pipeline.version_notes` | True | file:configs/pipelines/advanced-gpt5mini.toml ★ |
| `pipeline.generator` | controlled_v2 | file:configs/pipelines/advanced-gpt5mini.toml ★ |

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

Attempts: assessment by `gpt-5-mini` (controlled_v2), routed by `supported_answer_candidate`. Proposed outcome: **answer**. Raw model output:

```json
{
  "case_id": "DEV-16388",
  "main_need": "Explain and provide a code-level correction/workaround so deleting a keyed widget's session_state key does not leave the frontend showing a stale value (backend/frontend divergence).",
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
  "answer_candidate": {
    "message": "This behaviour matches documented widget semantics: deleting a widget key does not force the frontend to reset and the widget frontend can keep its previous value, which the backend may later re-adopt. The supported correction is to write the desired value into session_state (before the widget command runs) instead of deleting the key. Concretely, change the delete callback to assign the widget's default (or another intended value) rather than using del. Alternatively, use the documented temporary-key pattern to control what value the widget is initialized with. See [S1] and [S2].",
    "evidence_refs": [
      "S1",
      "S2"
    ]
  },
  "internal_investigation": null,
  "workflow_findings": [
    {
      "report_ref": "R3",
      "problem": "Deleting a keyed widget's entry from st.session_state inside a callback does not force the frontend widget to reset; the frontend keeps showing the old value and may later cause the backend to re-adopt that stale value.",
      "correction": "Replace the deletion with an explicit assignment of the intended default value in the callback. Example edit: def delete():\n    st.session_state['foo'] = 'default'\nThis ensures the session_state key exists with the desired value before the widget command runs on the rerun, keeping frontend and backend consistent.",
      "evidence_refs": [
        "S1",
        "S2"
      ]
    },
    {
      "report_ref": "R7",
      "problem": "The delete() callback runs as a prefix to the script rerun; deleting the key in that prefix leads to a race with frontend state synchronization.",
      "correction": "Make state changes that should determine a widget's value in the callback (prefix) so that when the script reruns the widget command sees the intended session_state value. In this app, set st.session_state['foo'] to the desired default in delete() rather than deleting the key.",
      "evidence_refs": [
        "S2"
      ]
    },
    {
      "report_ref": "R7",
      "problem": "If you want separate programmatic storage versus what the widget shows (and explicit control over initialization), deleting the widget key is not the right pattern.",
      "correction": "Use the documented temporary-key pattern: render the widget with a temporary key (e.g. '_foo') and copy values between '_foo' and 'foo' in your code so you can control when the permanent key is updated or cleared. See S1 for the example pattern.",
      "evidence_refs": [
        "S1"
      ]
    },
    {
      "report_ref": "R13",
      "problem": "The observed two-step divergence (backend returns default on the immediate rerun; frontend still shows old value; next rerun backend adopts stale frontend) is a manifestation of the documented widget-state synchronization rules rather than a missing call site in the sample app.",
      "correction": "Adopt the explicit-set pattern (assign default in callback) or the temp-key pattern above. If truly necessary to fully remove the key and force a frontend rebuild, avoid deleting the key in a callback and instead change app flow so the widget command is not called that run (which will cause a fresh widget to be created), or explicitly reinitialize via an app-level rerun path that recreates the widget identity.",
      "evidence_refs": [
        "S1",
        "S2"
      ]
    }
  ]
}
```

### 6. validator: `NoValidator`

Outcome kept: **answer**.
Violations: none.

### 7. summarizer: `build_summary`

```json
{
  "problem": "Deleting a widget's session_state key leaves the frontend showing the stale value (backend/frontend divergence for one rerun)",
  "outcome": "answer",
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
  "next_step": "Apply the proposed correction within its stated scope; it has not been executed by this controller.",
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

Outcome: **answer**

Reply to the reporter:

> This behaviour matches documented widget semantics: deleting a widget key does not force the frontend to reset and the widget frontend can keep its previous value, which the backend may later re-adopt. The supported correction is to write the desired value into session_state (before the widget command runs) instead of deleting the key. Concretely, change the delete callback to assign the widget's default (or another intended value) rather than using del. Alternatively, use the documented temporary-key pattern to control what value the widget is initialized with. See [S1] and [S2].
> 
> Required edits:
> - Replace the deletion with an explicit assignment of the intended default value in the callback. Example edit: def delete():
>     st.session_state['foo'] = 'default'
> This ensures the session_state key exists with the desired value before the widget command runs on the rerun, keeping frontend and backend consistent. [R3][S1][S2]
> - Make state changes that should determine a widget's value in the callback (prefix) so that when the script reruns the widget command sees the intended session_state value. In this app, set st.session_state['foo'] to the desired default in delete() rather than deleting the key. [R7][S2]
> - Use the documented temporary-key pattern: render the widget with a temporary key (e.g. '_foo') and copy values between '_foo' and 'foo' in your code so you can control when the permanent key is updated or cleared. See S1 for the example pattern. [R7][S1]
> - Adopt the explicit-set pattern (assign default in callback) or the temp-key pattern above. If truly necessary to fully remove the key and force a frontend rebuild, avoid deleting the key in a callback and instead change app flow so the widget command is not called that run (which will cause a fresh widget to be created), or explicitly reinitialize via an app-level rerun path that recreates the widget identity. [R13][S1][S2]

Citations: `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#12836-17083`, `doc:pinned:content/develop/concepts/architecture/widget-behavior.md#0-9535`

Model calls during this run:

| stage | type | mode | tokens in | tokens out | cost (USD) | status |
|---|---|---|---|---|---|---|
| scope_classify | chat | cache | 874 | 25 | 0.000000 | ok |
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| rerank | chat | cache | 3528 | 44 | 0.000000 | ok |
| answer | chat | cache | 8432 | 3021 | 0.000000 | ok |

Time: 0.150s

## Latest evaluation results

**full** run `e2e-versionA-gpt5mini` (2026-10-18T11:39:30+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
evidence   recall @1 0.21  @3 0.60  @6 0.71  @10 0.73   MRR 0.74
           answers cite gold evidence 1.00
decisions  outcome accuracy 0.73   (ambiguous 1.00  answerable 0.83  escalation 0.00)
grounding  citation violations 0.00   answers without citation 0.00   errors 0.00   budget stops 0
cost       $0.1238 (+$0.0088 replayed, not spent)   20 live calls, 45 cache hits   tokens in/out 195626/50952
time       mean 50.797s   p95 103.398s
```

**retrieval** run `retr-dev-version-B` (2026-10-18T11:25:02+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
       15 have gold evidence; 0 do not (excluded from recall)
evidence   recall @1 0.32  @3 0.60  @6 0.71  @10 0.73   MRR 0.89
           hit    @1 0.80  @3 1.00  @6 1.00  @10 1.00   gold in the prompt packet 0.73
passages   recall @1 0.31  @3 0.54  @6 0.67  @10 0.68   MRR 0.89
           hit    @1 0.80  @3 1.00  @6 1.00  @10 1.00   coverage@10 0.68   in the prompt packet 0.68
           ambiguous   recall@10 0.81  (n=6)
           answerable  recall@10 0.67  (n=6)
           escalation  recall@10 0.72  (n=3)
cost       $0.0070 (+$0.0016 replayed, not spent)   12 live calls, 45 cache hits   tokens in/out 54737/864
time       mean 8.679s   p95 22.070s
```
