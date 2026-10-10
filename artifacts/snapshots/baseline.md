# Pipeline snapshot: baseline

| | |
|---|---|
| Taken | 2026-10-10T08:07:29+00:00 |
| Profile | configs/pipelines/baseline.toml |
| Mode | live |
| Retriever | `dense` |
| Models | chat `gpt-4o-mini`, embedding `text-embedding-3-small` |
| Config fingerprint | `7f4782abadd71892` |

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
| `models.embedding` | text-embedding-3-small | file:configs/default.toml |
| `models.base_url` | (unset) | default |
| `models.api_key_env` | OPENAI_API_KEY | file:configs/default.toml |
| `prompts.system` | configs/prompts/answer_system.txt | file:configs/default.toml |
| `prompts.output_format` | configs/prompts/answer_output_format.txt | file:configs/default.toml |
| `chunking.method` | fixed | file:configs/pipelines/baseline.toml ★ |
| `chunking.max_tokens` | 512 | file:configs/pipelines/baseline.toml ★ |
| `chunking.overlap_tokens` | 128 | file:configs/pipelines/baseline.toml ★ |
| `retrieval.top_k` | 6 | file:configs/pipelines/baseline.toml ★ |
| `retrieval.candidate_k` | (unset) | default |
| `retrieval.historical_filter` | True | file:configs/pipelines/baseline.toml ★ |
| `retrieval.embed_batch_size` | 64 | file:configs/default.toml |
| `retrieval.bm25_k1` | 1.5 | file:configs/default.toml |
| `retrieval.bm25_b` | 0.75 | file:configs/default.toml |
| `retrieval.hybrid_pool` | 30 | file:configs/default.toml |
| `retrieval.rrf_k` | 60 | file:configs/default.toml |
| `context.max_chunks` | (unset) | default |
| `context.max_tokens` | 3200 | file:configs/pipelines/baseline.toml ★ |
| `context.max_query_chars` | 4000 | file:configs/pipelines/baseline.toml ★ |
| `context.max_body_chars` | 4000 | file:configs/pipelines/baseline.toml ★ |
| `generation.max_output_tokens` | 900 | file:configs/default.toml |
| `generation.max_retries` | 1 | file:configs/default.toml |
| `generation.temperature` | 0.0 | file:configs/default.toml |
| `generation.timeout_seconds` | 60.0 | file:configs/default.toml |

Derived: retrieval pool 6, chunks allowed in the prompt 6.

## Walkthrough of one case

**text_area inside an expander doesn't load from session state **

> ### Checklist ### Summary This is a weird one but quite pesky! ### Reproducible Code Example ```Python import streamlit as st from time import sleep if not st.session_state: st.session_state.update({"text": "Hello, World!"}) if st.button("Press before expanding"): with st.spinner(): sleep(1) with st.expander("Text Field inside"): st.text_area("Text", key="text") st.write(st.session_state) ``` ### Steps To Reproduce 1. Run this code 2. Press the button 3. Wait... 4. Expand Text Field ### Expected Behavior The text field is prefilled with "Hello World" If you DO NOT press the button before expa…

Gold evidence for this case: `doc:pinned:content/develop/api-reference/caching-and-state/session_state.md`, `doc:pinned:content/develop/concepts/architecture/widget-behavior.md`

### Environment read from the report

browser = `chrome`, operating_system = `mac`, python_version = `3.10.5`, streamlit_version = `1.13.0`

### 1. query: `PlainQuery`

- text_area inside an expander doesn't load from session state ### Checklist ### Summary This is a weird one but quite pesky! ### Reproducible Code Example ```Python import streamlit as st from time import sleep if not st.session_state: st.session_state.update({"text": "Hello, World!"}) if st.button(…

### 2. retriever: `DenseRetriever`

6 candidates (pool 6):

| rank | score | source | type | chunk | gold |
|---|---|---|---|---|---|
| 1 | 0.7049 | issue:4651 | issue | `issue:4651:000` |  |
| 2 | 0.6517 | issue:7084 | issue | `issue:7084:000` |  |
| 3 | 0.6480 | issue:10007 | issue | `issue:10007:000` |  |
| 4 | 0.6479 | issue:6370 | issue | `issue:6370:000` |  |
| 5 | 0.6396 | issue:7862 | issue | `issue:7862:000` |  |
| 6 | 0.6294 | issue:7087 | issue | `issue:7087:000` |  |

### 3. reranker: `NoRerank`

Order unchanged; kept the top 6.

### 4. packet: `DefaultPacketBuilder`

6 chunks, 1601 tokens shown to the model:

| label | chunk | tokens |
|---|---|---|
| S1 | `issue:4651:000` | 249 |
| S2 | `issue:7084:000` | 512 |
| S3 | `issue:10007:000` | 192 |
| S4 | `issue:6370:000` | 134 |
| S5 | `issue:7862:000` | 224 |
| S6 | `issue:7087:000` | 290 |

### 5. generator: `SingleCallGenerator`

Attempts: ok. Proposed outcome: **request_info**. Raw model output:

```json
{
  "outcome": "request_info",
  "confidence": "medium",
  "user_reply": "It seems that the issue you're experiencing may be related to how the session state is updated when a blocking operation occurs. Can you confirm if the button press is causing any blocking operation (like sleep or a long-running task) before the expander is opened? This information will help narrow down the cause of the problem.",
  "known_facts": [
    {
      "statement": "User inputs on st.text_area are not saved in st.session_state when blocking happens after a button click.",
      "evidence": [
        "S5"
      ]
    }
  ],
  "open_questions": [
    "Is the button press causing any blocking operation before the expander is opened?"
  ],
  "next_step": "Please check if there is any blocking operation occurring after the button press.",
  "citations": []
}
```

### 6. validator: `NoValidator`

Outcome kept: **request_info**.
Violations: none.

### 7. summarizer: `build_summary`

```json
{
  "problem": "text_area inside an expander doesn't load from session state ",
  "outcome": "request_info",
  "environment": {
    "known": {
      "streamlit_version": "1.13.0",
      "python_version": "3.10.5",
      "browser": "chrome",
      "operating_system": "mac"
    },
    "unknown": []
  },
  "known_facts": [
    {
      "statement": "User inputs on st.text_area are not saved in st.session_state when blocking happens after a button click.",
      "evidence": [
        "issue:7862:000"
      ]
    }
  ],
  "open_questions": [
    "Is the button press causing any blocking operation before the expander is opened?"
  ],
  "next_step": "Please check if there is any blocking operation occurring after the button press.",
  "sources": [
    {
      "chunk_id": "issue:7862:000",
      "type": "issue",
      "url": "https://github.com/streamlit/streamlit/issues/7862",
      "version": "- 1.29.0"
    }
  ]
}
```

### Result

Outcome: **request_info**

Reply to the reporter:

> It seems that the issue you're experiencing may be related to how the session state is updated when a blocking operation occurs. Can you confirm if the button press is causing any blocking operation (like sleep or a long-running task) before the expander is opened? This information will help narrow down the cause of the problem.

Citations: none

Model calls during this run:

| stage | type | mode | tokens in | tokens out | cost (USD) | status |
|---|---|---|---|---|---|---|
| query | embedding | cache | 0 | 0 | 0.000000 | ok |
| answer | chat | cache | 4597 | 186 | 0.000000 | ok |

Time: 0.024s

## Latest evaluation results

**full** run `baseline-dev-20261010-080355` (2026-10-10T08:04:50+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
evidence   recall @1 0.00  @3 0.00  @6 0.03  @10 0.03   MRR 0.01
           answers cite gold evidence 0.25
decisions  outcome accuracy 0.33   (ambiguous 0.33  answerable 0.17  escalation 0.67)
grounding  citation violations 0.00   answers without citation 0.00   errors 0.00   budget stops 0
cost       $0.0116   15 live calls, 15 cache hits   tokens in/out 66446/2751
time       mean 3.638s   p95 8.564s
```

**retrieval** run `baseline-retrieval-dev-20261010-080126` (2026-10-10T08:01:56+00:00, mode live):

```
cases: 15 (ambiguous 6, answerable 6, escalation 3)
       15 have gold evidence; 0 do not (excluded from recall)
evidence   recall @1 0.00  @3 0.00  @6 0.03  @10 0.03   MRR 0.01
           hit    @1 0.00  @3 0.00  @6 0.07  @10 0.07   gold in the prompt packet 0.03
           ambiguous   recall@10 0.00  (n=6)
           answerable  recall@10 0.08  (n=6)
           escalation  recall@10 0.00  (n=3)
cost       $0.0001   15 live calls, 0 cache hits   tokens in/out 7110/0
time       mean 1.935s   p95 10.851s
```
