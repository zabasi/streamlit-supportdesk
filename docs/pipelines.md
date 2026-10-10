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
