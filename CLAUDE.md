# CLAUDE.md

Guide for anyone (person or coding assistant) changing this code: where things live, how to add a
module, and the rules every change follows. Setup and commands are in `README.md`; the output of
every step is in `docs/schemas.md`.

## Verify every change

```bash
make test                                   # all tests, offline and free; must pass
ruff check src tests scripts
python scripts/gen_config_docs.py           # after adding or changing a setting
python scripts/gen_schema_docs.py           # after changing a type in docs/schemas.md
```

## Where things live

| Part | File |
|---|---|
| Indexing graph: load, screen, chunk, audit, embed, store | `src/supportdesk/indexing.py` |
| Support graph: intake ... summarize; extra nodes | `src/supportdesk/pipelines/staged.py` |
| Stage interfaces / baseline implementations | `pipelines/stages/protocols.py` / `pipelines/stages/defaults.py` |
| The two pipelines | `pipelines/baseline.py`, `pipelines/advanced.py`, listed in `registry.py` |
| Case workflow: routing, human review, local tracker | `src/supportdesk/casework.py` |
| Chunkers | `data/chunking.py` (`CHUNKERS`) |
| Retrievers | `retrieval/retriever.py`, listed in `registry.py` (`RETRIEVERS`) |
| Model calls, budget, cache, fake models | `src/supportdesk/llm/` |
| Settings | `config.py` (types and defaults), `configs/default.toml`, `configs/pipelines/*.toml` |
| Prompts | `configs/prompts/*.txt` |
| Evaluation files, metrics, leakage | `src/supportdesk/eval/` |

## Adding a module

**A stage implementation** (query, reranker, packet, generator, validator, summarizer). Write a class
in its own file under `src/supportdesk/pipelines/stages/`, matching the signature in
`stages/protocols.py`, with a one-line docstring (snapshots and reports show it). Use it in
`pipelines/advanced.py`: `stages["reranker"] = LLMReranker(clients, settings)`.

**A new node** (for example grade evidence, rewrite and retrieve again). Write `fn(state) -> dict`: it reads
the plain-data state and returns only the keys it changes, existing ones (`hits`, `packet`, ...) or
its own data under `extra`. Add it in `pipelines/advanced.py`:
`extra_nodes = [("grade_evidence", grade_evidence, "rerank")]`. Loops (conditional edges) need a
change to the graph builder in `staged.py`; keep the baseline's graph unchanged.

**A chunker.** Add a class to `data/chunking.py`, register it in `CHUNKERS` and its name in
`config.CHUNKING_METHODS`, then set `chunking.method` and its own `paths.index_dir` in the pipeline's
profile. Every chunk must satisfy `corpus[source_id].text[start_char:end_char] == chunk.text`;
`check_chunks` enforces it.

**A retriever.** A class with `search(query, *, k, as_of) -> list[Hit]` in `retrieval/`, one line in
`RETRIEVERS` (`registry.py`); select it with `pipeline.retriever` in a profile.

**A pipeline.** A `build_<name>(retriever, clients, settings)` function, one line in `PIPELINES`,
and a profile `configs/pipelines/<name>.toml`.

**A setting.** A field with `_opt(help, default)` on the right section in `config.py`, the same key
in `configs/default.toml`, then `python scripts/gen_config_docs.py`.

## Rules

1. **The baseline does not change by accident.** `tests/test_golden_baseline.py` pins its exact
   output. Change `advanced` instead. If a change to the baseline is intended, regenerate with
   `PYTHONPATH=src:. python -m tests.golden.generate` and check that only the fields you meant changed.
2. **Every model call goes through `clients`** (`clients.chat.complete(..., stage="name")`,
   `clients.embedding.embed(..., stage="name")`), so the budget guard, cache and cost ledger see it.
   Never create OpenAI or LangChain model clients directly; use the adapters in
   `integrations/langchain.py` if a LangChain component is needed.
3. **No numbers in code.** Sizes, limits, thresholds and model names are settings.
4. **Graph state is plain data** (dicts, lists, strings, numbers), so runs can be checkpointed and
   resumed. Convert to objects inside a node, not in the state.
5. **Corpus text is never modified.** Chunk offsets point into it; the screen node refuses a corpus
   with personal data instead of editing it.
6. **Evaluation material never reaches the index or the prompt.** Rubrics are evaluator-only; the
   index excludes every evaluation issue (`supportdesk index` checks); the test split runs only
   with `--final`, after development is finished.
7. **Only the case workflow writes to the tracker,** and only through its rules: a post is keyed by
   its proposal id (never twice), a stale proposal posts nothing, an approval names one proposal.
8. **Don't catch gateway errors in stages.** The pipeline turns them into failure outcomes.
9. **Tests are offline.** Fake models (`llm/fake.py`) and scripted answers (`clients.chat.script`);
   no test may need a key or the network. Every change comes with a test.
10. **Spend carefully.** The API budget is small. Measure retrieval with `eval-retrieval` (almost
    free) before running `eval`; reuse runs through the cache.
11. **Secrets stay out:** keys only in `.env` (git-ignored), never in code, settings files, logs or
    commits.
12. **Docs describe the current code only:** what it does and how to use it, not how it came to be.
