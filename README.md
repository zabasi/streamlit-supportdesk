# streamlit supportdesk

A technical-support assistant for [Streamlit](https://streamlit.io). Given a user's problem report,
it searches the documentation, earlier GitHub issues and discussions, and release notes, then
answers with citations, asks the one question that would decide the case, or escalates to a
maintainer.

## 1. How it works

Two LangGraph graphs; every step is a node.

```
indexing (once):   load -> screen -> chunk -> audit -> embed -> store
per report:        intake -> query -> retrieve -> rerank -> packet -> generate -> validate -> summarize
```

- **generate** decides: `answer` when the evidence supports a confident answer, `request_info` when
  one missing fact decides the case, `escalate` when a maintainer must look. An answer the model
  rates low-confidence is escalated in code.
- **`supportdesk case`** then acts: answers and questions are posted to the local tracker;
  escalations wait for a maintainer's approval (`case.review = all` makes every reply wait).
  `ask`, `eval` and the web page stop after summarize and post nothing.
- **Pipelines:** `baseline` (dense search, 6 chunks) is the reference; `advanced` (hybrid search,
  10 chunks) is where improvements go. Experiments each change one thing in the baseline, so their
  effect can be measured alone: `clean_query` (the query without the issue template),
  `doc_slots` (3 of the 6 results kept for documentation), `hybrid_rrf` and `hybrid_split` (dense
  and keyword search, merged or side by side), `rewrite_llm` and `split_llm` (a model writes one or several search queries) and
  `split_sentences` (several queries without a model). Combinations build on the cleaned query:
  `clean_doc_slots`, `clean_hybrid_doc_slots`, `clean_split_doc_slots` and `clean_hybrid_split`.
  `supportdesk pipelines` lists them all.

## 2. Set up

Python 3.10 or newer.

```bash
python -m pip install -e ".[dev,live]"
make test                       # offline and free: checks the install works
cp .env.example .env            # then fill in OPENAI_API_KEY and OPENAI_BASE_URL (the Metis gateway)
```

`supportdesk` reads `./.env` on every run. The corpus goes in `data/processed/corpus.jsonl` and the
evaluation files in `data/eval/`. The corpus is not stored in git; it is distributed separately.

## 3. Configure

Settings live in `configs/default.toml`; each pipeline overrides some in
`configs/pipelines/<name>.toml`. Any setting can also be changed without editing those files, from
lowest to highest priority:

- a file named by `SUPPORTDESK_CONFIG`, or passed with `--config FILE`;
- `.env` or the environment, as `SUPPORTDESK__SECTION__KEY=value`;
- the command line, as `--set section.key=value`.

`supportdesk config` shows each value and where it came from; every setting is listed in
`docs/configuration.md`.

The mode, for example, is the setting `run.mode`: `offline` (free fake models, the default),
`replay` (recorded responses only) or `live` (real calls, which cost money). Set it in
`configs/default.toml` (`[run] mode = "live"`), in `.env` (`SUPPORTDESK__RUN__MODE=live`), or for
one command with `--mode live`.

## 4. Try it offline (free, no key)

```bash
export SUPPORTDESK_CONFIG=configs/sample.toml   # a small synthetic corpus
supportdesk index
supportdesk ask --title "st.session_state lost after refresh" --body "Values reset on F5"
supportdesk web                                 # the web page, at http://127.0.0.1:8765
```

Or run everything at once with `make demo`: it indexes the sample, evaluates both pipelines and
writes a comparison report (`artifacts_sample/reports/report.html`) and a snapshot of each pipeline
(`artifacts_sample/snapshots/`).

Offline mode uses fake models: everything runs, but the answer text comes from a template.

## 5. Go live

If you ran the offline try in this shell, leave the sample first: `unset SUPPORTDESK_CONFIG`.
Otherwise every command, live ones included, keeps using the sample corpus and `index_sample/`.

1. **Key:** `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env`.
2. **Prices:** put the real Metis prices in `configs/pricing.json` and set `"verified": true`. Live
   mode refuses to start until then, because the budget guard relies on them.
3. **Budget:** spending is tracked per machine and stops at `budget.ceiling_usd`. Set your limit in
   `.env`, for example `SUPPORTDESK__BUDGET__CEILING_USD=0.8`.
4. **Index once:** `supportdesk index --mode live` embeds the corpus with the real model (under
   one cent) and is also the first check that the key works. Share the `index/` folder: unpacked to
   `index/` on another machine, it is used as is, with no need to index again. Rebuild it when the
   corpus or the evaluation cases change.
5. **Ask one question:** `supportdesk ask --mode live --title "..." --body "..."`. Commands refuse
   an index built with another embedding model, so an offline index has to be rebuilt first.
6. **Stay live:** to stop typing `--mode live`, add `SUPPORTDESK__RUN__MODE=live` to `.env`. Every
   command then runs live, and spends money, until you remove the line; `--mode offline` on a
   command still overrides it.

## 6. Commands

| Command | What it does |
|---|---|
| `supportdesk ask --title "..." --body "..."` | Answer one report; nothing is posted |
| `supportdesk web` | Local web page at http://127.0.0.1:8765 |
| `supportdesk case CASE-1 --title "..." --body "..."` | Open a case (a follow-up is `--body` only) and act on the answer |
| `supportdesk review CASE-1` | Show what is waiting; decide with `--approve`, `--edit "TEXT"` or `--reject` |
| `supportdesk tracker CASE-1` | The case's messages, notes and events (`--reset` clears the tracker) |
| `supportdesk eval-retrieval --pipeline baseline --split dev` | Retrieval quality only: Recall@k, MRR; almost free |
| `supportdesk eval --pipeline baseline --split dev` | Full evaluation, one model call per case |
| `supportdesk report RUN_ID RUN_ID` | Compare runs side by side (HTML) |
| `supportdesk snapshot --all` | Each pipeline's steps, settings and one traced run |
| `supportdesk check-cases` | Check the evaluation files against the corpus and index |
| `supportdesk config` | Every setting and where its value comes from |
| `make test` / `make demo` | All tests, offline / the full offline demo on the sample |

## Further reading

| File | For |
|---|---|
| `CLAUDE.md` | Adding a stage or node, and the rules all code follows |
| `docs/schemas.md` | The output of every step, field by field |
| `docs/pipelines.md` | Stages, pipelines, comparing changes |
| `docs/evaluation.md` | Evaluation files, metrics, the test split |
| `docs/configuration.md` | Every setting |