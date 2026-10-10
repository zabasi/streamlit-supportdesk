# supportdesk

A technical-support assistant for [Streamlit](https://streamlit.io). Given a user's problem report,
it searches the documentation, earlier GitHub issues and discussions, and release notes, then
answers with citations, asks the one question that would decide the case, or escalates to a
maintainer.

## 1. Set up

Python 3.10 or newer.

```bash
python -m pip install -e ".[dev,live]"
cp .env.example .env            # then fill in OPENAI_API_KEY and OPENAI_BASE_URL (the Metis gateway)
```

`supportdesk` reads `./.env` on every run. The corpus goes in `data/processed/corpus.jsonl` and the
evaluation files in `data/eval/`.

## 2. Try it offline (free, no key)

```bash
export SUPPORTDESK_CONFIG=configs/sample.toml   # a small synthetic corpus
supportdesk index
supportdesk ask --title "st.session_state lost after refresh" --body "Values reset on F5"
make test
```

Offline mode uses fake models: everything runs, but the answer text comes from a template.

## 3. Go live

1. **Key:** `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env`.
2. **Prices:** put the real Metis prices in `configs/pricing.json` and set `"verified": true`. Live
   mode refuses to start until then, because the budget guard relies on them.
3. **Budget:** spending is tracked per machine and stops at `budget.ceiling_usd`. Set your limit in
   `.env`, for example `SUPPORTDESK__BUDGET__CEILING_USD=0.8`.
4. **Check one call:** `supportdesk ask --mode live --title "..." --body "..."`
5. **Index once:** `supportdesk index --mode live` (under one cent), then share the `index/` folder.
   Unpacked to `index/` on another machine, it is used as is; there is no need to index again. Rebuild when the corpus or the
   evaluation cases change.

Add `--mode live` to any command, or put `SUPPORTDESK__RUN__MODE=live` in `.env`.

## 4. Use it

| Command | What it does |
|---|---|
| `supportdesk ask --title "..." --body "..."` | Answer one report; nothing is posted |
| `supportdesk case CASE-1 --title "..." --body "..."` | Open a case (a follow-up is `--body` only) and act on the answer |
| `supportdesk review CASE-1` | Show what is waiting; decide with `--approve`, `--edit "TEXT"` or `--reject` |
| `supportdesk tracker CASE-1` | The case's messages, notes and events (`--reset` clears the tracker) |
| `supportdesk eval-retrieval --pipeline baseline --split dev` | Retrieval quality only: Recall@k, MRR; almost free |
| `supportdesk eval --pipeline baseline --split dev` | Full evaluation, one model call per case |
| `supportdesk report RUN_ID RUN_ID` | Compare runs side by side (HTML) |
| `supportdesk snapshot --all` | Each pipeline's steps, settings and one traced run |
| `supportdesk check-cases` | Check the evaluation files against the corpus and index |
| `supportdesk web` | Local web page at http://127.0.0.1:8765 |
| `supportdesk config` | Every setting and where its value comes from |

## 5. Configure

Settings live in `configs/default.toml`; each pipeline overrides some in
`configs/pipelines/<name>.toml`. Anything can be overridden for one run with
`--set section.key=value`, or in `.env` as `SUPPORTDESK__SECTION__KEY=value`.
`supportdesk config` shows each value and where it came from; every setting is listed in
`docs/configuration.md`.

## 6. How it works

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
  10 chunks) is where improvements go.

## Further reading

| File | For |
|---|---|
| `CLAUDE.md` | Adding a stage or node, and the rules all code follows |
| `docs/schemas.md` | The output of every step, field by field |
| `docs/pipelines.md` | Stages, pipelines, comparing changes |
| `docs/evaluation.md` | Evaluation files, metrics, the test split |
| `docs/configuration.md` | Every setting |
