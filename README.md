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
  missing facts decide the case, `escalate` when a maintainer must look. In the baseline one model
  call decides; in `advanced` the model writes a structured assessment and fixed Python rules choose
  the action (controlled routing, `docs/generation/README.md`).
- **`supportdesk case`** then acts: by default answers and questions are posted to the local tracker
  and escalations wait for a maintainer's approval. Use `--set case.review=all` so every reply waits
  for approval, as the project brief requires for any visible change.
  `ask`, `eval` and the web page stop after summarize and post nothing.
- **Pipelines:** `baseline` (dense search, 6 chunks, one `gpt-4o-mini` call) is the reference.
  `advanced` is the final design:
  - **scope check:** one `gpt-4o-mini` call first turns away reports that are not about Streamlit;
  - **retrieval:** the cleaned report is the query; dense search returns 15 candidates, 10 of them
    documentation, and the 15 documentation sections whose *symptom questions* (written once per
    section, offline) best match the report add a chunk each; one `gpt-4o-mini` call reranks them; the
    best 8 are kept (at least 5 documentation), each grown to its whole sections, within 4,300 tokens;
  - **answer:** `gpt-5.4-mini` writes a structured assessment and Python routes the action; each
    evidence piece carries **version notes** (what version it describes, compared with the reporter's,
    and how it may be used), and a question asks for **every** material missing fact at once.

  On the 15 development cases: 12 correct decisions (the baseline 5), every answer citing gold
  evidence, about $0.024 per case. The previous design (without the two answer-step fixes) is
  `candidate-6-doc2query15-rerank-section-controlled-v2`; the same final design with the cheaper
  `gpt-5-mini` is `advanced-notes-questions-gpt5mini` (11 correct, never escalates). Dozens of
  experiment pipelines change one thing at a time; `supportdesk pipelines` lists them all and
  `docs/pipelines.md` describes them.

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
5. **Symptom questions (for `advanced`):** `python scripts/build_section_questions.py --mode live`
   writes them once into `data/questions` (about $0.02). `advanced` stops with a clear message
   without them; rebuild them whenever the documentation in the corpus changes.
6. **Ask one question:** `supportdesk ask --mode live --title "..." --body "..."`. Commands refuse
   an index built with another embedding model, so an offline index has to be rebuilt first.
7. **Stay live:** to stop typing `--mode live`, add `SUPPORTDESK__RUN__MODE=live` to `.env`. Every
   command then runs live, and spends money, until you remove the line; `--mode offline` on a
   command still overrides it.

## 6. Commands

| Command | What it does |
|---|---|
| `supportdesk ask --title "..." --body "..."` | Answer one report; nothing is posted |
| `supportdesk web` | Local web page at http://127.0.0.1:8765: saved multi-turn conversations, the evidence the model saw, runs imported from `eval`, downloads |
| `supportdesk case CASE-1 --title "..." --body "..."` | Open a case (a follow-up is `--body` only) and act on the answer |
| `supportdesk review CASE-1` | Show what is waiting; decide with `--approve`, `--edit "TEXT"` or `--reject` |
| `supportdesk tracker CASE-1` | The case's messages, notes and events (`--reset` clears the tracker) |
| `supportdesk eval-retrieval --pipeline baseline --split dev` | Retrieval quality only: Recall@k, MRR; almost free |
| `supportdesk eval --pipeline baseline --split dev` | Full evaluation, one model call per case |
| `supportdesk eval --pipeline advanced --split test --final` | The test split: run once, at the very end |
| `supportdesk report RUN_ID RUN_ID` | Compare runs side by side (HTML) |
| `supportdesk snapshot --all` | Each pipeline's steps, settings and one traced run |
| `supportdesk pipelines` | Every registered pipeline |
| `python scripts/confusion_matrix.py RUN_ID ...` | Decision confusion matrices, as Markdown |
| `python scripts/export_cases.py RUN_ID` | One JSON file per case for review (`--no-reference` for blind reading) |
| `python scripts/judge_agreement.py MODEL_DIR HUMAN_DIR` | Agreement between the LLM judge and human reviews |
| `supportdesk check-cases` | Check the evaluation files against the corpus and index |
| `supportdesk config` | Every setting and where its value comes from |
| `make test` / `make demo` | All tests, offline / the full offline demo on the sample |

## Further reading

| File | For |
|---|---|
| `CLAUDE.md` | Adding a stage or node, and the rules all code follows |
| `docs/schemas.md` | The output of every step, field by field |
| `docs/pipelines.md` | Stages, pipelines, comparing changes |
| `docs/evaluation.md` | Evaluation files, metrics, case analysis, the test split |
| `docs/generation/README.md` | The answer step: controlled routing, its fixes and versions |
| `docs/evaluation/judge_prompt.md` | The LLM judge's criteria (the prompt is `configs/prompts/judge_case_system.txt`) |
| `docs/evaluation/human_review_form.md` | The form for human review of a case |
| `docs/snapshots/advanced.md` | A snapshot of the final design: stages, settings, one traced case |
| `docs/configuration.md` | Every setting |
| `docs/configuration.md` | Every setting |