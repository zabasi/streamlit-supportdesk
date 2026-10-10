# Schemas

What each step reads and writes. Generated from the code by `scripts/gen_schema_docs.py`; do not
edit by hand (a test fails when this file is out of date).

| Step | Output | Type |
| --- | --- | --- |
| corpus file | one source per line of `corpus.jsonl` | [corpus record](#corpus-record) |
| index: load | one record per source, same shape for every type | [SourceRecord](#sourcerecord) |
| index: chunk | exact spans of the sources | [Chunk](#chunk) |
| index: store | `index/chunks.jsonl` (Chunk per line), `manifest.json`, `qdrant/` | [Chunk](#chunk) |
| support: intake | the report and the environment read from it | [CaseInput](#caseinput) |
| support: retrieve, rerank | ranked chunks | [Hit](#hit) |
| support: packet | the evidence the model sees, labelled S1, S2, ... | [Packet](#packet) |
| support: generate, validate | the model's decision, parsed and checked | [ParsedAnswer](#parsedanswer); model JSON: [model output](#model-output) |
| support: run | one record per report (`ask`, `eval`) | [CaseResult](#caseresult) |
| graph state | what LangGraph saves between nodes (plain data) | [SupportState](#supportstate), [CaseState](#casestate) |
| case workflow | the local tracker | [tracker tables](#tracker-tables) |
| evaluation | one evaluation case, inputs and evaluator-only labels | [EvalCase](#evalcase) |

## corpus record

One line of `corpus.jsonl`. Only `text` is embedded.

```json
{"source_id": "issue:3532",          // unique; issue:N, comment:N, doc:..., release:..., linked:...
 "source_type": "issue",              // issue | comment | documentation | release_section | linked_reference
 "text": "Title: ...",                // the prepared text; chunk offsets point into it
 "metadata": {"html_url": "...", "created_at": "...", "...": "..."},
 "text_sha256": "..."}                // sha256 of text; checked on load
```

## SourceRecord

One source, in the same shape whatever its type.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `source_id` | `str` | `required` |  |
| `source_type` | `str` | `required` |  |
| `title` | `str` | `required` |  |
| `text` | `str` | `required` |  |
| `url` | `str` | `required` |  |
| `parent_source_id` | `str \| None` | `None` |  |
| `issue_number` | `int \| None` | `None` |  |
| `linked_issue_numbers` | `frozenset[int]` | `frozenset()` | issues a linked reference was linked from |
| `version` | `str \| None` | `None` |  |
| `as_of` | `str \| None` | `None` | when it became public; None = undated (always visible) |
| `meta` | `dict` | `{}` |  |
| `text_sha256` | `str` | `''` | of ``text``, exactly as delivered |

## Chunk

A citable span of one source: ``text == corpus[source_id].text[start_char:end_char]`` (Python code-point offsets, end exclusive), checked when the index is built.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `chunk_id` | `str` | `required` |  |
| `source_id` | `str` | `required` |  |
| `source_type` | `str` | `required` | issue \| comment \| documentation \| release_section \| linked_reference |
| `title` | `str` | `required` |  |
| `text` | `str` | `required` |  |
| `url` | `str` | `''` |  |
| `parent_source_id` | `str \| None` | `None` |  |
| `issue_number` | `int \| None` | `None` |  |
| `version` | `str \| None` | `None` | Streamlit version, documentation commit or release version, if known |
| `as_of` | `str \| None` | `None` | ISO timestamp the content became public; None = undated |
| `meta` | `dict` | `{}` | state, labels, author_association, ... |
| `start_char` | `int \| None` | `None` | inclusive offset into the source text |
| `end_char` | `int \| None` | `None` | exclusive |
| `source_text_sha256` | `str \| None` | `None` | the parent source's text_sha256 |

## CaseInput

Everything the assistant is allowed to see at the start of a case.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case_id` | `str` | `required` |  |
| `title` | `str` | `required` |  |
| `body` | `str` | `required` |  |
| `created_at` | `str \| None` | `None` |  |
| `history` | `list[dict]` | `[]` | [{"role", "body", "created_at"}] |
| `environment` | `dict` | `{}` | known fields only; missing stays absent |

## Hit

Hit(chunk: 'Chunk', score: 'float', rank: 'int', source: 'str')

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `chunk` | `Chunk` | `required` |  |
| `score` | `float` | `required` |  |
| `rank` | `int` | `required` |  |
| `source` | `str` | `required` | dense \| bm25 \| hybrid |

## Packet

Packet(chunks: 'list[Chunk]' = <factory>, labels: 'dict[str, str]' = <factory>, token_count: 'int' = 0, dropped: 'list[dict]' = <factory>, sha256: 'str' = '')

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `chunks` | `list[Chunk]` | `[]` |  |
| `labels` | `dict[str, str]` | `{}` | label (S1, S2, ...) to chunk_id |
| `token_count` | `int` | `0` |  |
| `dropped` | `list[dict]` | `[]` |  |
| `sha256` | `str` | `''` |  |

## model output

The JSON the generate node asks the model for (`configs/prompts/answer_output_format.txt`); parsed into ParsedAnswer.

```
{
  "outcome": "answer" | "request_info" | "escalate",
  "confidence": "high" | "medium" | "low",
  "user_reply": "message to the reporter",
  "known_facts": [{"statement": "...", "evidence": ["S1"]}],
  "open_questions": ["..."],
  "next_step": "the single most useful next action",
  "citations": ["S1"]
}
```

## ParsedAnswer

ParsedAnswer(outcome: 'str', user_reply: 'str', known_facts: 'list[dict]' = <factory>, open_questions: 'list[str]' = <factory>, next_step: 'str | None' = None, citations: 'list[str]' = <factory>, violations: 'list[str]' = <factory>, attempts: 'list[dict]' = <factory>, raw_output: 'str' = '', confidence: 'str | None' = None)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `outcome` | `str` | `required` |  |
| `user_reply` | `str` | `required` |  |
| `known_facts` | `list[dict]` | `[]` |  |
| `open_questions` | `list[str]` | `[]` |  |
| `next_step` | `str \| None` | `None` |  |
| `citations` | `list[str]` | `[]` | chunk ids, all inside the packet |
| `violations` | `list[str]` | `[]` |  |
| `attempts` | `list[dict]` | `[]` | model-call log, filled by the generator |
| `raw_output` | `str` | `''` | the model's text as received, for inspection |
| `confidence` | `str \| None` | `None` | high \| medium \| low, as the model reported it |

## CaseResult

One uniform record for every pipeline, so baseline and final compare like for like.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case_id` | `str` | `required` |  |
| `pipeline` | `str` | `required` |  |
| `mode` | `str` | `required` |  |
| `outcome` | `str` | `required` | config.OUTCOMES or config.FAILURE_OUTCOMES |
| `confidence` | `str \| None` | `None` | the model's own: high \| medium \| low |
| `user_reply` | `str \| None` | `None` |  |
| `maintainer_summary` | `dict` | `{}` |  |
| `known_facts` | `list[dict]` | `[]` |  |
| `open_questions` | `list[str]` | `[]` |  |
| `next_step` | `str \| None` | `None` |  |
| `citations` | `list[str]` | `[]` | chunk_ids, all inside the packet |
| `violations` | `list[str]` | `[]` |  |
| `retrieved` | `list[dict]` | `[]` | rank-ordered {chunk_id, source_id, ...} |
| `packet_hash` | `str \| None` | `None` |  |
| `usage` | `dict` | `{}` |  |
| `error` | `str \| None` | `None` |  |
| `trace` | `dict` | `{}` |  |

## SupportState

dict() -> new empty dictionary dict(mapping) -> new dictionary initialized from a mapping object's     (key, value) pairs dict(iterable) -> new dictionary initialized as if via:     d = {}     for k, v in iterable:         d[k] = v dict(**kwargs) -> new dictionary initialized with the name=value pairs     in the keyword argument list.  For example:  dict(one=1, two=2)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case` | `dict` | optional | CaseInput |
| `env` | `dict` | optional | environment read from the report |
| `queries` | `list` | optional |  |
| `candidates` | `list` | optional | retriever output (Hit), the whole pool |
| `hits` | `list` | optional | after the reranker, top_k |
| `packet` | `dict` | optional | Packet: what the model sees |
| `generated` | `dict` | optional | ParsedAnswer as the generator returned it |
| `answer` | `dict` | optional | ParsedAnswer after validation |
| `summary` | `dict` | optional | maintainer summary |
| `extra` | `dict` | optional | data written by a pipeline's extra nodes (see StagedPipeline.extra_nodes) |

## CaseState

dict() -> new empty dictionary dict(mapping) -> new dictionary initialized from a mapping object's     (key, value) pairs dict(iterable) -> new dictionary initialized as if via:     d = {}     for k, v in iterable:         d[k] = v dict(**kwargs) -> new dictionary initialized with the name=value pairs     in the keyword argument list.  For example:  dict(one=1, two=2)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case` | `dict` | optional |  |
| `env` | `dict` | optional |  |
| `queries` | `list` | optional |  |
| `candidates` | `list` | optional |  |
| `hits` | `list` | optional |  |
| `packet` | `dict` | optional |  |
| `generated` | `dict` | optional |  |
| `answer` | `dict` | optional |  |
| `summary` | `dict` | optional |  |
| `extra` | `dict` | optional |  |
| `revision` | `int` | optional | the user's message count when this run started |
| `proposal` | `dict` | optional | the proposal waiting for review |
| `review` | `dict` | optional | the maintainer's decision |
| `action` | `dict` | optional | what was done (or why not) |

## tracker tables

The local tracker (`<artifacts_dir>/tracker.sqlite`) used by `supportdesk case`, `review` and `tracker`.

```sql
create table if not exists cases (case_id text primary key, title text not null,
status text not null, labels text not null default '[]', updated_at text not null);
create table if not exists messages (id integer primary key autoincrement, case_id text not null,
role text not null, body text not null, proposal_id text unique, created_at text not null);
create table if not exists notes (id integer primary key autoincrement, case_id text not null,
body text not null, proposal_id text unique, created_at text not null);
create table if not exists reviews (proposal_id text primary key, case_id text not null,
thread_id text not null, revision integer not null, status text not null, created_at text not null);
create table if not exists events (id integer primary key autoincrement, case_id text not null,
kind text not null, detail text not null, created_at text not null);
```

## EvalCase

EvalCase(case_id: 'str', family_id: 'str', split: 'str', category: 'str', input: 'CaseInput', gold_evidence: 'list[str]' = <factory>, acceptable_outcomes: 'list[str]' = <factory>, missing_info: 'list[str]' = <factory>, source_issue_number: 'int | None' = None, duplicate_issue_numbers: 'list[int]' = <factory>, notes: 'str' = '', kind: 'str' = 'issue', independent: 'bool' = True, evidence_spans: 'list[dict]' = <factory>, rubric: 'dict' = <factory>)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case_id` | `str` | `required` |  |
| `family_id` | `str` | `required` | duplicate reports share a family; a family is never split across dev and test |
| `split` | `str` | `required` | dev \| test |
| `category` | `str` | `required` | answerable \| ambiguous \| escalation |
| `input` | `CaseInput` | `required` |  |
| `gold_evidence` | `list[str]` | `[]` | source_ids acceptable as evidence |
| `acceptable_outcomes` | `list[str]` | `[]` |  |
| `missing_info` | `list[str]` | `[]` | facts a good reply should ask for |
| `source_issue_number` | `int \| None` | `None` | the real issue this case was derived from |
| `duplicate_issue_numbers` | `list[int]` | `[]` |  |
| `notes` | `str` | `''` |  |
| `kind` | `str` | `'issue'` | issue \| history_variant \| source_usage |
| `independent` | `bool` | `True` | False: a derivative of another case; not counted in sample size |
| `evidence_spans` | `list[dict]` | `[]` | exact gold spans (source_id, start/end_char) |
| `rubric` | `dict` | `{}` | evaluator-only; never shown to the system |
