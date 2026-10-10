"""Write docs/schemas.md from the code's own types: the output of every step, field by field.
`python scripts/gen_schema_docs.py` rewrites it; tests/test_docs.py checks it is up to date."""
from __future__ import annotations

import dataclasses
import inspect
import re
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportdesk.casework import CaseState, Tracker
from supportdesk.data.corpus import SourceRecord
from supportdesk.eval.cases import EvalCase
from supportdesk.pipelines.packet import Packet
from supportdesk.pipelines.parsing import ParsedAnswer
from supportdesk.pipelines.state import SupportState
from supportdesk.retrieval.retriever import Hit
from supportdesk.types import CaseInput, CaseResult, Chunk

INTRO = """# Schemas

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
"""

CORPUS_RECORD = """## corpus record

One line of `corpus.jsonl`. Only `text` is embedded.

```json
{"source_id": "issue:3532",          // unique; issue:N, comment:N, doc:..., release:..., linked:...
 "source_type": "issue",              // issue | comment | documentation | release_section | linked_reference
 "text": "Title: ...",                // the prepared text; chunk offsets point into it
 "metadata": {"html_url": "...", "created_at": "...", "...": "..."},
 "text_sha256": "..."}                // sha256 of text; checked on load
```
"""


def _comments(cls) -> dict[str, str]:
    try:
        lines = inspect.getsource(cls).splitlines()
    except (OSError, TypeError):
        return {}
    found = {}
    for line in lines:
        match = re.match(r"\s+(\w+)\s*:[^#]*#\s*(.+)$", line)
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def _type(hint) -> str:
    text = hint if isinstance(hint, str) else getattr(hint, "__name__", None) or str(hint)
    return text.replace("typing.", "").replace("NoneType", "None")


def _row(name: str, kind: str, default: str, note: str) -> str:
    """One table row; `|` inside a cell is escaped so it does not split the column."""
    cell = lambda text: text.replace("|", "\\|")
    shown = default if default == "optional" else f"`{cell(default)}`"
    return f"| `{name}` | `{cell(kind)}` | {shown} | {cell(note)} |"


def _section(cls) -> str:
    doc = (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
    notes = _comments(cls)
    rows = ["| Field | Type | Default | Notes |", "| --- | --- | --- | --- |"]
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                default = repr(f.default_factory())
            else:
                default = "required"
            rows.append(_row(f.name, _type(f.type), default, notes.get(f.name, "")))
    else:  # TypedDict
        for name, hint in typing.get_type_hints(cls).items():
            rows.append(_row(name, _type(hint), "optional", notes.get(name, "")))
    return f"## {cls.__name__}\n\n{doc}\n\n" + "\n".join(rows) + "\n"


def model_output() -> str:
    path = ROOT / "configs" / "prompts" / "answer_output_format.txt"
    return ("## model output\n\nThe JSON the generate node asks the model for "
            "(`configs/prompts/answer_output_format.txt`); parsed into ParsedAnswer.\n\n"
            f"```\n{path.read_text(encoding='utf-8').strip()}\n```\n")


def tracker_tables() -> str:
    sql = "\n".join(line.strip() for line in Tracker.SCHEMA.strip().splitlines())
    return ("## tracker tables\n\nThe local tracker (`<artifacts_dir>/tracker.sqlite`) used by "
            f"`supportdesk case`, `review` and `tracker`.\n\n```sql\n{sql}\n```\n")


def render() -> str:
    parts = [INTRO, CORPUS_RECORD, _section(SourceRecord), _section(Chunk), _section(CaseInput),
             _section(Hit), _section(Packet), model_output(), _section(ParsedAnswer),
             _section(CaseResult), _section(SupportState), _section(CaseState), tracker_tables(),
             _section(EvalCase)]
    return "\n".join(parts)


if __name__ == "__main__":
    (ROOT / "docs" / "schemas.md").write_text(render(), encoding="utf-8")
    print(f"wrote {ROOT / 'docs' / 'schemas.md'}")
