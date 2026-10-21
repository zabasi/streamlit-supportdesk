# Data gathering

How the corpus in `data/processed/corpus.jsonl` was collected and prepared, kept with its code, the
frozen v06 dataset it came from, and its checks. Nothing else in the repository reads from this folder:
the system uses `data/processed/corpus.jsonl` and `data/eval/`. Exporting this folder's retrieval corpus
reproduces `data/processed/corpus.jsonl` byte for byte.

## Contents

| Path | What it is |
|---|---|
| `acquisition/scripts/` | The original collection code: downloaders for GitHub issues and comments, the documentation and the release notes, and the early normalization, unchanged |
| `acquisition/run_collection.py` | Runs the four downloaders and the early normalization in the layout they expect. By default it only prints the plan; `--run` downloads |
| `acquisition/.env.example` | Example settings for a new collection (search scope, target count). No token; put a real one only in `acquisition/.env`, which git ignores |
| `frozen_v06/` | The reviewed v06 dataset: the five retrieval source files, source relationships, review settings, the versioned preparation scripts, archives of each stage, exact-span checks and the original evaluation files |
| `export_corpus.py` | Writes the five retrieval files as one corpus JSONL, checking ids and text hashes |
| `requirements-acquisition.txt` | Python packages the downloaders need |
| `provenance.json`, `verification.json`, `file_manifest.json` | Where the package came from, the results of its checks, and the SHA-256 of every file |

## Collection and preparation are two stages

The collection code downloads public sources and writes them in an early schema
(`acquisition/data/raw/*.json` and `acquisition/data/corpus/corpus.json`, both ignored by git). It does
not reproduce the reviewed v06 dataset: GitHub results change over time, and a new download needs source
review, preparation, and the evaluation cases' issues kept out before it can be indexed. The v06
preparation scripts instead rebuild the reviewed dataset from the v05 baseline and the review settings
included in `frozen_v06/`.

`acquisition/scripts/run_pipeline.py` is the original orchestration script, kept unchanged; it expects a
`scripts/validate_dataset.py` that was never part of the collection code, so use `run_collection.py`
instead. The v06 validator checks v06 data only, not the early collection schema.

## Verify (offline)

From the repository root, with Python 3.10 or newer; no network, embeddings or model calls:

```console
python -X utf8 data_gathering/frozen_v06/scripts/validate_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/test_v06_contracts.py
python -X utf8 data_gathering/frozen_v06/scripts/load_retrieval_sources.py
python -X utf8 data_gathering/acquisition/run_collection.py
```

The validator checks the 540 records, the exact evidence spans and the evaluation split; the contract
tests use fixed fixture responses, not model answers; the last command prints the collection plan
without downloading anything. To check every file against its recorded hash:

```console
cd data_gathering && python3 -c "import json, hashlib, pathlib; m = json.load(open('file_manifest.json')); print([p for p, h in m.items() if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() != h] or 'all files match')"
```

Every file is kept byte for byte (`.gitattributes` marks the folder as binary, so git never converts line
endings). After a deliberate change to a file, update its hash in `file_manifest.json`.

## Export the corpus

Only these five files are retrieval sources:

```text
frozen_v06/retrieval_corpus/issues.jsonl
frozen_v06/retrieval_corpus/comments.jsonl
frozen_v06/retrieval_corpus/documents.jsonl
frozen_v06/retrieval_corpus/release_sections.jsonl
frozen_v06/retrieval_corpus/linked_references.jsonl
```

```console
python -X utf8 data_gathering/export_corpus.py --output data/processed/corpus.jsonl
```

When the output already holds the same records, the exporter leaves it untouched; different data is an
error, never overwritten. The current `data/processed/corpus.jsonl` holds exactly these 540 records: 126
issues, 214 comments, 54 documentation records, 111 release sections and 35 linked references. They are
source records; the indexing step cuts them into chunks.

## What may be indexed

Only the five retrieval files. Never index this folder recursively: `frozen_v06/evaluator_only/`,
`agent_inputs/`, `audit/`, `config/`, rubrics, held-out comments and reports are not source text, and
indexing them would leak evaluation material. Keep each record's identity, version and usage restrictions
as metadata when chunking.

## Rebuild the v06 preparation

The builder rewrites files in `frozen_v06/`, so run it on a separate copy of the folder:

```console
python -X utf8 data_gathering/frozen_v06/scripts/build_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/validate_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/finalize_v06.py
```

The current scripts have plain names, apart from `finalize_v06.py` and `test_v06_contracts.py`; scripts
with a version suffix are earlier stages, kept for the record. Paths that appear in some archived scripts
are from the machines they first ran on, not instructions. `verification.json` records the result of an
offline rebuild in a separate copy.

## A new collection

Downloading uses only public sources (the GitHub API, the documentation repository at the pinned commit,
the release notes); it makes no model or embedding calls.

```console
python -m pip install -r data_gathering/requirements-acquisition.txt
cp data_gathering/acquisition/.env.example data_gathering/acquisition/.env   # then set the token and the scope
python -X utf8 data_gathering/acquisition/run_collection.py --run
```

The downloaders follow GitHub pagination and skip pull requests. The example scope in `.env.example` is
illustrative; the original run's settings were not kept. A new collection produces raw data in the early
schema, which still needs the review and preparation described above before it can replace the corpus.

## Evaluation files in this folder

`frozen_v06/agent_inputs/` and `frozen_v06/evaluator_only/` hold the 15 development and 15 test cases as
first prepared, with their original rubrics. The evaluation files the system uses are in `data/eval/`,
which includes later revisions of the rubrics and the annotated evidence; the files here are the record of
the original version and do not replace them.
