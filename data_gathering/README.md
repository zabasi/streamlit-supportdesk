# Data gathering and preparation handoff

Place the `data_gathering/` directory at the root of `streamlit-supportdesk`.
This is an additive package: it does not replace the team's pipeline, current
development/test revisions, credentials, indexes or generation code.

## Contents and responsibilities

- `acquisition/scripts/`: recovered original acquisition and early normalization
  scripts, preserved byte for byte from the source package's
  `audit/colleague_scripts/`. Includes issue/comment/document/release downloaders.
- `acquisition/run_collection.py`: new handoff entrypoint that runs the four
  downloaders and early corpus normalization in their expected directory layout.
  The default is a plan only. Internet access requires `--run`.
- `acquisition/.env.example`: a new example configuration, not recovered historical
  settings. Contains no token. Put a real token only in an untracked `.env`.
- `frozen_v06/`: the recovered immutable delivery with five retrieval source files,
  source relationships, review configuration, versioned preparation scripts,
  raw/evaluation provenance archives, exact-span checks and separate evaluators.
- `export_corpus.py`: exports the five explicit retrieval files to one JSONL, checks
  IDs/text hashes, and refuses to overwrite different existing corpus data.
- `requirements-acquisition.txt`: dependencies used by the historical downloaders.
- `provenance.json`, `file_manifest.json`, `verification.json`: recovery identity,
  package checksums and checks actually performed for this handoff.

## Important distinction: new collection versus the frozen delivery

The early collector emits `acquisition/data/raw/*.json` and
`acquisition/data/corpus/corpus.json`, using the EARLY `id`/`text`-preparation schema.
It does not automatically recreate the reviewed v06 corpus or its held-out split.
Fresh API results can change and need source review, preparation and evaluation
quarantine before indexing. The shipped v06 builder instead reproduces the reviewed
delivery from its included immutable v05 baseline and review configuration.

The original `acquisition/scripts/run_pipeline.py` is preserved as historical code.
It references an early `scripts/validate_dataset.py` not present beside the recovered
collector scripts. Do not use that historical orchestration entrypoint unchanged.
Use the new `run_collection.py` for acquisition and the v06 validator for v06 data.
The v06 validator is not a validator for the early raw-collection schema.

## Offline verification (no network, embedding or model calls)

From the repository root, with Python 3.10 or newer:

```console
python -X utf8 data_gathering/frozen_v06/scripts/validate_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/test_v06_contracts.py
python -X utf8 data_gathering/frozen_v06/scripts/load_retrieval_sources.py
python -X utf8 data_gathering/acquisition/run_collection.py
```

The tests use explicit fixture responses; they are not agent-quality measurements.

## Rebuild preparation from the retained baseline

Run these only on a working copy because the builder regenerates delivery files:

```console
python -X utf8 data_gathering/frozen_v06/scripts/build_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/validate_dataset.py
python -X utf8 data_gathering/frozen_v06/scripts/finalize_v06.py
```

Active entrypoints have unsuffixed names, except `finalize_v06.py` and
`test_v06_contracts.py`. Older suffixed builders/validators are retained history.
The initial scratch paths visible in some archived code are not portable acquisition
instructions. The active bundled v06 builder uses its local baseline when run from
the existing `frozen_v06/scripts/` directory. Offline rebuild was checked in an
isolated copy; see `verification.json` for the actual result.

## Start a NEW collection (not run while preparing this handoff)

```console
python -m pip install -r data_gathering/requirements-acquisition.txt
```

Copy `acquisition/.env.example` to `acquisition/.env`, configure the desired scope,
then:

```console
python -X utf8 data_gathering/acquisition/run_collection.py --run
```

The original code follows GitHub pagination and skips pull requests in the issue
selection. Documentation/release acquisition uses the pinned docs commit in the
configuration. The example search terms and target issue count are a new example,
not a claim to recover the exact historical `.env`. The script-preserved API
configuration was not live-tested for this handoff. Public-source downloads do not
use the team's paid LLM/embedding API.

## Connect the existing frozen corpus

Only these five files may supply retrieval sources:

```text
frozen_v06/retrieval_corpus/issues.jsonl
frozen_v06/retrieval_corpus/comments.jsonl
frozen_v06/retrieval_corpus/documents.jsonl
frozen_v06/retrieval_corpus/release_sections.jsonl
frozen_v06/retrieval_corpus/linked_references.jsonl
```

Optional merge into the repository's existing corpus path:

```console
python -X utf8 data_gathering/export_corpus.py --output data/processed/corpus.jsonl
```

If that path already contains the same records, the exporter leaves its bytes
untouched. Different data is an error. Recovered v06 records were checked against
the current local project: all 540 records are equal, including metadata and text.
Counts: 126 issues, 214 comments, 54 documentation records, 111 release sections,
35 linked references. These are prepared source records, not final retrieval chunks.

Never recursively index this package. `evaluator_only`, `agent_inputs`, `audit`,
`config`, rubrics, oracle comments and reports are not retrieval source text.
Keep source identity/version/usage restrictions as metadata when chunking.

## Current evaluation files

`frozen_v06/agent_inputs/` and `frozen_v06/evaluator_only/` document the original
15 development + 15 test delivery. Later selected-passage data and revised evaluator
contracts already exist in the project and remain authoritative for their named
experiments. Do not overwrite those later revisions with this baseline snapshot.
The purpose of this package is recovery of acquisition/preparation code and its
data provenance, not replacing the final team's chosen evaluation version.

## GitHub checklist

Commit this directory additively, retaining the source LICENSE and provenance.
Do not commit `.env`, tokens, local download outputs, virtual environments or cache
directories. Suggested patterns are included in `.gitignore`.
The historical `colleague_scripts` name is preserved in provenance; do not claim
unverified sole authorship or remove upstream source attribution.
No Git commit, push, network download, chunking, embedding, model call or historical
Streamlit reproduction was performed by this packaging operation.
