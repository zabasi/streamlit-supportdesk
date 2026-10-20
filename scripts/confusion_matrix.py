"""Print the decision confusion matrix of finished runs as Markdown (for the written report).

Rows are the decision each case's category expects (answer, request_info, escalate); columns are the
decision the run made, with an error or a failed validation as a column of its own. The same matrices
appear in the HTML of `supportdesk report`.

    python scripts/confusion_matrix.py e2e-baseline-6 e2e-candidate-5b
    python scripts/confusion_matrix.py artifacts/runs/e2e-candidate-9 --output artifacts/reports/matrices.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supportdesk.eval.grounded_metrics import format_confusion, matrix_correct, matrix_from_rows


def run_dir(name: str, runs: Path) -> Path:
    path = Path(name)
    return path if (path / "results.jsonl").exists() else runs / name


def markdown(dirs: list[Path]) -> str:
    out = []
    for d in dirs:
        rows = [json.loads(line) for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        matrix = matrix_from_rows(rows)
        if matrix is None:
            out.append(f"### {d.name}\n\nRetrieval only: no decisions.\n")
            continue
        out.append(f"### {d.name}: {matrix_correct(matrix)} of {matrix['n']} correct\n\n"
                   f"{format_confusion(matrix)}\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run ids (under --runs-dir) or run directories")
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, default=None, help="write the Markdown here instead of printing it")
    args = parser.parse_args(argv)
    dirs = [run_dir(r, args.runs_dir) for r in args.runs]
    missing = [str(d) for d in dirs if not (d / "results.jsonl").exists()]
    if missing:
        parser.error("no results.jsonl in: " + ", ".join(missing))
    text = markdown(dirs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
