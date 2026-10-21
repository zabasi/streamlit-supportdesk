"""Run the complete dataset collection, normalization, validation and template pipeline."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("Download and select GitHub Issues", "download_issues.py"),
    ("Download comments for selected issues", "download_comments.py"),
    ("Download pinned official documentation", "download_documents.py"),
    ("Download official release notes", "download_release_notes.py"),
    ("Normalize raw sources into corpus", "build_corpus.py"),
    ("Validate the dataset", "validate_dataset.py"),
    ("Create separate evaluation template", "create_evaluation_template.py"),
]


def main() -> int:
    print("Streamlit Support Dataset Pipeline")
    print(f"Project root: {ROOT}\n")
    for label, script in STEPS:
        print(f"\n{'=' * 72}\nSTEP: {label}\n{'=' * 72}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
        if result.returncode:
            print(f"\nSTOPPED: {script} returned exit code {result.returncode}.")
            print("Check your internet connection, GitHub API rate limits, and .env settings.")
            print("After fixing the issue, rerun this command. Do not treat a partial corpus as complete.")
            return result.returncode
    print("\nSUCCESS: all pipeline stages completed.")
    print("Outputs: data/raw/, data/corpus/, data/evaluation/cases.template.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
