from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    template = {
        "schema_version": "1.0",
        "dataset_version": "runtime_state_widgets-v01",
        "description": "Evaluation case template for the Streamlit technical support assistant.",
        "rules": {
            "total_cases": 30,
            "dev_cases": 15,
            "test_cases": 15,
            "minimum_multi_turn_cases": 10,
            "minimum_test_multi_turn_cases": 5,
            "split_unit": "family_id",
            "test_isolation": True,
            "evidence_must_exist_in_corpus": True,
            "closed_issue_is_not_ground_truth": True,
            "missing_information_must_not_be_guessed": True,
        },
        "case_template": {
            "case_id": "dev-001",
            "family_id": "family-001",
            "split": "dev",
            "multi_turn": True,
            "tags": [
                "session_state",
                "retrieval",
                "missing_information",
            ],
            "turns": [
                {
                    "turn_id": 1,
                    "user": "User question goes here.",
                    "gold_evidence_ids": [
                        "issue-123",
                        "comment-456",
                        "document-789",
                    ],
                    "expected_behavior": [
                        "Identify the reported problem.",
                        "Cite the relevant evidence.",
                        "Do not guess missing technical details.",
                    ],
                    "must_ask": [],
                    "must_not_claim": [
                        "Do not claim that a closed issue is automatically solved."
                    ],
                }
            ],
            "final_expected": {
                "answer_points": [
                    "Expected factual point 1.",
                    "Expected factual point 2.",
                ],
                "evidence_requirements": [
                    "At least one relevant issue or comment.",
                    "Use official documentation when applicable.",
                ],
                "maintainer_summary_points": [
                    "Problem summary.",
                    "Evidence-backed findings.",
                    "Open questions or next action.",
                ],
            },
        },
    }

    dev = {
        "schema_version": "1.0",
        "dataset_version": "runtime_state_widgets-v01",
        "split": "dev",
        "cases": [],
    }

    test = {
        "schema_version": "1.0",
        "dataset_version": "runtime_state_widgets-v01",
        "split": "test",
        "cases": [],
    }

    write_json(EVAL_DIR / "cases.template.json", template)
    write_json(EVAL_DIR / "dev.json", dev)
    write_json(EVAL_DIR / "test.json", test)

    print("Evaluation files created:")
    print(EVAL_DIR / "cases.template.json")
    print(EVAL_DIR / "dev.json")
    print(EVAL_DIR / "test.json")


if __name__ == "__main__":
    main()
