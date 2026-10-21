import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
DEV_PATH = ROOT / "data" / "evaluation" / "dev.json"
TEST_PATH = ROOT / "data" / "evaluation" / "test.json"


EXPECTED_TOTAL = 30
EXPECTED_DEV = 15
EXPECTED_TEST = 15
MIN_MULTI_TURN_TOTAL = 10
MIN_MULTI_TURN_TEST = 5


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_cases(data, path: Path):
    """
    Accept both supported formats:

    1. Direct list:
       [
           {...},
           {...}
       ]

    2. Wrapped object:
       {
           "cases": [
               {...},
               {...}
           ]
       }
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        cases = data.get("cases", [])

        if not isinstance(cases, list):
            raise ValueError(
                f"{path} contains a 'cases' field, but it is not a list."
            )

        return cases

    raise ValueError(
        f"{path} must contain either a JSON list or an object with a 'cases' list."
    )


def validate_case(case, corpus_ids, expected_split, errors):
    if not isinstance(case, dict):
        errors.append(
            f"{expected_split}: case is not a JSON object."
        )
        return

    required_fields = [
        "case_id",
        "family_id",
        "split",
        "multi_turn",
        "tags",
        "turns",
        "final_expected",
    ]

    for field in required_fields:
        if field not in case:
            errors.append(
                f"{expected_split}: {case.get('case_id', '<unknown>')} "
                f"missing field '{field}'."
            )

    case_id = case.get("case_id", "<unknown>")

    if case.get("split") != expected_split:
        errors.append(
            f"{case_id}: split is '{case.get('split')}', "
            f"expected '{expected_split}'."
        )

    if not isinstance(case.get("turns"), list):
        errors.append(
            f"{case_id}: 'turns' must be a list."
        )
        return

    if len(case["turns"]) == 0:
        errors.append(
            f"{case_id}: case must contain at least one turn."
        )

    for turn in case["turns"]:
        if not isinstance(turn, dict):
            errors.append(
                f"{case_id}: turn is not an object."
            )
            continue

        evidence_ids = turn.get("gold_evidence_ids", [])

        if not isinstance(evidence_ids, list):
            errors.append(
                f"{case_id}: gold_evidence_ids must be a list."
            )
            continue

        for evidence_id in evidence_ids:
            if evidence_id not in corpus_ids:
                errors.append(
                    f"{case_id}: evidence ID '{evidence_id}' "
                    f"does not exist in corpus."
                )


def main():
    print("=" * 60)
    print("VALIDATING FINAL EVALUATION DATASET")
    print("=" * 60)

    errors = []

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    for path in [CORPUS_PATH, DEV_PATH, TEST_PATH]:
        if not path.exists():
            errors.append(f"Missing file: {path}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    # ---------------------------------------------------------
    # Load corpus
    # ---------------------------------------------------------

    corpus_data = load_json(CORPUS_PATH)

    if isinstance(corpus_data, dict):
        corpus_records = corpus_data.get("records", [])
    elif isinstance(corpus_data, list):
        corpus_records = corpus_data
    else:
        raise ValueError("Invalid corpus.json format.")

    corpus_ids = set()

    for record in corpus_records:
        if isinstance(record, dict) and "id" in record:
            corpus_ids.add(record["id"])

    print(f"Corpus records: {len(corpus_records)}")
    print(f"Corpus IDs: {len(corpus_ids)}")

    # ---------------------------------------------------------
    # Load evaluation files
    # ---------------------------------------------------------

    dev_data = load_json(DEV_PATH)
    test_data = load_json(TEST_PATH)

    dev_cases = extract_cases(dev_data, DEV_PATH)
    test_cases = extract_cases(test_data, TEST_PATH)

    all_cases = dev_cases + test_cases

    print(f"DEV cases: {len(dev_cases)}")
    print(f"TEST cases: {len(test_cases)}")
    print(f"Total cases: {len(all_cases)}")

    # ---------------------------------------------------------
    # Count checks
    # ---------------------------------------------------------

    if len(all_cases) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} total cases, "
            f"found {len(all_cases)}."
        )

    if len(dev_cases) != EXPECTED_DEV:
        errors.append(
            f"Expected {EXPECTED_DEV} DEV cases, "
            f"found {len(dev_cases)}."
        )

    if len(test_cases) != EXPECTED_TEST:
        errors.append(
            f"Expected {EXPECTED_TEST} TEST cases, "
            f"found {len(test_cases)}."
        )

    # ---------------------------------------------------------
    # Unique case IDs
    # ---------------------------------------------------------

    case_ids = [
        case.get("case_id")
        for case in all_cases
        if isinstance(case, dict)
    ]

    duplicate_case_ids = [
        case_id
        for case_id, count in Counter(case_ids).items()
        if count > 1
    ]

    if duplicate_case_ids:
        errors.append(
            f"Duplicate case IDs: {duplicate_case_ids}"
        )

    # ---------------------------------------------------------
    # Family split isolation
    # ---------------------------------------------------------

    family_splits = {}

    for case in all_cases:
        if not isinstance(case, dict):
            continue

        family_id = case.get("family_id")
        split = case.get("split")

        if family_id is None:
            continue

        family_splits.setdefault(family_id, set()).add(split)

    crossing_families = {
        family_id: splits
        for family_id, splits in family_splits.items()
        if len(splits) > 1
    }

    if crossing_families:
        errors.append(
            "Families appear in multiple splits: "
            f"{crossing_families}"
        )

    # ---------------------------------------------------------
    # Multi-turn checks
    # ---------------------------------------------------------

    multi_turn_total = sum(
        1
        for case in all_cases
        if isinstance(case, dict) and case.get("multi_turn") is True
    )

    multi_turn_dev = sum(
        1
        for case in dev_cases
        if isinstance(case, dict) and case.get("multi_turn") is True
    )

    multi_turn_test = sum(
        1
        for case in test_cases
        if isinstance(case, dict) and case.get("multi_turn") is True
    )

    print(f"Multi-turn total: {multi_turn_total}")
    print(f"Multi-turn DEV: {multi_turn_dev}")
    print(f"Multi-turn TEST: {multi_turn_test}")

    if multi_turn_total < MIN_MULTI_TURN_TOTAL:
        errors.append(
            f"Need at least {MIN_MULTI_TURN_TOTAL} multi-turn cases; "
            f"found {multi_turn_total}."
        )

    if multi_turn_test < MIN_MULTI_TURN_TEST:
        errors.append(
            f"Need at least {MIN_MULTI_TURN_TEST} multi-turn TEST cases; "
            f"found {multi_turn_test}."
        )

    # ---------------------------------------------------------
    # Validate individual cases
    # ---------------------------------------------------------

    for case in dev_cases:
        validate_case(
            case,
            corpus_ids,
            "dev",
            errors,
        )

    for case in test_cases:
        validate_case(
            case,
            corpus_ids,
            "test",
            errors,
        )

    # ---------------------------------------------------------
    # Evidence statistics
    # ---------------------------------------------------------

    evidence_ids = set()

    for case in all_cases:
        if not isinstance(case, dict):
            continue

        for turn in case.get("turns", []):
            if not isinstance(turn, dict):
                continue

            for evidence_id in turn.get("gold_evidence_ids", []):
                evidence_ids.add(evidence_id)

    invalid_evidence = evidence_ids - corpus_ids

    print(f"Unique evidence IDs used: {len(evidence_ids)}")

    if invalid_evidence:
        errors.append(
            f"Invalid evidence IDs: {sorted(invalid_evidence)}"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 60)

    if errors:
        print("VALIDATION FAILED")
        print("=" * 60)

        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")

        raise SystemExit(1)

    print("VALIDATION PASSED")
    print("=" * 60)

    print()
    print("Evaluation dataset satisfies the structural requirements:")
    print(f"  Total cases       : {len(all_cases)}")
    print(f"  DEV cases         : {len(dev_cases)}")
    print(f"  TEST cases        : {len(test_cases)}")
    print(f"  Multi-turn total  : {multi_turn_total}")
    print(f"  Multi-turn DEV    : {multi_turn_dev}")
    print(f"  Multi-turn TEST   : {multi_turn_test}")
    print(f"  Unique evidence   : {len(evidence_ids)}")
    print("  Family isolation  : PASSED")
    print("  Evidence validity : PASSED")


if __name__ == "__main__":
    main()
