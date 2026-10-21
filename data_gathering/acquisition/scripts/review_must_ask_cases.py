import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
DEV_PATH = ROOT / "data" / "evaluation" / "dev.json"
TEST_PATH = ROOT / "data" / "evaluation" / "test.json"

OUTPUT_PATH = ROOT / "data" / "evaluation" / "must_ask_review.md"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_records(data, keys):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return value

    return []


def build_corpus_index(records):
    return {
        record.get("id"): record
        for record in records
        if record.get("id")
    }


def get_evidence_text(record):
    title = record.get("title", "")
    text = record.get("text", "")
    source_type = record.get("source_type", "")
    url = record.get("url", "")

    return {
        "title": title,
        "text": text,
        "source_type": source_type,
        "url": url,
    }


def has_must_ask(case):
    turns = case.get("turns", [])

    for turn in turns:
        must_ask = turn.get("must_ask", [])

        if isinstance(must_ask, list) and must_ask:
            return True

    return False


def print_separator():
    print("-" * 80)


def main():
    corpus_data = load_json(CORPUS_PATH)
    dev_data = load_json(DEV_PATH)
    test_data = load_json(TEST_PATH)

    corpus_records = unwrap_records(
        corpus_data,
        ["records", "corpus"],
    )

    dev_cases = unwrap_records(
        dev_data,
        ["cases", "evaluation"],
    )

    test_cases = unwrap_records(
        test_data,
        ["cases", "evaluation"],
    )

    corpus_index = build_corpus_index(corpus_records)

    all_cases = []

    for case in dev_cases:
        all_cases.append(case)

    for case in test_cases:
        all_cases.append(case)

    target_cases = []

    for case in all_cases:
        if not has_must_ask(case):
            target_cases.append(case)

    print("=" * 80)
    print("CASES REQUIRING MANUAL MUST-ASK REVIEW")
    print("=" * 80)

    print(f"Total evaluation cases: {len(all_cases)}")
    print(f"Cases without must_ask: {len(target_cases)}")
    print()

    markdown = []

    markdown.append("# Must-Ask Review")
    markdown.append("")
    markdown.append(
        f"Total evaluation cases: {len(all_cases)}"
    )
    markdown.append(
        f"Cases without must_ask: {len(target_cases)}"
    )
    markdown.append("")

    for index, case in enumerate(target_cases, start=1):
        case_id = case.get("case_id", "UNKNOWN")
        family_id = case.get("family_id", "UNKNOWN")
        split = case.get("split", "UNKNOWN")
        tags = case.get("tags", [])
        multi_turn = case.get("multi_turn", False)

        print_separator()

        print(
            f"[{index}/{len(target_cases)}] "
            f"{case_id}"
        )

        print(f"Family ID : {family_id}")
        print(f"Split     : {split}")
        print(f"Multi-turn: {multi_turn}")
        print(
            f"Tags      : "
            f"{', '.join(tags) if tags else 'None'}"
        )

        print()
        print("TURNS:")

        markdown.append(f"## {index}. {case_id}")
        markdown.append("")
        markdown.append(f"- Family ID: `{family_id}`")
        markdown.append(f"- Split: `{split}`")
        markdown.append(f"- Multi-turn: `{multi_turn}`")
        markdown.append(
            f"- Tags: "
            f"{', '.join(tags) if tags else 'None'}"
        )
        markdown.append("")

        turns = case.get("turns", [])

        for turn in turns:
            turn_id = turn.get("turn_id", "?")
            user_question = str(
                turn.get("user", "")
            ).strip()

            evidence_ids = turn.get(
                "gold_evidence_ids",
                [],
            )

            expected_behavior = turn.get(
                "expected_behavior",
                [],
            )

            must_not_claim = turn.get(
                "must_not_claim",
                [],
            )

            print()
            print(f"  TURN {turn_id}")
            print("  User:")
            print(f"    {user_question}")

            print()
            print("  Gold evidence:")

            if not evidence_ids:
                print("    NONE")

            for evidence_id in evidence_ids:
                record = corpus_index.get(evidence_id)

                if record is None:
                    print(
                        f"    {evidence_id} "
                        f"[MISSING FROM CORPUS]"
                    )
                    continue

                evidence = get_evidence_text(record)

                print(
                    f"    {evidence_id} "
                    f"[{evidence['source_type']}]"
                )

                print(
                    f"      Title: "
                    f"{evidence['title']}"
                )

                text = evidence["text"].strip()

                if len(text) > 1000:
                    text = text[:1000] + "..."

                print(
                    f"      Text: "
                    f"{text}"
                )

            print()
            print("  Expected behavior:")

            if expected_behavior:
                for item in expected_behavior:
                    print(f"    - {item}")
            else:
                print("    NONE")

            print()
            print("  Must NOT claim:")

            if must_not_claim:
                for item in must_not_claim:
                    print(f"    - {item}")
            else:
                print("    NONE")

            markdown.append(f"### Turn {turn_id}")
            markdown.append("")
            markdown.append("**User question:**")
            markdown.append("")
            markdown.append(user_question)
            markdown.append("")

            markdown.append("**Gold evidence:**")
            markdown.append("")

            if not evidence_ids:
                markdown.append("- NONE")

            for evidence_id in evidence_ids:
                record = corpus_index.get(evidence_id)

                if record is None:
                    markdown.append(
                        f"- `{evidence_id}` "
                        f"**MISSING FROM CORPUS**"
                    )
                    continue

                evidence = get_evidence_text(record)

                markdown.append(
                    f"- `{evidence_id}` "
                    f"({evidence['source_type']}) — "
                    f"{evidence['title']}"
                )

                text = evidence["text"].strip()

                if len(text) > 1000:
                    text = text[:1000] + "..."

                markdown.append("")
                markdown.append(
                    f"  > {text.replace(chr(10), ' ')}"
                )

            markdown.append("")
            markdown.append("**Expected behavior:**")
            markdown.append("")

            if expected_behavior:
                for item in expected_behavior:
                    markdown.append(f"- {item}")
            else:
                markdown.append("- NONE")

            markdown.append("")
            markdown.append("**Must NOT claim:**")
            markdown.append("")

            if must_not_claim:
                for item in must_not_claim:
                    markdown.append(f"- {item}")
            else:
                markdown.append("- NONE")

            markdown.append("")

        final_expected = case.get(
            "final_expected",
            {},
        )

        answer_points = final_expected.get(
            "answer_points",
            [],
        )

        evidence_requirements = final_expected.get(
            "evidence_requirements",
            [],
        )

        maintainer_summary_points = final_expected.get(
            "maintainer_summary_points",
            [],
        )

        print()
        print("  FINAL EXPECTED:")

        print("    Answer points:")
        for item in answer_points:
            print(f"      - {item}")

        print()
        print("    Evidence requirements:")
        for item in evidence_requirements:
            print(f"      - {item}")

        print()
        print("    Maintainer summary:")
        for item in maintainer_summary_points:
            print(f"      - {item}")

        markdown.append("### Final expected")
        markdown.append("")
        markdown.append("**Answer points:**")
        markdown.append("")

        for item in answer_points:
            markdown.append(f"- {item}")

        markdown.append("")
        markdown.append("**Evidence requirements:**")
        markdown.append("")

        for item in evidence_requirements:
            markdown.append(f"- {item}")

        markdown.append("")
        markdown.append("**Maintainer summary points:**")
        markdown.append("")

        for item in maintainer_summary_points:
            markdown.append(f"- {item}")

        markdown.append("")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(markdown))

    print()
    print("=" * 80)
    print("REVIEW FILE CREATED")
    print("=" * 80)
    print()
    print(f"Markdown: {OUTPUT_PATH}")
    print()
    print(
        "Review each case and decide whether it needs "
        "a discriminating must_ask question."
    )
    print()
    print("Recommended decisions:")
    print(
        "  A = Add must_ask because important information "
        "is missing."
    )
    print(
        "  B = No must_ask needed because the case is "
        "answerable from the evidence."
    )


if __name__ == "__main__":
    main()