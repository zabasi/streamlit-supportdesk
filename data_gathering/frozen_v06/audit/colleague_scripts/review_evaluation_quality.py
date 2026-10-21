import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.json"
DEV_PATH = ROOT / "data" / "evaluation" / "dev.json"
TEST_PATH = ROOT / "data" / "evaluation" / "test.json"

OUTPUT_JSON = ROOT / "data" / "evaluation" / "quality_review.json"
OUTPUT_MD = ROOT / "data" / "evaluation" / "quality_review.md"


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "it",
    "as",
    "by",
    "from",
    "at",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "my",
    "me",
    "i",
    "we",
    "you",
    "your",
    "issue",
    "problem",
    "error",
    "using",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_records(data, possible_keys):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in possible_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value

    return []


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9_]+", " ", text)

    return text


def tokens(text):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOPWORDS
    }


def text_overlap(question, evidence_text):
    q_tokens = tokens(question)
    e_tokens = tokens(evidence_text)

    if not q_tokens or not e_tokens:
        return 0.0

    return len(q_tokens & e_tokens) / len(q_tokens)


def build_corpus_index(corpus_records):
    return {
        record.get("id"): record
        for record in corpus_records
        if record.get("id")
    }


def get_case_turns(case):
    turns = case.get("turns", [])

    if not isinstance(turns, list):
        return []

    return turns


def get_evidence_ids(case):
    evidence_ids = []

    for turn in get_case_turns(case):
        ids = turn.get("gold_evidence_ids", [])

        if isinstance(ids, list):
            evidence_ids.extend(ids)

    return list(dict.fromkeys(evidence_ids))


def get_evidence_records(case, corpus_index):
    records = []

    for evidence_id in get_evidence_ids(case):
        record = corpus_index.get(evidence_id)

        if record is not None:
            records.append(record)

    return records


def record_source_counts(records):
    counter = Counter()

    for record in records:
        counter[record.get("source_type", "unknown")] += 1

    return dict(counter)


def evaluate_case(case, corpus_index):
    flags = []
    warnings = []

    case_id = case.get("case_id", "UNKNOWN")
    family_id = case.get("family_id")

    turns = get_case_turns(case)

    if not family_id:
        flags.append("MISSING_FAMILY_ID")

    if not case.get("tags"):
        flags.append("NO_TAGS")

    if not turns:
        flags.append("NO_TURNS")

    if case.get("multi_turn") and len(turns) < 2:
        flags.append("MULTI_TURN_WITH_ONE_TURN")

    if not case.get("multi_turn") and len(turns) >= 2:
        flags.append("MULTI_TURN_FLAG_INCONSISTENT")

    questions = []

    for turn in turns:
        question = str(turn.get("user", "")).strip()

        if not question:
            flags.append("EMPTY_USER_QUESTION")
        else:
            questions.append(question)

        expected_behavior = turn.get("expected_behavior", [])

        if not isinstance(expected_behavior, list) or not expected_behavior:
            flags.append("NO_EXPECTED_BEHAVIOR")

        evidence_ids = turn.get("gold_evidence_ids", [])

        if not isinstance(evidence_ids, list) or not evidence_ids:
            flags.append("NO_GOLD_EVIDENCE")
            continue

        for evidence_id in evidence_ids:
            if evidence_id not in corpus_index:
                flags.append("MISSING_EVIDENCE_ID")

    if len(questions) != len(set(questions)):
        flags.append("DUPLICATE_TURN_QUESTION")

    evidence_records = get_evidence_records(case, corpus_index)

    if evidence_records:
        source_counts = record_source_counts(evidence_records)

        if len(source_counts) == 1:
            warnings.append("SINGLE_EVIDENCE_SOURCE")

    for turn in turns:
        question = str(turn.get("user", "")).strip()

        evidence_ids = turn.get("gold_evidence_ids", [])

        if not question or not evidence_ids:
            continue

        evidence_text_parts = []

        for evidence_id in evidence_ids:
            record = corpus_index.get(evidence_id)

            if record:
                title = record.get("title", "")
                text = record.get("text", "")

                evidence_text_parts.append(
                    f"{title} {text}"
                )

        evidence_text = " ".join(evidence_text_parts)

        overlap = text_overlap(question, evidence_text)

        if overlap < 0.05:
            warnings.append("VERY_LOW_TEXT_OVERLAP")

    has_must_ask = False
    has_must_not_claim = False

    for turn in turns:
        must_ask = turn.get("must_ask", [])
        must_not_claim = turn.get("must_not_claim", [])

        if isinstance(must_ask, list) and must_ask:
            has_must_ask = True

        if isinstance(must_not_claim, list) and must_not_claim:
            has_must_not_claim = True

    if not has_must_ask:
        warnings.append("NO_MUST_ASK_REQUIREMENT")

    if not has_must_not_claim:
        warnings.append("NO_MUST_NOT_CLAIM_REQUIREMENT")

    final_expected = case.get("final_expected")

    if not isinstance(final_expected, dict):
        flags.append("NO_FINAL_EXPECTED")
    else:
        answer_points = final_expected.get("answer_points", [])
        evidence_requirements = final_expected.get(
            "evidence_requirements",
            [],
        )
        maintainer_summary_points = final_expected.get(
            "maintainer_summary_points",
            [],
        )

        if not isinstance(answer_points, list) or not answer_points:
            flags.append("NO_FINAL_ANSWER_POINTS")

        if (
            not isinstance(evidence_requirements, list)
            or not evidence_requirements
        ):
            flags.append("NO_FINAL_EVIDENCE_REQUIREMENTS")

        if (
            not isinstance(maintainer_summary_points, list)
            or not maintainer_summary_points
        ):
            flags.append("NO_MAINTAINER_SUMMARY_POINTS")

    quality_level = "PASS"

    if flags:
        quality_level = "FAIL"
    elif warnings:
        quality_level = "REVIEW"

    return {
        "case_id": case_id,
        "family_id": family_id,
        "split": case.get("split"),
        "multi_turn": case.get("multi_turn"),
        "turn_count": len(turns),
        "evidence_count": len(get_evidence_ids(case)),
        "evidence_source_counts": record_source_counts(evidence_records),
        "flags": list(dict.fromkeys(flags)),
        "warnings": list(dict.fromkeys(warnings)),
        "quality_level": quality_level,
    }


def build_report(corpus_records, dev_cases, test_cases):
    corpus_index = build_corpus_index(corpus_records)

    all_cases = []

    for case in dev_cases:
        all_cases.append(case)

    for case in test_cases:
        all_cases.append(case)

    case_reports = []

    for case in all_cases:
        case_reports.append(
            evaluate_case(
                case,
                corpus_index,
            )
        )

    flag_counts = Counter()
    warning_counts = Counter()
    quality_counts = Counter()
    source_counts = Counter()

    for report in case_reports:
        quality_counts[report["quality_level"]] += 1

        for flag in report["flags"]:
            flag_counts[flag] += 1

        for warning in report["warnings"]:
            warning_counts[warning] += 1

        for source_type, count in report["evidence_source_counts"].items():
            source_counts[source_type] += count

    severe_flags = {
        "NO_GOLD_EVIDENCE",
        "MISSING_EVIDENCE_ID",
        "EMPTY_USER_QUESTION",
        "NO_EXPECTED_BEHAVIOR",
        "NO_TURNS",
        "MULTI_TURN_FLAG_INCONSISTENT",
        "MULTI_TURN_WITH_ONE_TURN",
        "MISSING_FAMILY_ID",
        "NO_FINAL_EXPECTED",
        "NO_FINAL_ANSWER_POINTS",
        "NO_FINAL_EVIDENCE_REQUIREMENTS",
        "NO_MAINTAINER_SUMMARY_POINTS",
    }

    severe_case_count = 0

    for report in case_reports:
        if any(flag in severe_flags for flag in report["flags"]):
            severe_case_count += 1

    if severe_case_count > 0:
        overall_quality = "FAIL"
    elif warning_counts:
        overall_quality = "REVIEW"
    else:
        overall_quality = "PASS"

    return {
        "summary": {
            "corpus_records": len(corpus_records),
            "dev_cases": len(dev_cases),
            "test_cases": len(test_cases),
            "total_cases": len(all_cases),
            "multi_turn_cases": sum(
                1
                for case in all_cases
                if case.get("multi_turn") is True
            ),
            "quality_level": overall_quality,
            "severe_case_count": severe_case_count,
        },
        "flag_counts": dict(flag_counts),
        "warning_counts": dict(warning_counts),
        "quality_counts": dict(quality_counts),
        "evidence_source_counts": dict(source_counts),
        "cases": case_reports,
    }


def build_markdown(report):
    lines = []

    lines.append("# Evaluation Quality Review")
    lines.append("")

    summary = report["summary"]

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Corpus records: {summary['corpus_records']}"
    )
    lines.append(
        f"- DEV cases: {summary['dev_cases']}"
    )
    lines.append(
        f"- TEST cases: {summary['test_cases']}"
    )
    lines.append(
        f"- Total cases: {summary['total_cases']}"
    )
    lines.append(
        f"- Multi-turn cases: {summary['multi_turn_cases']}"
    )
    lines.append(
        f"- Severe cases: {summary['severe_case_count']}"
    )
    lines.append(
        f"- Overall quality: **{summary['quality_level']}**"
    )
    lines.append("")

    lines.append("## Quality counts")
    lines.append("")

    if report["quality_counts"]:
        for level, count in report["quality_counts"].items():
            lines.append(f"- `{level}`: {count}")
    else:
        lines.append("- None")

    lines.append("")

    lines.append("## Flag counts")
    lines.append("")

    if report["flag_counts"]:
        for flag, count in report["flag_counts"].items():
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- None")

    lines.append("")

    lines.append("## Warning counts")
    lines.append("")

    if report["warning_counts"]:
        for warning, count in report["warning_counts"].items():
            lines.append(f"- `{warning}`: {count}")
    else:
        lines.append("- None")

    lines.append("")

    lines.append("## Evidence source counts")
    lines.append("")

    if report["evidence_source_counts"]:
        for source_type, count in report["evidence_source_counts"].items():
            lines.append(f"- `{source_type}`: {count}")
    else:
        lines.append("- None")

    lines.append("")

    lines.append("## Case review")
    lines.append("")

    for case in report["cases"]:
        lines.append(
            f"### {case['case_id']} — {case['quality_level']}"
        )
        lines.append("")

        lines.append(
            f"- Family: `{case.get('family_id')}`"
        )
        lines.append(
            f"- Split: `{case.get('split')}`"
        )
        lines.append(
            f"- Multi-turn: `{case.get('multi_turn')}`"
        )
        lines.append(
            f"- Turns: {case.get('turn_count')}"
        )
        lines.append(
            f"- Evidence records: {case.get('evidence_count')}"
        )

        if case.get("evidence_source_counts"):
            source_text = ", ".join(
                f"{key}={value}"
                for key, value in case[
                    "evidence_source_counts"
                ].items()
            )
            lines.append(
                f"- Evidence sources: {source_text}"
            )

        flags = case.get("flags", [])

        if flags:
            lines.append("- Flags:")
            for flag in flags:
                lines.append(f"  - `{flag}`")
        else:
            lines.append("- Flags: None")

        warnings = case.get("warnings", [])

        if warnings:
            lines.append("- Warnings:")
            for warning in warnings:
                lines.append(f"  - `{warning}`")
        else:
            lines.append("- Warnings: None")

        lines.append("")

    lines.append("## Interpretation")
    lines.append("")

    if summary["quality_level"] == "PASS":
        lines.append(
            "The evaluation dataset passed the automated semantic "
            "quality checks."
        )
    elif summary["quality_level"] == "REVIEW":
        lines.append(
            "The evaluation dataset passed structural checks but "
            "contains cases that should receive manual semantic review."
        )
    else:
        lines.append(
            "The evaluation dataset contains severe quality issues "
            "that should be fixed before using it for final evaluation."
        )

    lines.append("")

    lines.append(
        "Important: this automated review checks evidence validity, "
        "case structure, basic question/evidence overlap, and "
        "evaluation requirements. It does not replace manual review "
        "of whether each gold answer and evidence relationship is "
        "semantically correct."
    )

    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("REVIEWING EVALUATION DATASET QUALITY")
    print("=" * 60)

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

    print(f"Corpus records: {len(corpus_records)}")
    print(f"DEV cases: {len(dev_cases)}")
    print(f"TEST cases: {len(test_cases)}")

    report = build_report(
        corpus_records,
        dev_cases,
        test_cases,
    )

    markdown = build_markdown(report)

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with OUTPUT_MD.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(markdown)

    print()
    print("=" * 60)
    print("QUALITY REVIEW COMPLETED")
    print("=" * 60)

    print(
        f"Overall quality: "
        f"{report['summary']['quality_level']}"
    )

    print(
        f"Severe cases: "
        f"{report['summary']['severe_case_count']}"
    )

    print()
    print("Quality counts:")

    for level, count in report["quality_counts"].items():
        print(f"  {level}: {count}")

    print()
    print("Flags:")

    if report["flag_counts"]:
        for flag, count in report["flag_counts"].items():
            print(f"  {flag}: {count}")
    else:
        print("  None")

    print()
    print("Warnings:")

    if report["warning_counts"]:
        for warning, count in report["warning_counts"].items():
            print(f"  {warning}: {count}")
    else:
        print("  None")

    print()
    print("Files:")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  MD  : {OUTPUT_MD}")

    print("=" * 60)


if __name__ == "__main__":
    main()