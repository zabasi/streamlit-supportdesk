"""HW03 grounding measures, scored against this project's annotated gold.

Retrieval gold is source_ids + selected_spans. There is no fact index, so:

- claim support is lexical overlap with the *cited* packet text (see generation.grounded)
- completeness is how many gold-span content words appear in the delivered answer
- a turn *should* abstain when the packet has no gold source (inventing is worse)
- a turn *should* answer when the packet does hold a gold source
"""
from __future__ import annotations

from ..generation.grounded import DECISIONS, content_words
from .cases import EvalCase
from .metrics import cites_gold

#: Gold labels for the matrix. Predicted may also include ``validation_failed``.
GOLD_LABELS = list(DECISIONS)


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    total = precision + recall
    return round(2 * precision * recall / total, 4) if total else 0.0


def _is_refusal(outcome: str | None) -> bool:
    """Anything other than a delivered ``answer`` (including request_info / escalate)."""
    return outcome != "answer"


def gold_completeness(answer: str | None, spans: list[dict]) -> float | None:
    """Share of gold-span content words that appear in the delivered answer."""
    gold_words = []
    for span in spans or []:
        gold_words.extend(content_words(span.get("text") or ""))
    gold_words = list(dict.fromkeys(gold_words))
    if not gold_words:
        return None
    if not (answer or "").strip():
        return 0.0
    blob = answer.lower()
    return sum(w in blob for w in gold_words) / len(gold_words)


def expected_label(case: EvalCase) -> str | None:
    """The gold decision: first acceptable outcome (each issue case has exactly one)."""
    for label in case.acceptable_outcomes or []:
        if label in GOLD_LABELS:
            return label
    return None


def confusion_matrix(pairs: list[tuple[EvalCase, dict]]) -> dict:
    """Counts of gold decision × delivered decision.

    Rows are gold (``answer``, ``request_info``, ``escalate``). Columns are what
    the turn delivered, including ``validation_failed`` when the checker blocked
    an ungrounded ``answer``.
    """
    predicted_labels = list(GOLD_LABELS)
    extra = sorted({
        (row.get("outcome") or "unknown")
        for _, row in pairs
        if (row.get("outcome") or "unknown") not in GOLD_LABELS
    })
    predicted_labels.extend(extra)
    counts = {gold: {pred: 0 for pred in predicted_labels} for gold in GOLD_LABELS}
    n = 0
    for case, row in pairs:
        gold = expected_label(case)
        if gold is None:
            continue
        pred = row.get("outcome") or "unknown"
        if pred not in counts[gold]:
            for row_counts in counts.values():
                row_counts[pred] = 0
            predicted_labels.append(pred)
        counts[gold][pred] += 1
        n += 1
    return {
        "labels_gold": list(GOLD_LABELS),
        "labels_predicted": predicted_labels,
        "counts": counts,
        "n": n,
    }


#: A finished run's results rows name each case's category; this is the decision each category expects.
EXPECTED_BY_CATEGORY = {"answerable": "answer", "ambiguous": "request_info", "escalation": "escalate"}


def matrix_from_rows(rows: list[dict]) -> dict | None:
    """The same matrix as ``confusion_matrix``, from a finished run's results rows (``results.jsonl``):
    rows are the decision each case's category expects, columns the decision the run made, with any other
    outcome (an error, a failed validation) as a column of its own. None for a run without decisions
    (retrieval only)."""
    pairs = [(EXPECTED_BY_CATEGORY[r["category"]], r.get("outcome") or "unknown") for r in rows
             if r.get("category") in EXPECTED_BY_CATEGORY and r.get("outcome") != "retrieval_only"]
    if not pairs:
        return None
    predicted = list(GOLD_LABELS) + sorted({p for _, p in pairs if p not in GOLD_LABELS})
    counts = {gold: {pred: 0 for pred in predicted} for gold in GOLD_LABELS}
    for gold, pred in pairs:
        counts[gold][pred] += 1
    return {"labels_gold": list(GOLD_LABELS), "labels_predicted": predicted, "counts": counts, "n": len(pairs)}


def matrix_correct(matrix: dict) -> int:
    """Cases on the diagonal: the decision the category expects."""
    return sum(matrix["counts"][g].get(g, 0) for g in matrix["labels_gold"])


def format_confusion(matrix: dict) -> str:
    """A markdown table: rows = gold, columns = predicted."""
    golds = matrix.get("labels_gold") or GOLD_LABELS
    preds = matrix.get("labels_predicted") or list(GOLD_LABELS)
    counts = matrix.get("counts") or {}
    header = "| gold \\ predicted | " + " | ".join(preds) + " | total |"
    sep = "|---|---:" + "|---:" * (len(preds)) + "|"
    lines = [header, sep]
    for gold in golds:
        row = counts.get(gold) or {}
        cells = [str(row.get(pred, 0)) for pred in preds]
        total = sum(row.get(pred, 0) for pred in preds)
        lines.append(f"| {gold} | " + " | ".join(cells) + f" | {total} |")
    col_tot = [sum((counts.get(g) or {}).get(pred, 0) for g in golds) for pred in preds]
    lines.append("| **total** | " + " | ".join(str(v) for v in col_tot)
                 + f" | {matrix.get('n', sum(col_tot))} |")
    return "\n".join(lines)


def packet_has_gold(row: dict, case: EvalCase) -> bool:
    if row.get("packet_has_gold") is not None:
        return bool(row["packet_has_gold"])
    sources = set(row.get("packet_source_ids") or [])
    if not sources:
        for item in row.get("packet") or []:
            sources.add(item.get("source_id"))
            if item.get("parent_source_id"):
                sources.add(item["parent_source_id"])
    return bool(sources & set(case.gold_evidence))


def evaluate_grounded(pairs: list[tuple[EvalCase, dict]]) -> dict:
    """Score grounded turns. Rejected answers still count toward claim support / citations."""
    support_yes = support_all = 0
    cited_in = cited_all = 0
    outside = 0
    stale = 0
    false_success = 0
    refuse_tp = refuse_fp = refuse_fn = 0
    should_answer = did_answer_ok = 0
    complete_sum = 0.0
    complete_n = 0
    cites_gold_yes = cites_gold_n = 0
    echo = 0
    per_case = []

    for case, row in pairs:
        packet = set(row.get("retrieved_chunk_ids") or [])
        citations = [str(c) for c in (row.get("citations") or [])]
        for citation in citations:
            cited_all += 1
            if citation in packet:
                cited_in += 1
            else:
                outside += 1
        for claim in row.get("claims") or []:
            if not isinstance(claim, dict) or not claim.get("asserts_fact"):
                continue
            support_all += 1
            support_yes += int(bool(claim.get("supported")))
        codes = [v.get("code") if isinstance(v, dict) else str(v)
                 for v in (row.get("violations") or [])]
        if row.get("outcome") == "answer" and codes:
            false_success += 1
        if "instruction_echo" in codes:
            echo += 1
        should_refuse = not packet_has_gold(row, case)
        did_refuse = _is_refusal(row.get("outcome"))
        refuse_tp += int(did_refuse and should_refuse)
        refuse_fp += int(did_refuse and not should_refuse)
        refuse_fn += int((not did_refuse) and should_refuse)
        if not should_refuse:
            should_answer += 1
            did_answer_ok += int(row.get("outcome") == "answer")
        if row.get("outcome") == "answer":
            cov = gold_completeness(row.get("answer"), case.evidence_spans)
            if cov is not None:
                complete_sum += cov
                complete_n += 1
            if case.gold_evidence:
                cites_gold_n += 1
                cites_gold_yes += int(cites_gold(case, {
                    "retrieved": list(row.get("packet") or []),
                    "citations": citations,
                }))
        per_case.append({
            "case_id": case.case_id,
            "category": case.category,
            "outcome": row.get("outcome"),
            "should_refuse": should_refuse,
            "packet_has_gold": packet_has_gold(row, case),
            "n_claims": len(row.get("claims") or []),
            "n_citations": len(citations),
            "citations_outside": sum(1 for c in citations if c not in packet),
            "violations": " ".join(c for c in codes if c),
            "abstention_reason": row.get("abstention_reason") or "",
            "gold_completeness": gold_completeness(row.get("answer"), case.evidence_spans)
            if row.get("outcome") == "answer" else None,
            "expected_outcomes": list(case.acceptable_outcomes),
            "outcome_ok": bool(case.acceptable_outcomes)
            and row.get("outcome") in case.acceptable_outcomes,
        })

    precision = _ratio(refuse_tp, refuse_tp + refuse_fp)
    recall = _ratio(refuse_tp, refuse_tp + refuse_fn)
    outcomes: dict[str, int] = {}
    for _, row in pairs:
        outcomes[row.get("outcome") or "unknown"] = outcomes.get(row.get("outcome") or "unknown", 0) + 1
    labeled = [(c, r) for c, r in pairs if c.acceptable_outcomes]
    outcome_hits = sum(r.get("outcome") in c.acceptable_outcomes for c, r in labeled)
    by_cat: dict[str, dict] = {}
    for category in sorted({c.category for c, _ in labeled}):
        sub = [(c, r) for c, r in labeled if c.category == category]
        by_cat[category] = {
            "n": len(sub),
            "outcome_accuracy": _ratio(
                sum(r.get("outcome") in c.acceptable_outcomes for c, r in sub), len(sub)),
            "outcomes": {o: sum(r.get("outcome") == o for _, r in sub)
                         for o in sorted({r.get("outcome") or "unknown" for _, r in sub})},
            "expected": list(sub[0][0].acceptable_outcomes) if sub else [],
        }
    matrix = confusion_matrix(pairs)
    headline = {
        "n_cases": len(pairs),
        "outcome_accuracy": _ratio(outcome_hits, len(labeled)),
        "by_category": by_cat,
        "confusion_matrix": matrix,
        "claim_support_rate": _ratio(support_yes, support_all),
        "citation_precision": _ratio(cited_in, cited_all),
        "citations_outside_packet": outside,
        "stale_answers": stale,
        "answered_after_failure": false_success,
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": _f1(precision, recall),
        "answer_when_packet_has_gold": _ratio(did_answer_ok, should_answer),
        "answer_completeness": round(complete_sum / complete_n, 4) if complete_n else None,
        "answer_cites_gold_rate": _ratio(cites_gold_yes, cites_gold_n),
        "instruction_echo": echo,
        "outcomes": dict(sorted(outcomes.items())),
        "n_claims": support_all,
        "n_citations": cited_all,
    }
    return {"kind": "grounded", "headline": headline, "per_case": per_case}


def _show(value) -> str:
    return "n/a" if value is None else str(value)


def format_grounded(m: dict) -> str:
    h = m.get("headline") or m
    lines = [
        (f"cases: {h.get('n_cases')}   outcomes {h.get('outcomes')}   "
        f"outcome accuracy {_show(h.get('outcome_accuracy'))}"),
    ]
    cats = h.get("by_category") or {}
    if cats:
        lines.append("           " + "  ".join(
            f"{cat} {_show(body.get('outcome_accuracy'))} (n={body.get('n')})"
            for cat, body in cats.items()))
    matrix = h.get("confusion_matrix")
    if matrix:
        lines += ["", "confusion (gold \\ predicted)", format_confusion(matrix)]
    lines += [
        (
            f"grounding  claim support {_show(h.get('claim_support_rate'))}   "
            f"citation precision {_show(h.get('citation_precision'))}   "
            f"outside packet {h.get('citations_outside_packet')}   "
            f"answered after failure {h.get('answered_after_failure')}"
        ),
        (
            f"refusals   abstention P/R/F1 "
            f"{_show(h.get('abstention_precision'))}/"
            f"{_show(h.get('abstention_recall'))}/"
            f"{_show(h.get('abstention_f1'))}   "
            f"answered when packet had gold {_show(h.get('answer_when_packet_has_gold'))}"
        ),
        (
            f"usefulness completeness vs gold spans {_show(h.get('answer_completeness'))}   "
            f"answers cite gold {_show(h.get('answer_cites_gold_rate'))}   "
            f"instruction echo {h.get('instruction_echo')}"
        ),
    ]
    return "\n".join(lines)
