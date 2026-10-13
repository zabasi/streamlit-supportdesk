"""Metrics for a run. Decisions are split by category so refusing everything cannot look good."""
from __future__ import annotations

import statistics

from ..tokenizer import tokenize_with_spans
from .cases import EvalCase


def hit_source_ids(retrieved: list[dict]) -> list[set[str]]:
    """Per rank: the ids a hit counts as. A comment also counts as its parent issue."""
    return [{r["source_id"], r["parent_source_id"]} - {None} for r in retrieved]


def recall_at_k(retrieved: list[dict], gold: list[str], k: int) -> float | None:
    """Fraction of gold sources found in the top k. None when the case has no gold evidence."""
    if not gold:
        return None
    found = set().union(*hit_source_ids(retrieved[:k])) if retrieved[:k] else set()
    return len(found & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved: list[dict], gold: list[str]) -> float | None:
    if not gold:
        return None
    for rank, ids in enumerate(hit_source_ids(retrieved), start=1):
        if ids & set(gold):
            return 1.0 / rank
    return 0.0


def cites_gold(case: EvalCase, result: dict) -> bool:
    """True when at least one cited chunk comes from (or is a comment on) a gold source."""
    ids_by_chunk = {r["chunk_id"]: {r["source_id"], r["parent_source_id"]} - {None}
                    for r in result["retrieved"]}
    return any(ids_by_chunk.get(cid, set()) & set(case.gold_evidence) for cid in result["citations"])


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[min(int(q * len(values)), len(values) - 1)], 4)


def evaluate(pairs: list[tuple[EvalCase, dict]], ks: tuple[int, ...]) -> dict:
    """``pairs`` = (case, CaseResult.to_dict()). Returns a JSON-serialisable summary."""
    n = len(pairs)
    out: dict = {"n_cases": n}
    # evidence quality
    out["recall_at_k"] = {str(k): _mean(recall_at_k(r["retrieved"], c.gold_evidence, k)
                                        for c, r in pairs) for k in ks}
    out["mrr"] = _mean(reciprocal_rank(r["retrieved"], c.gold_evidence) for c, r in pairs)
    # decision quality, overall and by category
    # Only cases with acceptable outcomes are auto-scored; the others (history variants, source
    # tasks) are scored by their rubric. Derivative cases never count towards the sample size.
    def ok(c, r):
        return r["outcome"] in c.acceptable_outcomes
    out["n_independent"] = sum(1 for c, _ in pairs if getattr(c, "independent", True))
    out["outcome_accuracy"] = _mean(float(ok(c, r)) for c, r in pairs if c.acceptable_outcomes)
    by_cat = {}
    for category in sorted({c.category for c, _ in pairs}):
        sub = [(c, r) for c, r in pairs if c.category == category]
        by_cat[category] = {"n": len(sub),
                            "outcome_accuracy": _mean(float(ok(c, r)) for c, r in sub if c.acceptable_outcomes),
                            "outcomes": {o: sum(r["outcome"] == o for _, r in sub)
                                         for o in sorted({r["outcome"] for _, r in sub})}}
    out["by_category"] = by_cat
    # grounding
    answered = [(c, r) for c, r in pairs if r["outcome"] == "answer"]
    out["answers"] = len(answered)
    out["citation_violation_rate"] = _mean(float(bool(r["violations"])) for _, r in pairs)
    out["answer_without_citation_rate"] = _mean(
        float("answer_without_citation" in r["violations"]) for _, r in answered)
    out["answer_cites_gold_rate"] = _mean(
        float(cites_gold(c, r)) for c, r in answered if c.gold_evidence)
    # operations
    out["error_rate"] = _mean(float(r["outcome"] == "error") for _, r in pairs)
    out["budget_stopped"] = sum(r["outcome"] == "budget_stopped" for _, r in pairs)
    out["cost"] = cost_summary([r["usage"] for _, r in pairs])
    return out


def cost_summary(usage: list[dict]) -> dict:
    """Spend, tokens and time over a list of per-case usage records."""
    out = {
        "total_usd": round(sum(u.get("cost_usd", 0.0) for u in usage), 8),
        "replayed_usd": round(sum(u.get("replayed_cost_usd", 0.0) for u in usage), 8),
        "mean_usd_per_case": _mean(u.get("cost_usd", 0.0) for u in usage),
        "upstream_calls": sum(u.get("calls", 0) for u in usage),
        "cache_hits": sum(u.get("cache_hits", 0) for u in usage),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "wall_seconds_mean": _mean(u.get("wall_seconds") for u in usage),
        "wall_seconds_p95": _pct([u.get("wall_seconds") for u in usage], 0.95),
    }
    if len(usage) > 1:
        out["wall_seconds_stdev"] = round(statistics.pstdev(
            [u.get("wall_seconds", 0.0) for u in usage]), 4)
    return out


def passage_coverage(chunks: list[dict], span: dict) -> float | None:
    """How much of one gold passage the chunks cover, from 0 to 1. Measured over the passage's words
    (a leading heading mark or a closing code fence just outside a chunk does not count against it)."""
    words = tokenize_with_spans(span["text"])
    if not words:
        return None
    start, end = span["start_char"] + words[0][1], span["start_char"] + words[-1][2]
    parts = sorted((max(start, c["start_char"]), min(end, c["end_char"])) for c in chunks
                   if c.get("source_id") == span["source_id"] and c.get("start_char") is not None
                   and c.get("end_char") is not None)
    covered, reached = 0, start
    for lo, hi in parts:
        lo = max(lo, reached)
        if hi > lo:
            covered, reached = covered + hi - lo, hi
    return covered / (end - start)


def _enough(judgement) -> bool:
    """A judge's first answer said nothing more was needed (relevance: yes; coverage: nothing missing)."""
    return judgement == "yes" or (isinstance(judgement, dict) and not judgement.get("missing"))


def _union(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        elif hi > lo:
            merged.append((lo, hi))
    return merged


def gold_share(chunks: list[dict], spans: list[dict]) -> float | None:
    """How much of the evidence text is gold: characters inside a gold passage over all characters of
    the chunks (overlapping chunks counted once)."""
    by_source: dict[str, list[tuple[int, int]]] = {}
    for c in chunks:
        if c.get("start_char") is not None and c.get("end_char") is not None:
            by_source.setdefault(c["source_id"], []).append((c["start_char"], c["end_char"]))
    total = sum(hi - lo for ranges in by_source.values() for lo, hi in _union(ranges))
    if not total:
        return None
    gold = 0
    for source_id, ranges in by_source.items():
        golds = _union([(sp["start_char"], sp["end_char"]) for sp in spans if sp["source_id"] == source_id])
        for lo, hi in _union(ranges):
            gold += sum(max(0, min(hi, g1) - max(lo, g0)) for g0, g1 in golds)
    return gold / total


def passage_recall(chunks: list[dict], spans: list[dict], threshold: float) -> float | None:
    """Share of a case's gold passages that the chunks cover at least ``threshold`` of."""
    covered = [passage_coverage(chunks, s) for s in spans]
    covered = [c for c in covered if c is not None]
    return sum(c >= threshold for c in covered) / len(covered) if covered else None


def passage_reciprocal_rank(chunks: list[dict], spans: list[dict], threshold: float) -> float | None:
    """1 / the first rank at which the chunks so far cover a gold passage; 0 if none ever does."""
    if not spans:
        return None
    for rank in range(1, len(chunks) + 1):
        if passage_recall(chunks[:rank], spans, threshold):
            return 1.0 / rank
    return 0.0


def _mean_coverage(chunks: list[dict], spans: list[dict]) -> float | None:
    return _mean(passage_coverage(chunks, s) for s in spans)


def evaluate_retrieval(pairs: list[tuple[EvalCase, dict]], ks: tuple[int, ...], *,
                       passage_threshold: float | None = None) -> dict:
    """Retrieval-only metrics over (case, row) pairs; only cases with gold evidence count.

    recall@k: share of gold sources in the top k (a comment counts as its parent issue); hit@k: any
    gold in the top k; mrr: 1/rank of the first gold; packet_recall: recall over the chunks that
    actually reach the prompt. With ``passage_threshold`` and gold passages (evidence_spans) the same
    is measured per passage: passage_recall_at_k, passage_hit_at_k, passage_mrr,
    passage_coverage_at_k and packet_passage_recall.
    """
    with_gold = [(c, r) for c, r in pairs if c.gold_evidence]

    def block(sub):
        return {
            "n": len(sub),
            "recall_at_k": {str(k): _mean(recall_at_k(r["retrieved"], c.gold_evidence, k) for c, r in sub)
                            for k in ks},
            "hit_at_k": {str(k): _mean(float((recall_at_k(r["retrieved"], c.gold_evidence, k) or 0) > 0)
                                       for c, r in sub) for k in ks},
            "mrr": _mean(reciprocal_rank(r["retrieved"], c.gold_evidence) for c, r in sub),
            "packet_recall": _mean(recall_at_k(r["packet"], c.gold_evidence, max(len(r["packet"]), 1))
                                   for c, r in sub),
            **(passages(sub) if passage_threshold is not None else {}),
            **evidence_size(sub),
            **corrective(sub),
        }

    def corrective(sub):
        traces = [r["rerank_trace"] for _, r in sub if isinstance(r.get("rerank_trace"), dict)]
        if not traces:
            return {}
        return {"corrective_cases": sum(1 for t in traces if t.get("turns")),
                "judged_enough_first": sum(1 for t in traces if _enough((t.get("judgements") or [None])[0]))}

    def evidence_size(sub):
        tokens = [r["packet_tokens"] for _, r in sub if r.get("packet_tokens") is not None]
        if not tokens:
            return {}
        return {"packet_tokens_median": float(statistics.median(tokens)),
                "packet_chunks_mean": round(statistics.mean(len(r["packet"]) for _, r in sub), 2)}

    def passages(sub):
        spans = [(c, r) for c, r in sub if getattr(c, "evidence_spans", None)]
        if not spans:
            return {}
        return {
            "passage_recall_at_k": {str(k): _mean(passage_recall(r["retrieved"][:k], c.evidence_spans,
                                                                 passage_threshold) for c, r in spans)
                                    for k in ks},
            "passage_coverage_at_k": {str(k): _mean(_mean_coverage(r["retrieved"][:k], c.evidence_spans)
                                                    for c, r in spans) for k in ks},
            "passage_hit_at_k": {str(k): _mean(float((passage_recall(r["retrieved"][:k], c.evidence_spans,
                                                                     passage_threshold) or 0) > 0)
                                               for c, r in spans) for k in ks},
            "passage_mrr": _mean(passage_reciprocal_rank(r["retrieved"], c.evidence_spans, passage_threshold)
                                 for c, r in spans),
            "packet_passage_recall": _mean(passage_recall(r["packet"], c.evidence_spans, passage_threshold)
                                           for c, r in spans),
            "packet_gold_share": _mean(gold_share(r["packet"], c.evidence_spans) for c, r in spans),
        }
    out = {"kind": "retrieval", "n_cases": len(pairs), "n_with_gold": len(with_gold),
           "n_without_gold": len(pairs) - len(with_gold), **block(with_gold)}
    out["by_category"] = {cat: block([(c, r) for c, r in with_gold if c.category == cat])
                          for cat in sorted({c.category for c, _ in with_gold})}
    out["error_rate"] = _mean(float(r["outcome"] == "error") for _, r in pairs)
    out["budget_stopped"] = sum(r["outcome"] == "budget_stopped" for _, r in pairs)
    out["cost"] = cost_summary([r["usage"] for _, r in pairs])
    return out


def _f(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_metrics(m: dict) -> str:
    """A readable summary for the terminal (the full numbers are in summary.json)."""
    lines = []
    cats = ", ".join(f"{k} {v['n']}" for k, v in m.get("by_category", {}).items())
    lines.append(f"cases: {m['n_cases']}" + (f" ({cats})" if cats else ""))
    if m.get("kind") == "retrieval":
        lines.append(f"       {m['n_with_gold']} have gold evidence; {m['n_without_gold']} do not "
                     "(excluded from recall)")
    ks = sorted(m.get("recall_at_k", {}), key=int)
    recall = "  ".join(f"@{k} {_f(m['recall_at_k'][k])}" for k in ks)
    lines.append(f"evidence   recall {recall}   MRR {_f(m.get('mrr'))}")
    if m.get("kind") == "retrieval":
        hit = "  ".join(f"@{k} {_f(m['hit_at_k'][k])}" for k in ks)
        lines.append(f"           hit    {hit}   gold in the prompt packet {_f(m.get('packet_recall'))}")
        if m.get("passage_recall_at_k"):
            pr = "  ".join(f"@{k} {_f(m['passage_recall_at_k'].get(k))}" for k in ks)
            top = ks[-1] if ks else None
            lines.append(f"passages   recall {pr}   MRR {_f(m.get('passage_mrr'))}")
            ph = "  ".join(f"@{k} {_f(m['passage_hit_at_k'].get(k))}" for k in ks)
            lines.append(f"           hit    {ph}   coverage@{top} {_f(m['passage_coverage_at_k'].get(top))}"
                         f"   in the prompt packet {_f(m.get('packet_passage_recall'))}")
        for cat, b in m.get("by_category", {}).items():
            top = ks[-1] if ks else None
            lines.append(f"           {cat:<11} recall@{top} {_f(b['recall_at_k'].get(top))}  (n={b['n']})")
    else:
        lines.append(f"           answers cite gold evidence {_f(m.get('answer_cites_gold_rate'))}")
        acc = "  ".join(f"{k} {_f(v['outcome_accuracy'])}" for k, v in m.get("by_category", {}).items())
        lines.append(f"decisions  outcome accuracy {_f(m.get('outcome_accuracy'))}   ({acc})")
        lines.append(f"grounding  citation violations {_f(m.get('citation_violation_rate'))}   "
                     f"answers without citation {_f(m.get('answer_without_citation_rate'))}   "
                     f"errors {_f(m.get('error_rate'))}   budget stops {m.get('budget_stopped', 0)}")
    c = m["cost"]
    replay = f" (+${c['replayed_usd']:.4f} replayed, not spent)" if c.get("replayed_usd") else ""
    lines.append(f"cost       ${c['total_usd']:.4f}{replay}   {c['upstream_calls']} live calls, "
                 f"{c['cache_hits']} cache hits   tokens in/out {c['input_tokens']}/{c['output_tokens']}")
    def secs(value):
        return "n/a" if value is None else f"{value:.3f}s"
    lines.append(f"time       mean {secs(c.get('wall_seconds_mean'))}   p95 {secs(c.get('wall_seconds_p95'))}")
    return "\n".join(lines)
