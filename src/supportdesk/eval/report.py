"""A self-contained HTML comparison of finished runs (escaped, no scripts, no external resources)."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

CSS = """body{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1c2330}
table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #dfe3ea;padding:6px 10px;
text-align:left;vertical-align:top}th{background:#f1f3f7}td.num{text-align:right;font-variant-numeric:tabular-nums}
.override{color:#a15c00;font-weight:600}.warn{background:#fff1dc;border-left:4px solid #a15c00;padding:8px 12px;margin:12px 0}.muted{color:#5d6879}
code{background:#f1f3f7;padding:0 4px;border-radius:3px}"""

#: (label, path into summary["metrics"], kind). kind: pct | num | usd | sec
ROWS = [
    ("Cases", ("n_cases",), "num"),
    ("Cases with gold evidence", ("n_with_gold",), "num"),
    ("Outcome accuracy (all)", ("outcome_accuracy",), "pct"),
    ("MRR", ("mrr",), "num"),
    ("Gold evidence in the prompt packet", ("packet_recall",), "pct"),
    ("Answers cite gold evidence", ("answer_cites_gold_rate",), "pct"),
    ("Cases with citation violations", ("citation_violation_rate",), "pct"),
    ("Answers without citation", ("answer_without_citation_rate",), "pct"),
    ("Error rate", ("error_rate",), "pct"),
    ("Stopped by budget", ("budget_stopped",), "num"),
    ("Cost (real spend)", ("cost", "total_usd"), "usd"),
    ("Cost replayed (not spent)", ("cost", "replayed_usd"), "usd"),
    ("Upstream calls", ("cost", "upstream_calls"), "num"),
    ("Input tokens", ("cost", "input_tokens"), "num"),
    ("Output tokens", ("cost", "output_tokens"), "num"),
    ("Mean time per case (s)", ("cost", "wall_seconds_mean"), "sec"),
    ("p95 time per case (s)", ("cost", "wall_seconds_p95"), "sec"),
]


def _dig(data: dict, path: tuple):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "usd":
        return f"${value:.6f}"
    if kind == "sec":
        return f"{value:.3f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _flatten(data: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _load_run(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (directory / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"summary": summary, "rows": {r["case_id"]: r for r in rows}, "order": [r["case_id"] for r in rows]}


def _override_note(summary: dict) -> str:
    """A visible warning on runs whose retriever was overridden: they are not the pipeline as defined."""
    own = summary.get("retriever_override_of")
    if not own:
        return ""
    return (f"<br><span class='override'>retriever override: {escape(str(summary.get('retriever')))} "
            f"(own: {escape(str(own))})</span>")


def render_report(run_dirs: list[Path]) -> str:
    runs = [_load_run(Path(d)) for d in run_dirs]
    titles = [f"{escape(r['summary']['pipeline'])}<br><span class='muted'>{escape(r['summary']['run_id'])}"
              f"<br>{escape(r['summary']['mode'])}</span>{_override_note(r['summary'])}" for r in runs]
    head = "".join(f"<th>{t}</th>" for t in titles)
    out = [f"<!doctype html><meta charset=utf-8><title>supportdesk report</title><style>{CSS}</style>",
           "<h1>supportdesk: run comparison</h1>"]
    if any(r["summary"].get("kind") == "retrieval" for r in runs):
        out.append("<div class=warn><b>Some runs are retrieval-only.</b> No answers were generated, so "
                   "decision and grounding rows show &mdash; for them.</div>")
    if any(r["summary"]["mode"] != "live" for r in runs):
        out.append("<div class=warn><b>Not all runs are live.</b> Offline runs use deterministic fakes "
                   "and prove the plumbing, not answer quality. Replay numbers reuse recorded real "
                   "responses; their cost is shown separately and was not spent again.</div>")
    stage_names = [k for r in runs for k in r["summary"].get("stages", {})]
    if stage_names:
        out.append("<h2>Configuration</h2><p class=muted>Which implementation filled each stage. "
                   "Runs that differ in one row are an ablation of that stage.</p>"
                   f"<table><tr><th>Stage</th>{head}</tr>")
        for stage in dict.fromkeys(stage_names):  # first-seen order
            cells = "".join(f"<td>{escape(str(r['summary'].get('stages', {}).get(stage, '—')))}</td>"
                            for r in runs)
            out.append(f"<tr><td>{escape(stage)}</td>{cells}</tr>")
        out.append("</table>")
    flat = [_flatten(r["summary"].get("settings", {}).get("config", {})) for r in runs]
    differing = [k for k in dict.fromkeys(k for f in flat for k in f)
                 if not k.startswith("paths.") and len({json.dumps(f.get(k)) for f in flat}) > 1]
    if len(runs) > 1 and differing:
        out.append("<h2>Settings that differ</h2><p class=muted>Every other setting is identical "
                   f"across these runs.</p><table><tr><th>Setting</th>{head}</tr>")
        for key in differing:
            cells = "".join(f"<td>{escape(str(f.get(key, '—')))}</td>" for f in flat)
            out.append(f"<tr><td><code>{escape(key)}</code></td>{cells}</tr>")
        out.append("</table>")
    out.append(f"<h2>Metrics</h2><table><tr><th>Metric</th>{head}</tr>")
    ks = sorted({int(k) for r in runs for k in r["summary"]["metrics"].get("recall_at_k", {})})
    hit_ks = sorted({int(k) for r in runs for k in r["summary"]["metrics"].get("hit_at_k", {})})
    rows = (ROWS[:3] + [(f"Recall@{k}", ("recall_at_k", str(k)), "pct") for k in ks]
            + [(f"Hit@{k}", ("hit_at_k", str(k)), "pct") for k in hit_ks] + ROWS[3:])
    for label, path, kind in rows:
        cells = "".join(f"<td class=num>{_fmt(_dig(r['summary']['metrics'], path), kind)}</td>" for r in runs)
        out.append(f"<tr><td>{escape(label)}</td>{cells}</tr>")
    out.append("</table><h2>By category</h2>"
               "<p class=muted>Decision quality is reported per category so refusing everything, or "
               "answering everything, cannot look like success. Retrieval-only runs show recall.</p>"
               f"<table><tr><th>Category</th>{head}</tr>")
    categories = sorted({c for r in runs for c in r["summary"]["metrics"].get("by_category", {})})
    for cat in categories:
        cells = ""
        for r in runs:
            data = r["summary"]["metrics"].get("by_category", {}).get(cat)
            if not data:
                cells += "<td class=num>—</td>"
            elif "outcome_accuracy" in data:
                cells += f"<td class=num>{_fmt(data['outcome_accuracy'], 'pct')} (n={data['n']})</td>"
            else:  # retrieval-only run: show recall at its largest k instead
                top = max(data["recall_at_k"], key=int)
                cells += (f"<td class=num>recall@{top} {_fmt(data['recall_at_k'][top], 'pct')} "
                          f"(n={data['n']})</td>")
        out.append(f"<tr><td>{escape(cat)}</td>{cells}</tr>")
    out.append("</table><h2>Cases</h2>"
               f"<table><tr><th>Case</th><th>Category</th>{head}</tr>")
    for case_id in runs[0]["order"]:
        cells = ""
        for r in runs:
            row = r["rows"].get(case_id)
            cells += "<td>—</td>" if row is None else (
                f"<td>{escape(row['outcome'])}"
                + (f"<br><span class=muted>cites {len(row['citations'])}</span>" if row["citations"] else "")
                + (f"<br><span class=muted>⚠ {escape(', '.join(row['violations']))}</span>" if row["violations"] else "")
                + (f"<br><span class=muted>{escape(row['error'])}</span>" if row.get("error") else "")
                + "</td>")
        out.append(f"<tr><td><code>{escape(case_id)}</code></td>"
                   f"<td>{escape(runs[0]['rows'][case_id].get('category', ''))}</td>{cells}</tr>")
    out.append("</table>")
    return "".join(out)


def write_report(run_dirs: list[Path], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(run_dirs), encoding="utf-8")
