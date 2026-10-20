"""Compare two passage funnels (scripts/diagnose_passages.py output) side by side, as one HTML page.

For each run: the summary (how many gold passages end up in the prompt, dropped by the reranker, on the
right page but another chunk, or never among the candidates). Then one row per gold passage with where it
ends up in each run; passages whose outcome differs come first and are highlighted (gained: now in the
prompt; lost: no longer in the prompt; moved: lost in both, at another stage).

    python scripts/compare_funnels.py artifacts/reports/funnel-advanced.md artifacts/reports/funnel-questions-15.md \
      --names "advanced" "questions, 15 sections" --output artifacts/reports/funnels.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

IN_PROMPT = "in the prompt"


def read_funnel(path: Path) -> tuple[dict[str, int], dict[tuple[str, str, str], dict]]:
    """The summary counts and the passages (keyed by case, page and opening words) of one funnel file."""
    summary, passages, table = {}, {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Where the gold passage ends up":
            table = "summary"
        elif cells[0] == "Case":
            table = "passages"
        elif table == "summary" and len(cells) >= 2:
            summary[cells[0]] = int(cells[1])
        elif table == "passages" and len(cells) >= 6:
            case, page, start, where, rank, cover = cells[:6]
            key = (case, page, start)
            n = 2
            while key in passages:  # the same opening words twice in one case: keep both
                key, n = (case, page, f"{start} ({n})"), n + 1
            passages[key] = {"where": where, "rank": rank, "cover": cover}
    return summary, passages


def change(a: str | None, b: str | None) -> str:
    if a == b:
        return "same"
    if b == IN_PROMPT:
        return "gained"
    if a == IN_PROMPT:
        return "lost"
    return "moved"


def render(names: tuple[str, str], first, second) -> str:
    (sa, pa), (sb, pb) = first, second
    stages = list(dict.fromkeys([*sa, *sb]))
    rows = []
    for key in sorted(set(pa) | set(pb), key=lambda k: (k[0], k[1], k[2])):
        a, b = pa.get(key, {}), pb.get(key, {})
        rows.append((change(a.get("where"), b.get("where")), key, a, b))
    order = {"gained": 0, "lost": 1, "moved": 2, "same": 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    counts = {k: sum(r[0] == k for r in rows) for k in order}
    e = html.escape
    head = "".join(f"<th>{e(n)}</th>" for n in names)
    summary = "".join(f"<tr><td>{e(s)}</td><td class=num>{sa.get(s, 0)}</td><td class=num>{sb.get(s, 0)}</td>"
                      f"<td class=num>{sb.get(s, 0) - sa.get(s, 0):+d}</td></tr>" for s in stages)
    body = []
    for kind, (case, page, start), a, b in rows:
        body.append(f"<tr class={kind}><td>{kind}</td><td><code>{e(case)}</code></td><td>{e(page)}</td>"
                    f"<td>{e(start)}</td><td>{e(a.get('where', '-'))}</td><td>{e(b.get('where', '-'))}</td>"
                    f"<td class=num>{e(a.get('rank', '-'))}</td><td class=num>{e(b.get('rank', '-'))}</td></tr>")
    return f"""<!doctype html><meta charset=utf-8><title>Passage funnels: {e(names[0])} vs {e(names[1])}</title>
<style>
:root{{--bg:#fff;--fg:#1c2330;--line:#dfe3ea;--head:#f1f3f7;--gain:#e6f4ea;--lose:#fdecea;--move:#fff6e0;--muted:#5d6879}}
@media (prefers-color-scheme: dark){{:root{{--bg:#14171c;--fg:#e4e7ec;--line:#2c323b;--head:#1d222a;--gain:#16321f;
--lose:#3a1c1c;--move:#352c14;--muted:#9aa3b0}}}}
body{{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;background:var(--bg);color:var(--fg)}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid var(--line);padding:5px 9px;
text-align:left;vertical-align:top}}th{{background:var(--head)}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tr.gained td{{background:var(--gain)}}tr.lost td{{background:var(--lose)}}tr.moved td{{background:var(--move)}}
.muted{{color:var(--muted)}}.wrap{{overflow-x:auto}}code{{font-size:13px}}
</style>
<h1>Passage funnels</h1>
<p class=muted>Where each gold passage ends up in two runs. Gained: now in the prompt. Lost: no longer in the
prompt. Moved: lost in both runs, at a different stage.</p>
<h2>Summary</h2><div class=wrap><table><tr><th>Where the gold passage ends up</th>{head}<th>Change</th></tr>{summary}</table></div>
<p><b>{counts['gained']}</b> gained, <b>{counts['lost']}</b> lost, <b>{counts['moved']}</b> moved between loss stages,
{counts['same']} unchanged.</p>
<h2>Every gold passage</h2><div class=wrap><table><tr><th>Change</th><th>Case</th><th>Page</th><th>Passage starts with</th>
<th>{e(names[0])}</th><th>{e(names[1])}</th><th>Doc rank ({e(names[0])})</th><th>Doc rank ({e(names[1])})</th></tr>
{''.join(body)}</table></div>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--names", nargs=2, default=None, metavar=("FIRST", "SECOND"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    names = tuple(args.names or (args.first.stem, args.second.stem))
    page = render(names, read_funnel(args.first), read_funnel(args.second))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
