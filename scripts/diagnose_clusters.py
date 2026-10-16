"""Judge a clustering before spending on retrieval runs: format bias, where the gold sits, routing reach.

For the clusters in paths.clusters (scripts/build_clusters.py) and the development cases:
  - format bias: each cluster's mix of source types; a cluster that is nearly all one type (all comments,
    all release notes) groups by writing style, not by topic
  - gold co-location: which clusters hold each case's gold passages
  - routing reach: the share of gold passages whose cluster is among those the pipeline's own queries are
    routed to, with the current retrieval.cluster_* settings. The queries come from the pipeline's query
    step with the same settings, so they are exactly what a run searches with: the cleaned report alone
    while retrieval.decompose is false, the report and the concept queries while it is true. With a hard
    filter (cluster_mode = filter), a gold passage outside the routed clusters cannot be found.
In replay mode (the default) any model call comes from the cache, so it costs nothing.

    python scripts/diagnose_clusters.py
    python scripts/diagnose_clusters.py --set retrieval.cluster_route=centroid --set retrieval.cluster_top=3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from supportdesk.config import Settings
from supportdesk.eval.cases import load_cases
from supportdesk.llm.factory import build_clients
from supportdesk.registry import RETRIEVERS, make_pipeline
from supportdesk.retrieval.index import VectorIndex


def diagnose(settings: Settings, index: VectorIndex, clients, cases) -> dict:
    """The three measurements, as numbers (main prints them)."""
    # groups: clusters (paths.clusters, one per chunk) or topic classes (paths.class_labels with
    # pipeline.retriever = class_doc_slots, one or more per chunk); each group is a set of chunk positions
    if settings.pipeline.retriever == "class_doc_slots":
        router = RETRIEVERS["class_doc_slots"](index, clients, settings)
        groups = {name: set(members) for name, members in router.members.items()}
        names = {name: name for name in groups}
    else:
        router = RETRIEVERS["cluster_doc_slots"](index, clients, settings)
        groups = {n: set(router.members[n]) for n in range(len(router.clusters))}
        names = {n: row["label"] for n, row in enumerate(router.clusters)}
    groups_of: dict[int, set] = {}
    for name, members in groups.items():
        for i in members:
            groups_of.setdefault(i, set()).add(name)
    clusters = []
    for name, members in groups.items():
        if not members:
            continue
        types = Counter(index.chunks[i].source_type for i in members)
        clusters.append({"cluster": name, "label": names[name], "size": len(members), "types": dict(types),
                         "documentation": types.get("documentation", 0),
                         "main_type_share": max(types.values()) / len(members)})
    by_source: dict[str, list[int]] = {}
    for i, c in enumerate(index.chunks):
        by_source.setdefault(c.source_id, []).append(i)
    # the query step of the pipeline being diagnosed, built from the same settings, so the routed queries
    # are exactly those a run searches with (one cleaned report while decompose is off)
    writer = make_pipeline(settings.pipeline.name, "dense_doc_slots", index, clients, settings).query
    rows, reached, passages = [], 0, 0
    for case in cases:
        gold = []  # the groups of the chunk holding most of each gold passage
        for span in case.evidence_spans:
            a, b = span["start_char"], span["end_char"]
            overlap = {i: max(0, min(b, index.chunks[i].end_char) - max(a, index.chunks[i].start_char))
                       for i in by_source.get(span["source_id"], [])}
            best = max(overlap, key=overlap.get, default=None)
            if best is not None and overlap[best] > 0:
                gold.append(groups_of.get(best, set()))
        routed = set()
        queries = writer(case.input, {})
        for query in queries:
            vector = clients.embedding.embed([query], stage="query")[0][0]
            routed.update(router.routes(query, vector))
        hit = sum(bool(g & routed) for g in gold)  # reachable when any of its groups is routed
        reached, passages = reached + hit, passages + len(gold)
        rows.append({"case": case.case_id, "queries": len(queries),
                     "gold_clusters": sorted(set().union(*gold), key=str) if gold else [],
                     "routed": sorted(routed, key=str),
                     "reached": hit, "passages": len(gold)})
    return {"clusters": clusters, "cases": rows, "reach": reached / passages if passages else 0.0,
            "mean_main_type_share": sum(c["main_type_share"] for c in clusters) / len(clusters),
            "clusters_without_documentation": sum(c["documentation"] == 0 for c in clusters)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["replay", "live", "offline"], default="replay",
                        help="replay (default) reuses advanced's cached queries; live writes any that are missing")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--classes", action="store_true",
                        help="diagnose the topic labels of paths.class_labels (class_doc_slots) instead of clusters")
    args = parser.parse_args(argv)
    settings = Settings.load(overrides=[f"run.mode={args.mode}", "pipeline.name=advanced",
                                        "pipeline.retriever=" + ("class_doc_slots" if args.classes
                                                                 else "cluster_doc_slots"), *args.set])
    clients = build_clients(settings)
    index = VectorIndex.load(settings.paths.index_dir, expect_model=clients.embedding_model)
    cases = load_cases(*settings.eval.case_files, split=args.split)
    r = settings.retrieval
    found = diagnose(settings, index, clients, cases)
    if args.classes:
        print(f"topic classes: {settings.paths.class_labels} | route={r.class_route} top={r.class_top} | {len(cases)} cases\n")
    else:
        print(f"clusters: {settings.paths.clusters} | route={r.cluster_route} top={r.cluster_top} | {len(cases)} cases\n")
    print("format bias (a good topic cluster mixes source types):")
    for c in found["clusters"]:
        kinds = ", ".join(f"{t} {n}" for t, n in sorted(c["types"].items(), key=lambda kv: -kv[1]))
        print(f"  {c['cluster']!s:>2}: {c['size']:>3} chunks, {c['main_type_share']:>4.0%} one type ({kinds}) | {c['label']}")
    print(f"  mean share of the main type: {found['mean_main_type_share']:.0%} | "
          f"clusters without documentation: {found['clusters_without_documentation']}\n")
    decompose = "on: the report and the concept queries" if r.decompose else "off: the cleaned report alone"
    print(f"queries routed per case: the pipeline's own (decompose {decompose})")
    print("gold and routing per case (gold clusters -> clusters the queries are routed to):")
    for row in found["cases"]:
        mark = "ok" if row["reached"] == row["passages"] else "MISSES GOLD"
        print(f"  {row['case']:>10}: {row['queries']} quer{'y' if row['queries'] == 1 else 'ies'}, "
              f"gold in {row['gold_clusters']} -> routed {row['routed']} | "
              f"{row['reached']}/{row['passages']} passages reachable {mark}")
    spread = Counter(len(row["gold_clusters"]) for row in found["cases"])
    print(f"\ncases whose gold sits in 1 / 2 / 3+ clusters: {spread.get(1, 0)} / {spread.get(2, 0)} / "
          f"{sum(n for k, n in spread.items() if k >= 3)}")
    print(f"routing reach: {found['reach']:.0%} of gold passages are in a routed cluster "
          "(with cluster_mode = filter, the rest cannot be found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
