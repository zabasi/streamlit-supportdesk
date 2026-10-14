# Cluster routing (experiment)

The idea: group the chunks into topic clusters of about *n* chunks, label each cluster with keywords, and
at search time look only inside the clusters a query is about. This is cluster-based retrieval with
routing (also called topic routing or metadata pre-filtering).

## How it is built

1. `scripts/build_clusters.py` runs k-means on the chunk embeddings, making clusters of about
   `retrieval.cluster_size` chunks (100 by default, so about 7 clusters for 636 chunks). One model call per
   cluster names it and lists up to 20 keywords, from the 30 chunks nearest its centre
   (`configs/prompts/label_cluster_system.txt`); unusable output falls back to the cluster's most
   distinctive words. It writes `paths.clusters` (`data/clusters/clusters.jsonl`) and a manifest. Build once
   per index; about $0.003.
2. The `cluster_doc_slots` retriever (`retrieval/clusters.py`) is a drop-in for `advanced`'s retriever. For
   each query (the cleaned report and each concept query separately) it picks the best
   `retrieval.cluster_top` clusters, then searches with the usual places for documentation.

| Setting | Values | Meaning |
|---|---|---|
| `retrieval.cluster_route` | `keywords` (default) / `embedding` | route by the query's words among each cluster's keywords, or by meaning against each cluster's name and keywords |
| `retrieval.cluster_mode` | `filter` (default) / `boost` | search only the routed clusters, or search everything with their chunks ranked higher by `retrieval.cluster_boost` |
| `retrieval.cluster_top` | 2 | clusters each query is routed to |
| `retrieval.cluster_size` | 100 | about this many chunks per cluster (used when building) |

The defaults are the plain version of the idea: keyword routing and a hard filter. A wrong route is then a
certain miss, which is why `embedding`, `boost` and a larger `cluster_top` are there to soften it.

## Running the experiment

```bash
python scripts/build_clusters.py --mode live         # prints each cluster's size, types, label and keywords

T=$(date +%H%M)
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off --run-id adv-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set pipeline.retriever=cluster_doc_slots --run-id clu-kw-filter-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set pipeline.retriever=cluster_doc_slots --set retrieval.cluster_route=embedding --run-id clu-emb-filter-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set pipeline.retriever=cluster_doc_slots --set retrieval.cluster_route=embedding \
  --set retrieval.cluster_mode=boost --run-id clu-emb-boost-$T
supportdesk report adv-$T clu-kw-filter-$T clu-emb-filter-$T clu-emb-boost-$T --output docs/reports/cluster-routing.html
python scripts/diagnose_passages.py --pipeline advanced --mode live --set pipeline.scope_gate=off \
  --set pipeline.retriever=cluster_doc_slots --output docs/reports/funnel-cluster-routing.md
```

Only the reranking calls are new for each variant (about $0.005 each); the query rewrites replay from the
cache. In the funnel, the question is whether "page not among the candidates" goes down without "dropped by
the reranker" going up.
