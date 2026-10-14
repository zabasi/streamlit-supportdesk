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

## Clustering by topic rather than by writing style

On our corpus, plain k-means grouped the chunks by source type: all release notes in one cluster, comments
in another, the documentation in three. Issues, comments, release notes and documentation are each written
in their own style, and the embeddings carry that more strongly than the subject; and since the corpus is
about one area (Session State and widgets), the topics differ little. With a hard filter, a report was often
routed to clusters with no documentation at all. `retrieval.cluster_method` addresses this:

| `cluster_method` | How the clusters are made |
|---|---|
| `kmeans` | k-means on the embeddings as they are (the first version) |
| `centered` | each source type's average embedding is removed first, so what remains is mostly the subject |
| `doc_anchored` | the documentation is clustered alone (centered), then every other chunk joins its nearest documentation cluster: every cluster is a topic that holds documentation |

`retrieval.cluster_route = centroid` routes a query by the average of each cluster's own chunk vectors (in the
same space as the search) instead of by the embedding of its written name and keywords.

## Judging a clustering before running retrieval

`scripts/diagnose_clusters.py` measures a clustering with no reranking calls: each cluster's mix of source
types (format bias), which clusters hold each case's gold passages, and the routing reach, the share of gold
passages whose cluster is among those advanced's queries are routed to. In replay mode (the default) the
queries come from the cache, so it is free:

```bash
python scripts/diagnose_clusters.py
python scripts/diagnose_clusters.py --set retrieval.cluster_route=centroid --set retrieval.cluster_top=3
```

Only clusterings with little format bias and a high routing reach are worth a retrieval run.

## With structure-aware chunks

The `token_structure` chunker keeps sections and code blocks whole, so its documentation chunks follow
topics. Build its index and clusters in their own files, then compare advanced on that index with and
without routing:

```bash
supportdesk index --mode live --set chunking.method=token_structure --set paths.index_dir=index/token_structure
python scripts/build_clusters.py --mode live --set paths.index_dir=index/token_structure \
  --set paths.clusters=data/clusters/token_structure_anchored.jsonl --set retrieval.cluster_method=doc_anchored
python scripts/diagnose_clusters.py --set chunking.method=token_structure --set paths.index_dir=index/token_structure \
  --set paths.clusters=data/clusters/token_structure_anchored.jsonl --set retrieval.cluster_route=centroid
```

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
