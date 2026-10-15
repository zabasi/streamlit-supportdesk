#!/usr/bin/env bash
# Run the retrieval comparison suite on the development cases and build one report.
#
#   bash scripts/run_retrieval_suite.sh main
#   bash scripts/run_retrieval_suite.sh hist --set 'eval.case_files=["data/eval_historical"]'
#
# The first argument names the runs (<prefix>-<config>); any further arguments (for example a historical
# copy of the cases) are passed to every run. The ladder goes from the baseline to advanced one step at a
# time; the options are each added on top of advanced. All runs are retrieval-only and keep the scope gate
# off, so they compare retrieval alone. Model calls already made replay from the cache.
set -uo pipefail
MODE="${SUITE_MODE:-live}"   # SUITE_MODE=offline checks the script without model calls
PREFIX="${1:?usage: run_retrieval_suite.sh PREFIX [extra --set options]}"
shift
EXTRA=("$@")
CLUSTERS="data/clusters/anchored.jsonl"
RUNS=()

run() {  # run NAME PIPELINE [settings...]
  local name="$1" pipeline="$2"
  shift 2
  echo "== $PREFIX-$name ($pipeline $*)"
  if supportdesk eval-retrieval --pipeline "$pipeline" --split dev --mode "$MODE" \
      --set pipeline.scope_gate=off "$@" "${EXTRA[@]}" --run-id "$PREFIX-$name"; then
    RUNS+=("$PREFIX-$name")
  else
    echo "!! $PREFIX-$name failed; it is left out of the report" >&2
  fi
}

# the ladder: each step adds one idea to the one before
run baseline        baseline
run clean-slots     clean_doc_slots
run rerank          clean_doc_slots_rerank_docs_prompt
run decompose       clean_doc_slots_rerank_docs_prompt_decompose
run merged          clean_doc_slots_rerank_docs_prompt_decompose_merged
run advanced        advanced

# options on top of advanced
run sectrim         advanced --set retrieval.complete=section_trim
run top8            advanced --set retrieval.top_k=8 --set context.max_tokens=4300
run hybrid          advanced --set pipeline.retriever=hybrid_doc_slots
if [ -f "$CLUSTERS" ]; then
  CL=(--set pipeline.retriever=cluster_doc_slots --set "paths.clusters=$CLUSTERS"
      --set retrieval.cluster_route=centroid --set retrieval.cluster_top=3)
  run clusters         advanced "${CL[@]}"
  run clusters-sectrim advanced "${CL[@]}" --set retrieval.complete=section_trim
else
  echo "-- $CLUSTERS not found: cluster runs skipped (build it with scripts/build_clusters.py)" >&2
fi

mkdir -p docs/reports
supportdesk report "${RUNS[@]}" "${EXTRA[@]}" --output "docs/reports/suite-$PREFIX.html"
for name in advanced sectrim; do
  settings=()
  [ "$name" = sectrim ] && settings=(--set retrieval.complete=section_trim)
  python scripts/diagnose_passages.py --pipeline advanced --mode "$MODE" --set pipeline.scope_gate=off \
    "${settings[@]}" "${EXTRA[@]}" --output "docs/reports/funnel-$PREFIX-$name.md"
done
echo "report: docs/reports/suite-$PREFIX.html | funnels: docs/reports/funnel-$PREFIX-{advanced,sectrim}.md"
