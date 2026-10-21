# Corpus and evaluation alignment v06

540 selected source records. Every v05 text and text hash is unchanged. Only runtime metadata and evaluator schemas were revised. No chunking, embedding, network collection, paid API call or execution of source examples was performed.



{
  "version": "v06",
  "source_records": 540,
  "source_texts_changed": 0,
  "source_counts": {
    "issue": 126,
    "comment": 214,
    "documentation": 54,
    "release_section": 111,
    "linked_reference": 35
  },
  "base_cases": 30,
  "base_split_counts": {
    "development": 15,
    "test": 15
  },
  "mapped_inherited_evidence_spans": 60,
  "base_evidence_spans": 64,
  "additional_official_spans": 4,
  "coverage_cases_reviewed": 30,
  "bounded_next_step_supported": 30,
  "confirmed_historical_fixes": 0,
  "missing_information_items": 30,
  "environment_metadata_changes": 21,
  "case_dates_added": 30,
  "metadata_source_id_duplicates_removed": 540,
  "cleanup_audit_id_duplicates_removed": 472,
  "related_reference_edges": 202,
  "issues_with_related_issue_numbers": 15,
  "multi_turn_scenarios": 10,
  "history_variants": 6,
  "history_split_counts": {
    "development": 3,
    "test": 3
  },
  "supplementary_source_use_tasks": 8,
  "supplementary_split_counts": {
    "development": 4,
    "test": 4
  },
  "chunks_built": false,
  "embeddings_built": false,
  "runtime_solution_tests_performed": false,
  "paid_api_calls": 0,
  "human_reviewed": false
}

The exact 60 inherited spans are mapped to corpus text and four existing official API spans supplement the cases for data-editor keys, date bounds, chat callbacks and scheduled fragments. All 30 cases permit a bounded next step; none requires an unavailable exact fixed release. Source-use tasks and history fixtures are reported separately from the 30 base cases. Runtime solution verification and tracker integration remain unexecuted.
