# Human review form (one per sampled case)

Fill one copy per case and save it as JSON, `<human dir>/<case_id>.json`, with the same fields as the
judge (`configs/prompts/judge_case_system.txt`), so `scripts/judge_agreement.py` can compare them.

1. Read the case file exported with `--no-reference` (blind). Write down the decision you would have made
   and why, before looking at the system's decision or the rubric.
2. Read the system's decision, its reply, its evidence and its summaries.
3. Read `judge_reference` in the full export, then fill the fields below.

Scores: 0 fails, 1 partly, 2 fully.

```json
{
  "case_id": "",
  "reviewer": "",
  "blind_decision": "answer | request_info | escalate",
  "evidence_quality": {"score": 0, "relevant_labels": [], "irrelevant_labels": [],
                       "labelled_evidence_present": "yes | partly | no", "unsupported_claims": [], "notes": ""},
  "decision_quality": {"stored_decision": "", "decision_acceptable": true, "score": 0,
                       "fits_available_information": "", "question_targets_required_information": "yes | partly | no | not_applicable",
                       "needless_repeated_check": false, "forbidden_or_unacceptable_claims": [],
                       "version_and_uncertainty_handling": 0, "notes": ""},
  "operational_success": {"stored_decision_is_allowed": true, "reply_matches_stored_decision": true,
                          "routing_matches_assessment": "yes | no | not_applicable",
                          "action_without_confirmation_or_duplicate": false, "notes": ""},
  "response_and_summary": {"user_response": 0, "maintainer_summary": 0,
                           "facts_reports_and_uncertainty_kept_apart": "yes | partly | no", "notes": ""},
  "failure_analysis": {"failed": false, "failure_type": "none", "likely_cause": "", "proposed_fix": "",
                       "expected_effect": "", "confidence": "low | medium | high"},
  "overall": {"verdict": "pass | partial | fail", "rationale": ""}
}
```
