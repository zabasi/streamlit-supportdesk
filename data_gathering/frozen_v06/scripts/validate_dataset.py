#!/usr/bin/env python3
"""Offline differential, exact-span and evaluation-schema checks for v06."""
from pathlib import Path
import json,hashlib,collections,re
from load_retrieval_sources import load_sources,ALLOWED
ROOT=Path(__file__).resolve().parents[1]
def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def readl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def check_evidence(e,by):
 r=by[e['source_id']];a,b=e['start_char'],e['end_char']
 assert type(a) is int and type(b) is int and 0<=a<b<=len(r['text'])
 assert e['offset_basis']=='retrieval_corpus.text' and e['end_exclusive'] is True
 assert r['text'][a:b]==e['text'],e['evidence_id']
 assert sha(e['text'])==e['evidence_sha256'] and r['text_sha256']==e['source_text_sha256']
 assert 'source_url' not in e and 'commit' not in e and 'retrieved_at' not in e
def run(root=ROOT):
 base=root/'audit/alignment_v06/baseline_v05';records=load_sources(root);by={r['source_id']:r for r in records}
 old=sum([readl(base/'retrieval_corpus'/n) for n in ALLOWED],[])
 assert len(records)==len(by)==540 and [r['source_id'] for r in records]==[r['source_id'] for r in old]
 for r,b in zip(records,old):
  assert r['text']==b['text'] and r['source_type']==b['source_type'] and r['text_sha256']==b['text_sha256']==sha(r['text'])
  assert 'source_id' not in r['metadata'] and 'source_type' not in r['metadata']
  assert 'audit_source_id' not in r['metadata'].get('text_cleanup',{})
  for k,v in b['metadata'].items():
   if k in ['source_id','source_type']:continue
   if k=='text_cleanup':v={a:b for a,b in v.items() if a!='audit_source_id'}
   assert r['metadata'][k]==v,(r['source_id'],k)
  for sid in r['metadata'].get('linked_source_ids',[]):assert sid in by
  for ref in r['metadata'].get('related_references',[]):
   assert r['text'][ref['mention_start_char']:ref['mention_end_char']]==ref['mention_text']
   if ref['target_source_id'] is not None:assert ref['target_source_id'] in by
 assert all(r['metadata']['created_at'] for r in records if r['source_type']=='issue')
 # The old evaluation, raw sources and configs remain exact in archival paths.
 protected=json.loads((root/'audit/link_review_v05/protected_v04_hashes.json').read_text())
 changed_paths={f'agent_inputs/{s}_inputs.jsonl' for s in ['development','test']}|{f'evaluator_only/{s}_rubrics.jsonl' for s in ['development','test']}|{'evaluator_only/multi_turn_scenarios.json'}
 for path,digest in protected.items():
  p=base/path if path in changed_paths else root/path
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,path
 inputs=[];golds=[]
 for s in ['development','test']:
  aa=readl(root/f'agent_inputs/{s}_inputs.jsonl');gg=readl(root/f'evaluator_only/{s}_rubrics.jsonl');bb=readl(base/f'agent_inputs/{s}_inputs.jsonl')
  assert len(aa)==len(gg)==15 and [a['case_id'] for a in aa]==[b['case_id'] for b in bb]==[g['case_id'] for g in gg]
  assert collections.Counter(g['case_category'] for g in gg)=={'answerable':6,'needs_clarification':6,'needs_escalation':3}
  for a,g,b in zip(aa,gg,bb):
   assert a['user_report']==b['user_report'] and a['user_report_sha256']==sha(a['user_report'])
   assert a['initial_conversation']==[] and a['created_at']
   assert a['case_source_id'] not in by
   issue=json.loads((root/f"evaluator_only/source_threads/{a['case_id'].split('-')[-1]}.json").read_text())['issue']
   assert a['created_at']==issue['created_at'] and a['updated_at']==issue['updated_at']
   for key,e in a['reported_environment_evidence'].items():assert a['user_report'][e['start_char']:e['end_char']]==e['text']
   for value in a['reported_environment'].values():
    assert value is None or not(value.startswith(('-', '#')) or 'get it with' in value.lower() or value=='version:')
   for e in g['acceptable_retrievable_evidence']:check_evidence(e,by)
   assert g['answerability_review']['runtime_verified'] is False
   assert g['required_missing_information'] or g['case_category']=='answerable'
   for m in g['required_missing_information']:assert m['ask_again_if_already_supplied'] is False and m['status'] in ['unknown','new_observation']
  inputs+=aa;golds+=gg
 mappings=json.loads((root/'audit/alignment_v06/evidence_span_mapping.json').read_text());assert len(mappings)==60
 canonical={d['source_id']:d for d in readl(root/'audit/input_data/previous_evaluation/retrieval_corpus/documents.jsonl')}
 for m in mappings:
  e=m['archived_canonical_evidence'];assert canonical[e['source_id']]['body'][e['start_char']:e['end_char']]==e['text'] and sha(e['text'])==e['evidence_sha256']
 mt=json.loads((root/'evaluator_only/multi_turn_scenarios.json').read_text());oldmt=json.loads((base/'evaluator_only/multi_turn_scenarios.json').read_text());assert len(mt)==10
 for s,b in zip(mt,oldmt):
  assert s['turns']==b['turns'] and s['evaluator_events']==b['evaluator_events'] and len(s['steps'])==4
  assert s['steps'][2]['release_condition']=='assistant_response_1_recorded'
  assert s['execution_status']=='scenario_specification_not_executed'
  assert len(s['workflow_event_steps'])==len(s['evaluator_events'])
 histories=readl(root/'agent_inputs/history_variant_inputs.jsonl');hgold=readl(root/'evaluator_only/history_variant_rubrics.jsonl')
 assert len(histories)==len(hgold)==6 and collections.Counter(a['split'] for a in histories)=={'development':3,'test':3}
 ib={a['case_id']:a for a in inputs};gb={g['case_id']:g for g in golds};sm={s['case_id']:s for s in mt}
 for a,g in zip(histories,hgold):
  orig=ib[a['base_case_id']];assert a['split']==orig['split']==g['split'] and len(a['initial_conversation'])==2
  assert a['initial_conversation'][0]['content']==orig['user_report'] and a['initial_conversation'][1]['origin']=='designed_assistant_fixture_not_actual_model_output'
  assert a['user_report']==sm[a['base_case_id']]['turns'][1]['body'] and sha(a['user_report'])==a['user_report_sha256']
  for e in g['acceptable_retrievable_evidence']:check_evidence(e,by)
  for m in g['required_missing_information']:assert m['information_id'] not in g['resolved_information_ids']
 si=readl(root/'agent_inputs/source_usage_inputs.jsonl');sg=readl(root/'evaluator_only/source_usage_rubrics.jsonl')
 assert len(si)==len(sg)==8 and collections.Counter(x['split'] for x in si)=={'development':4,'test':4}
 assert collections.Counter(g['target_source_type'] for g in sg)=={'issue':2,'comment':2,'release_section':2,'linked_reference':2}
 for a,g in zip(si,sg):
  assert a['case_id']==g['case_id'] and a['split']==g['split']
  for e in g['acceptable_retrievable_evidence']:check_evidence(e,by)
 # Supplementary source targets are distinct across the development/test subsets.
 targets={s:{g['acceptable_retrievable_evidence'][0]['source_id'] for g in sg if g['split']==s} for s in ['development','test']}
 assert not targets['development']&targets['test']
 gold_keys={'acceptable_retrievable_evidence','required_missing_information','oracle_only_comments','oracle_only_direct_release_notes','reference_reasoning','expected_next_step','forbidden_claims','answerability_review','claim_support'}
 for a in inputs+histories+si:assert not gold_keys&set(a)
 blocked=json.loads((root/'audit/heldout_issue_numbers.json').read_text())
 held={int(a['case_id'].split('-')[-1]) for a in inputs}
 assert not {r['metadata']['number'] for r in records if r['source_type']=='issue'}&held
 summary=json.loads((root/'audit/alignment_v06/summary.json').read_text())
 assert summary['missing_information_items']==sum(len(g['required_missing_information']) for g in golds)==30
 assert summary['base_evidence_spans']==sum(len(g['acceptable_retrievable_evidence']) for g in golds)==64
 result={'version':'v06','structural_status':'passed','source_records':540,'checks':{'all_540_v05_texts_and_code_unchanged':True,'stable_source_ids_and_roles':True,'no_runtime_duplicate_source_ids':True,'all_64_base_evidence_spans_exact_in_corpus':True,'all_60_archived_evidence_spans_preserved':True,'source_hash_and_evidence_hash_guards':True,'all_30_original_case_reports_preserved':True,'dates_and_environment_evidence_resolve':True,'base_15_plus_15_and_category_balance_preserved':True,'all_10_scenarios_have_step_rubrics':True,'six_history_variants_same_split_as_base':True,'eight_supplementary_tasks_cover_four_source_types':True,'no_gold_in_agent_inputs':True,'heldout_case_sources_not_in_index':True,'archived_raw_sources_configs_and_old_evaluation_preserved':True,'related_reference_spans_and_target_ids_resolve':True},'corpus_coverage_review':'30 bounded next-step rubrics reviewed by AI assistant; historical resolution and end-to-end agent behavior not verified','chunks_built':False,'embeddings_built':False,'runtime_solutions_verified':False,'tracker_workflow_tests_executed':False,'human_reviewed':False}
 return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
