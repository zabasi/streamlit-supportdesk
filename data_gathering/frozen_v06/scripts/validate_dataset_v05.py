#!/usr/bin/env python3
"""Offline v05 integrity checks. No runtime solution certification."""
from pathlib import Path
import json,hashlib,re,collections
from load_retrieval_sources import load_sources
ROOT=Path(__file__).resolve().parents[1]
def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def run():
 audit=ROOT/'audit/link_review_v05';data=json.loads((audit/'summary.json').read_text());records=load_sources();by={r['source_id']:r for r in records}
 assert len(by)==len(records)==data['source_records']
 base=[json.loads(l) for name in ['issues','comments','documents','release_sections'] for l in (audit/'baseline_v04'/f'{name}.jsonl').read_text().splitlines()]
 assert len(base)==472
 assert [r['source_id'] for r in records if r['source_id'] in {b['source_id'] for b in base}]==[b['source_id'] for b in base]
 for b in base:
  actual=by[b['source_id']];assert actual['text']==b['text'] and actual['text_sha256']==b['text_sha256']
  for k,v in b['metadata'].items():assert actual['metadata'][k]==v,(b['source_id'],k)
 for r in records:
  assert r['text_sha256']==sha(r['text'])
  for target in r['metadata'].get('linked_source_ids',[]):assert target in by
  if r['source_type']=='linked_reference':
   assert r['metadata']['embedding_allowed'] and r['metadata']['reviewer_kind']=='ai_assistant'
   assert r['metadata']['runtime_verified'] is False and r['metadata']['linked_from_source_ids']
 decisions=json.loads((audit/'link_decisions.json').read_text());inventory=json.loads((audit/'link_inventory.json').read_text())
 assert len(decisions)==len(inventory)==data['unique_reference_urls']
 assert {x['url'] for x in inventory}=={x['url'] for x in decisions}
 for d in decisions:
  assert d['status'] and d['original_url_preserved_in_text']
  for target in d['target_source_ids']:assert target in by
 added={r['source_id']:r for r in json.loads((audit/'added_sources.json').read_text())}
 spans=json.loads((audit/'selection_spans.json').read_text())
 assert len(added)==len(spans)==data['added_records']
 blocked=set(json.loads((ROOT/'audit/heldout_issue_numbers.json').read_text())['blocked_reference_numbers_including_external_PRs'])
 for s in spans:
  r=by[s['source_id']];assert r==added[r['source_id']]
  t=s['normalized_source_text'];ex=[]
  for item in s['spans']:
   piece=t[item['start_char']:item['end_char']];assert sha(piece)==item['sha256'];ex.append(piece.strip())
  assert s['prefix']+'\n\n'.join(ex)==r['text'] and s['text_sha256']==r['text_sha256']
  assert len(re.findall(r'(?m)^```[^\n]*$',r['text']))%2==0
  prose=re.sub(r'(?ms)^```[^\n]*\n.*?^```','',r['text'])
  assert not re.search(r'(?i)\b(?:thanks|thank you|hi there|hello everyone|happy streamlit|still have questions)\b',prose),r['source_id']
  assert not re.search(r'(?<![\w/])@[a-zA-Z0-9]+[-_][a-zA-Z0-9_-]+',prose),r['source_id']
  refs={int(n) for n in re.findall(r'(?:github\.com/streamlit/streamlit/(?:issues|pull)/|(?<!\w)#)(\d+)',r['text'])}
  assert not refs&blocked,(r['source_id'],refs&blocked)
  if 'pinned_original_spans' in s:
   for x in s['pinned_original_spans']:assert sha(s['pinned_original_text'][x['start_char']:x['end_char']])==x['sha256']
 for rel,h in json.loads((audit/'protected_v04_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
 for name in ['development','test']:assert len((ROOT/'agent_inputs'/f'{name}_inputs.jsonl').read_text().splitlines())==15
 checks={'unique_sources':True,'text_hashes':True,'all_472_original_texts_and_metadata_fields_preserved':True,'all_reference_urls_have_decisions':True,'all_relations_resolve':True,'all_new_text_reconstructs_from_selected_spans':True,'original_evaluation_archives_and_config_unchanged':True,'new_sources_exclude_blocked_evaluation_references':True,'new_prose_has_no_reviewed_social_phrases_or_personal_addresses':True,'fenced_code_balanced':True}
 result={'structural_status':'passed','version':'v05','source_records':len(records),'source_counts':dict(collections.Counter(r['source_type'] for r in records)),'checks':checks,'semantic_truth_certified':False,'agent_behavior_evaluated':False,'chunks_built':False,'embeddings_built':False,'baseline_v04_validation':'Passed before enrichment; original texts, evaluation and source archives remain exact.','limitations':['Link audit is first-hop and excludes URLs in protected code examples.','Unretrievable, private and deliberately unexpanded destinations are documented, not claimed as inspected.','Source claims and examples have not been executed.','Fresh API documentation declares its snapshot version and must not be assumed to describe old issues.','Link metadata requires explicit runtime support for neighbor retrieval.','No RAG quality metrics or paid project API calls were run.']}
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()
