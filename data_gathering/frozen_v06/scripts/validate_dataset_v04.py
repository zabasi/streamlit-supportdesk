#!/usr/bin/env python3
"""Offline provenance, code preservation, split integrity and hygiene checks.

PASS is not proof of semantic truth or an executed RAG quality score.
"""
import json,re,hashlib,contextlib,io
from pathlib import Path
from load_retrieval_sources import load_sources
from clean_for_chunking_v04 import FILES,rows,code_tokens,clean_record,narrative
from validate_dataset_v03 import run as legacy_validate

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def run():
 with contextlib.redirect_stdout(io.StringIO()) as captured:legacy_validate()
 result=json.loads(captured.getvalue())
 records=load_sources();prior=[r for f in FILES for r in rows(ROOT/'audit/prior_v03_retrieval'/f)]
 assert [r['source_id'] for r in records]==[r['source_id'] for r in prior]
 by={r['source_id']:r for r in records};old={r['source_id']:r for r in prior}
 audit=rows(ROOT/'audit/text_cleanup_v04.jsonl');changes={r['source_id']:r for r in audit}
 for r in records:
  prev=old[r['source_id']];assert code_tokens(r['text'])==code_tokens(prev['text']),r['source_id']
  expected,events=clean_record(prev);assert expected==r['text'],r['source_id']
  assert r['metadata']['text_cleanup']['previous_text_sha256']==prev['text_sha256']
  if r['source_type']=='comment':assert r['metadata']['allowed_use']==prev['metadata']['allowed_use']
  if expected!=prev['text']:
   a=changes[r['source_id']];assert a['before_text_sha256']==prev['text_sha256'] and a['after_text_sha256']==r['text_sha256']
   reconstructed=prev['text']
   for edit in reversed(a['edit_spans']):
    assert prev['text'][edit['before_start']:edit['before_end']]==edit['removed']
    assert expected[edit['after_start']:edit['after_end']]==edit['replacement']
    reconstructed=reconstructed[:edit['before_start']]+edit['replacement']+reconstructed[edit['before_end']:]
   assert reconstructed==expected
  else:assert r['source_id'] not in changes
  prose=narrative(r['text'])
  assert not re.search(r'(?<![\w/])@[A-Za-z0-9][A-Za-z0-9_-]*(?![\w.-])',prose),r['source_id']
  assert not re.search(r'\b(?:thank you|thanks for|hi there|sorry for)\b',prose,re.I),r['source_id']
 assert 'Thanks for reporting' not in by['comment:884690277']['text']
 assert len(changes)==read('audit/cleanup_summary_v04.json')['changed_text_records']
 for rel,sha in read('audit/v03_preserved_file_hashes.json').items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 corrections=rows(ROOT/'audit/metadata_corrections_v04.jsonl')
 assert len(corrections)==33 and len({c['source_id'] for c in corrections})==12
 for c in corrections:
  assert old[c['source_id']]['metadata'][c['field']]==c['before']
  assert by[c['source_id']]['metadata'][c['field']]==c['after'] is None
 assert by['issue:6970']['metadata']['streamlit_version'] is None
 result['text_preparation_version']='v04'
 result['cleanup_checks']={'selected_ids_and_order_unchanged':True,'comment_roles_unchanged':True,'protected_code_tokens_exactly_preserved_in_all_sources':True,'reproducible_cleanup':True,'edit_spans_reconstruct_cleaned_text':True,'reviewed_social_phrases_absent_from_narrative':True,'personal_handles_absent_outside_code_and_technical_syntax':True,'evaluation_archive_config_and_pending_bytes_unchanged':True,'metadata_corrected_fields':33,'metadata_corrected_issues':12}
 result['cleanup_summary']=read('audit/cleanup_summary_v04.json')
 result['checks'].extend(['v04 code-protected differential cleanup and edit-span reconstruction','unchanged v03 source IDs/order and comment roles','unchanged evaluator inputs/rubrics and original archive bytes','corrected empty environment field metadata','no reviewed social thanks or personal addresses in narrative'])
 if not (ROOT/'SHA256SUMS.json').exists():(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 return result

if __name__=='__main__':run()
