#!/usr/bin/env python3
"""Offline contract tests, using fixtures and placeholder responses only."""
from pathlib import Path
import json,copy,hashlib
from evaluation_session import start_session,agent_view,record_response
from map_evidence_to_chunks import map_evidence
from validate_dataset import run,check_evidence
ROOT=Path(__file__).resolve().parents[1]
def rejects(fn):
 try:fn()
 except (ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
def main():
 tests=0
 for s in json.loads((ROOT/'evaluator_only/multi_turn_scenarios.json').read_text()):
  state=start_session(s['scenario_id']);view=agent_view(state);assert len(view['messages'])==1 and view['messages'][0]['role']=='user'
  assert set(view)=={'case_id','messages','input_metadata'}
  # Future follow-up is not stored or disclosed until response 1 is recorded.
  assert 'turns' not in state and 'steps' not in state and 'oracle_only_comments' not in state
  nextstate=record_response(state,'Offline fixture response one',1);assert state['response_count']==0
  assert len(agent_view(nextstate)['messages'])==3 and nextstate['messages'][-1]['content']==s['turns'][1]['body']
  rejects(lambda:record_response(nextstate,'Duplicate response',1))
  done=record_response(nextstate,'Offline fixture response two',2);assert done['status']=='complete'
  rejects(lambda:record_response(done,'Extra response',3));tests+=1
 text='سلام abc\nکد 123';digest=hashlib.sha256(text.encode()).hexdigest();source={'source_id':'fixture:unicode','text':text,'text_sha256':digest}
 e={'evidence_id':'fixture:evidence','source_id':source['source_id'],'start_char':2,'end_char':11,'text':text[2:11],'source_text_sha256':digest,'evidence_sha256':hashlib.sha256(text[2:11].encode()).hexdigest(),'offset_basis':'retrieval_corpus.text','end_exclusive':True}
 def chunk(cid,a,b):return {'chunk_id':cid,'source_id':source['source_id'],'start_char':a,'end_char':b,'text':text[a:b],'source_text_sha256':digest}
 chunks=[chunk('fixture:a',0,6),chunk('fixture:b',5,len(text))];m=map_evidence([source],[e],chunks)[0];assert m['complete_span_covered'] and m['covered_characters']==9 and len(m['chunk_overlaps'])==2;tests+=1
 m=map_evidence([source],[e],[chunk('fixture:a',0,6)])[0];assert not m['complete_span_covered'] and m['covered_characters']==4;tests+=1
 bad=copy.deepcopy(chunks);bad[0]['source_text_sha256']='stale';rejects(lambda:map_evidence([source],[e],bad));tests+=1
 bad=copy.deepcopy(chunks);bad[0]['text']='wrong';rejects(lambda:map_evidence([source],[e],bad));tests+=1
 bad=copy.deepcopy(e);bad['start_char']+=1;rejects(lambda:check_evidence(bad,{source['source_id']:source}));tests+=1
 bad=copy.deepcopy(e);bad['source_text_sha256']='stale';rejects(lambda:check_evidence(bad,{source['source_id']:source}));tests+=1
 run();tests+=1
 result={'contract_checks_passed':tests,'staged_scenarios_checked':10,'responses':'explicit offline placeholders, not model answers','fixtures_only':True,'real_chunks_built':False,'agent_quality_evaluated':False,'tracker_workflow_executed':False}
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
