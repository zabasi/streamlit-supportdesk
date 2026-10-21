#!/usr/bin/env python3
"""Conversation-only staged input adapter. No model calls, scoring or tracker writes."""
from pathlib import Path
import json,argparse,copy
ROOT=Path(__file__).resolve().parents[1]
def readl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def scenario(scenario_id,root=ROOT):
 for s in json.loads((root/'evaluator_only/multi_turn_scenarios.json').read_text()):
  if s['scenario_id']==scenario_id:return s
 raise ValueError('Unknown scenario')
def start_session(scenario_id,root=ROOT):
 s=scenario(scenario_id,root);a=next(x for x in readl(root/f"agent_inputs/{s['split']}_inputs.jsonl") if x['case_id']==s['case_id'])
 return {'scenario_id':scenario_id,'case_id':a['case_id'],'response_count':0,'status':'awaiting_assistant_1','messages':copy.deepcopy(a['initial_conversation'])+[{'role':'user','content':a['user_report']}],'input_metadata':{'created_at':a['created_at'],'reported_environment':a['reported_environment'],'title':a['title']}}
def agent_view(state):
 return {'case_id':state['case_id'],'messages':[{'role':m['role'],'content':m['content']} for m in state['messages']],'input_metadata':copy.deepcopy(state['input_metadata'])}
def record_response(state,response,expected_response_index,root=ROOT):
 if not isinstance(response,str) or not response.strip():raise ValueError('Assistant response must be non-empty')
 if state['status']=='complete':raise ValueError('Conversation already complete')
 if expected_response_index!=state['response_count']+1:raise ValueError('Duplicate or out-of-order response')
 out=copy.deepcopy(state);out['messages'].append({'role':'assistant','content':response});out['response_count']+=1
 if out['response_count']==1:
  s=scenario(out['scenario_id'],root);follow=s['turns'][1]
  if follow['release_after']!='assistant_response_to_turn_1':raise ValueError('Unsupported release condition')
  out['messages'].append({'role':'user','content':follow['body']});out['status']='awaiting_assistant_2'
 else:out['status']='complete'
 return out
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 a=sub.add_parser('start');a.add_argument('scenario_id');a.add_argument('state_file',type=Path)
 a=sub.add_parser('respond');a.add_argument('state_file',type=Path);a.add_argument('--response-file',required=True,type=Path);a.add_argument('--expected-response-index',required=True,type=int)
 args=p.parse_args()
 if args.action=='start':
  if args.state_file.exists():raise ValueError('Refusing to overwrite an existing session state')
  state=start_session(args.scenario_id)
 else:state=record_response(json.loads(args.state_file.read_text()),args.response_file.read_text(),args.expected_response_index)
 args.state_file.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');print(json.dumps(agent_view(state),ensure_ascii=False))
if __name__=='__main__':main()
