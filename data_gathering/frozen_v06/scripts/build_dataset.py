from pathlib import Path
import json,hashlib,re,shutil,collections,difflib
BASE=Path('/workspace/scratch/3726f16581db')
OLD=BASE/'tmp/v06_work/Streamlit_Session_State_Reviewed_v05'
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else BASE/'output/Streamlit_Session_State_Reviewed_v06'
NAMES=['issues','comments','documents','release_sections','linked_references']
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def readl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def writej(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def writel(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in v),encoding='utf-8')
if not ROOT.exists():
 shutil.copytree(OLD,ROOT,ignore=shutil.ignore_patterns('__pycache__'))
 baseline=ROOT/'audit/alignment_v06/baseline_v05'
 for d in ['retrieval_corpus','agent_inputs']:
  shutil.copytree(OLD/d,baseline/d)
 for f in ['development_rubrics.jsonl','test_rubrics.jsonl','multi_turn_scenarios.json']:
  p=baseline/'evaluator_only'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/'evaluator_only'/f,p)
 for f in ['manifest.json','validation.json','README.md','REVIEW_REPORT.md','Embedding_Structure_v05.json','SHA256SUMS.json']:
  if (OLD/f).exists():shutil.copy2(OLD/f,baseline/f)
 for n in ['build_dataset','validate_dataset']:
  shutil.copy2(ROOT/'scripts'/f'{n}.py',ROOT/'scripts'/f'{n}_v05.py')
 shutil.copy2(BASE/'tmp/v06_case_review.json',ROOT/'config/evaluation_case_review_v06.json')
 shutil.copy2(Path(__file__),ROOT/'scripts/build_dataset.py')
 # Root reports are regenerated; inherited versions remain in the baseline archive.
 if (ROOT/'Embedding_Structure_v05.json').exists():(ROOT/'Embedding_Structure_v05.json').unlink()
else:
 # The bundled builder is portable and reads the immutable baseline.
 pass
if Path(__file__).parent.name=='scripts':ROOT=Path(__file__).resolve().parents[1]
baseline=ROOT/'audit/alignment_v06/baseline_v05'
records=sum([readl(baseline/f'retrieval_corpus/{n}.jsonl') for n in NAMES],[])
by={r['source_id']:r for r in records};removed=[];relations=[]
blocked=set(json.loads((ROOT/'audit/heldout_issue_numbers.json').read_text())['blocked_reference_numbers_including_external_PRs'])
def mask_code(t):
 t=re.sub(r'(?ms)^```[^\n]*\n.*?^```[^\n]*',lambda m:'\n'.join(' '*len(x) for x in m[0].split('\n')),t)
 return re.sub(r'`[^`\n]*`',lambda m:' '*len(m[0]),t)
for r in records:
 m=r['metadata'];fields=[]
 for key in ['source_id','source_type']:
  if key in m:fields.append('metadata.'+key);m.pop(key)
 if 'audit_source_id' in m.get('text_cleanup',{}):m['text_cleanup'].pop('audit_source_id');fields.append('metadata.text_cleanup.audit_source_id')
 removed.append({'source_id':r['source_id'],'removed_runtime_fields':fields})
 t=mask_code(r['text']);refs={}
 for x in re.finditer(r'https?://github\.com/streamlit/streamlit/(issues|pull)/(\d+)',t):
  kind='issue' if x[1]=='issues' else 'pull_request';num=int(x[2]);refs[(kind,num)]=(x.start(),x.end())
 url_numbers={n for _,n in refs}
 for x in re.finditer(r'(?<![\w/])#(\d+)\b',t):
  num=int(x[1])
  if num not in url_numbers:refs[('issue',num)]=(x.start(),x.end())
 own=m.get('number',m.get('issue_number'))
 related=[]
 for (kind,num),(a,b) in sorted(refs.items(),key=lambda x:x[1][0]):
  if num==own and kind=='issue':continue
  target=f'issue:{num}' if kind=='issue' else None
  status='corpus_present' if target in by else ('excluded_evaluation_reference' if num in blocked else 'not_in_corpus')
  rel={'number':num,'target_kind':kind,'relation_type':'explicit_mention_not_same_cause_or_verified_fix','target_source_id':target if target in by else None,'target_status':status,'mention_start_char':a,'mention_end_char':b,'mention_text':r['text'][a:b]}
  related.append(rel);relations.append({'source_id':r['source_id'],**rel})
 if related:m['related_references']=related
 if r['source_type']=='issue':m['related_issue_numbers']=[v['number'] for v in related if v['target_kind']=='issue']
for n in NAMES:writel(ROOT/f'retrieval_corpus/{n}.jsonl',[r for r in records if {'issues':'issue','comments':'comment','documents':'documentation','release_sections':'release_section','linked_references':'linked_reference'}[n]==r['source_type']])
writej(ROOT/'audit/alignment_v06/runtime_metadata_changes.json',removed)
writej(ROOT/'audit/alignment_v06/related_references.json',relations)

def norm_with_map(t):
 mask=[False]*len(t)
 patterns=[r'!\[[^\]]*\]\([^\n)]*\)',r'<Cloud\b[^>]*?/\s*>',r'<GitHubIssuesHint\b[^>]*?/\s*>',r'</?(?:Note|Important)>']
 for pat in patterns:
  for m in re.finditer(pat,t):mask[m.start():m.end()]=[True]*(m.end()-m.start())
 out=[];positions=[];pending=None
 for i,c in enumerate(t):
  if mask[i]:continue
  if c.isspace():
   if out and pending is None:pending=i
   continue
  if pending is not None:out.append(' ');positions.append(pending);pending=None
  out.append(c);positions.append(i)
 return ''.join(out),positions
norms={sid:norm_with_map(r['text']) for sid,r in by.items()}
mapping=[]
def evidence(case,sid,body,heading,archived=None):
 needle,_=norm_with_map(body);hay,mp=norms[sid];idx=hay.find(needle)
 if idx<0:raise ValueError(('Unmapped evidence',case,sid,heading))
 if hay.find(needle,idx+1)>=0:raise ValueError(('Ambiguous evidence',case,sid,heading))
 a=mp[idx];b=mp[idx+len(needle)-1]+1;text=by[sid]['text'][a:b]
 e={'evidence_id':'evidence:'+sha(case+'|'+sid+'|'+heading+'|'+sha(body))[:20],'source_id':sid,'heading':heading,'start_char':a,'end_char':b,'text':text,'evidence_sha256':sha(text),'source_text_sha256':by[sid]['text_sha256'],'offset_basis':'retrieval_corpus.text','offset_unit':'python_unicode_code_points','end_exclusive':True,'version_scope':'Use source metadata and explicit version notes; a current snapshot does not certify historical behavior.'}
 if archived:mapping.append({'case_id':case,'evidence_id':e['evidence_id'],'source_id':sid,'archived_canonical_evidence':archived,'corpus_start_char':a,'corpus_end_char':b,'source_text_sha256':e['source_text_sha256'],'evidence_sha256':e['evidence_sha256'],'mapping_method':'unique_normalized_text_match_then_exact_corpus_slice'})
 return e
reviews=json.loads((ROOT/'config/evaluation_case_review_v06.json').read_text())
inputs=[];rubrics=[];env_changes=[];coverage=[]
def parse_environment(t):
 out={};ev={}
 patterns={'streamlit_version':r'(?im)^[ \t]*-?[ \t]*Streamlit(?:[ \t]+version)?[ \t]*[:,][ \t]*([^\r\n]*)','python_version':r'(?im)^[ \t]*-?[ \t]*Python(?:[ \t]+version[ \t]*:[ \t]*|[ \t]*:[ \t]*|[ \t]+(?=[0-9]))([^\r\n]*)','operating_system':r'(?im)^[ \t]*-?[ \t]*(?:Operating System|OS(?:[ \t]+version)?)[ \t]*:[ \t]*([^\r\n]*)','browser':r'(?im)^[ \t]*-?[ \t]*Browser(?:[ \t]+version)?[ \t]*:[ \t]*([^\r\n]*)'}
 for k,pat in patterns.items():
  matches=list(re.finditer(pat,t));found=matches[-1] if matches else None
  v=found[1].strip() if found else None
  if not v or v.lower() in ['na','n/a','none','_no response_'] or 'get it with' in v.lower():v=None
  if v and k in ['streamlit_version','python_version']:v=re.sub(r'^(?:(?:Streamlit|Python)[ \t,]*)?(?:version[ \t]+)?(?=\d)','',v,flags=re.I)
  if v is not None:
   out[k]=v;ev[k]={'start_char':found.start(1),'end_char':found.end(1),'text':found[1],'offset_basis':'agent_input.user_report','interpretation':'Explicit report string; no version normalization or inference.'}
  else:out[k]=None
 return out,ev
for split in ['development','test']:
 aa=readl(baseline/f'agent_inputs/{split}_inputs.jsonl');gg=readl(baseline/f'evaluator_only/{split}_rubrics.jsonl')
 for a,g in zip(aa,gg):
  cid=a['case_id'];assert cid==g['case_id'];review=reviews[cid]
  issue=json.loads((ROOT/f"evaluator_only/source_threads/{cid.split('-')[-1]}.json").read_text())['issue']
  a.update({'case_source_id':issue['source_id'],'case_source_access':'withheld_case_source_not_in_retrieval_corpus','created_at':issue['created_at'],'updated_at':issue['updated_at'],'source_url':issue['html_url'],'user_report_sha256':sha(a['user_report']),'schema_version':'6.0'})
  new,env_ev=parse_environment(a['user_report'])
  for key,value in new.items():
   old=a['reported_environment'].get(key)
   if old!=value:env_changes.append({'case_id':cid,'field':key,'before':old,'after':value,'evidence':env_ev.get(key)})
  a['reported_environment']=new;a['reported_environment_evidence']=env_ev;a['environment_note']='The verbatim report remains authoritative. Metadata stores explicit line values without inferring or normalizing a version.'
  g['acceptable_retrievable_evidence']=[evidence(cid,e['source_id'],e['text'],e['heading'],e.copy()) for e in g['acceptable_retrievable_evidence']]
  extra={
   'DEV-7044':('doc:linked:f62d2bc4926afc5516ff','A key lets you access the widget\'s value via','Additionally, if key is provided,','Data editor key is read-only'),
   'TEST-6966':('doc:linked:78cb80651cf78a0964fe','on_submit\n','Because st.chat_input is a trigger widget','Chat input callback and explicit arguments'),
   'TEST-4884':('doc:linked:e3c6cd07859b580cf16c','value\n','key\n','Date input initial value and bounds'),
   'DEV-10805':('doc:linked:b40cdea94393c153da48','When a user interacts with an input widget','To trigger an app rerun','Fragment reruns and run_every')}
  if cid in extra:
   sid,start,end,h=extra[cid];t=by[sid]['text'];s=t.index(start);e=t.index(end,s);g['acceptable_retrievable_evidence'].append(evidence(cid,sid,t[s:e],h))
  g['schema_version']='6.0';g['required_missing_information']=[]
  for key,question,impact,status in review['missing']:
   g['required_missing_information'].append({'information_id':cid+':'+key,'field':key,'question':question,'why_needed':impact,'decision_impact':impact,'status':status,'required_for':'discriminating_diagnosis_or_reproduction','ask_again_if_already_supplied':False})
  g['known_facts']=review['known']+[{'reported_environment':new}]
  g['verification_tasks']=[{'task_id':cid+':verify','instruction':g['acceptable_next_step'],'already_executed':False,'do_not_repeat_reported_completed_checks':True}]
  g['answerability_review']={'reviewer_kind':'ai_assistant','human_reviewed':False,'review_date':'2026-10-09','status':'supported_for_bounded_next_step','permitted_answer_scope':review['scope'],'confirmed_historical_fix_available':False,'runtime_verified':False,'unresolved_items':[x['information_id'] for x in g['required_missing_information']],'source_review_basis':'Exact current corpus evidence, unchanged issue report and code, and explicit version guards; withheld comments/releases cannot authorize an answer.'}
  g['claim_support']=[{'claim_id':cid+':documented_context','claim':'Use the cited documentation for the bounded concepts in the permitted answer scope.','evidence_ids':[e['evidence_id'] for e in g['acceptable_retrievable_evidence']],'basis':'retrievable_corpus','version_guard':'Respect each source snapshot; do not assert all old versions behave identically.'},{'claim_id':cid+':case_observation','claim':'The report and code describe the reproduction summarized in reference_reasoning; a reported symptom is not a certified cause or fix.','basis':'agent_input_report','input_ref':cid,'input_field':'user_report','input_sha256':a['user_report_sha256']}]
  g['forbidden_claims']=[g['unacceptable_conclusion'],'A closed label or a withheld oracle comment proves an exact fixed release.','The proposed code was executed or the historical bug was independently reproduced during dataset preparation.']
  g['notes']+=['Evidence offsets now address delivered corpus text. URLs and snapshot metadata are resolved via source_id.','Missing-information questions are discriminating requests, not a requirement to repeat facts already supplied.']
  c={'case_id':cid,'split':split,'case_category':g['case_category'],'permitted_answer_scope':review['scope'],'review_status':'bounded_next_step_supported','required_missing_information':g['required_missing_information'],'retrievable_evidence_ids':[e['evidence_id'] for e in g['acceptable_retrievable_evidence']],'retrievable_source_ids':list(dict.fromkeys(e['source_id'] for e in g['acceptable_retrievable_evidence'])),'unavailable_requirements_removed':False,'exact_fixed_release_required':False,'historical_resolution_verified':False,'reviewer_kind':'ai_assistant','human_reviewed':False}
  coverage.append(c);inputs.append(a);rubrics.append(g)
 writel(ROOT/f'agent_inputs/{split}_inputs.jsonl',[x for x in inputs if x['split']==split]);writel(ROOT/f'evaluator_only/{split}_rubrics.jsonl',[x for x in rubrics if x['split']==split])
writel(ROOT/'evaluator_only/evidence_coverage_v06.jsonl',coverage)
writej(ROOT/'audit/alignment_v06/evidence_span_mapping.json',mapping)
writej(ROOT/'audit/alignment_v06/environment_corrections.json',env_changes)

byinput={a['case_id']:a for a in inputs};bygold={g['case_id']:g for g in rubrics}
mt=json.loads((baseline/'evaluator_only/multi_turn_scenarios.json').read_text());hist_inputs=[];hist_gold=[]
hist_questions={
 'DEV-7044':'Before the editor is declared, do you assign companies_df into st.session_state["companies_df"]?',
 'DEV-5591':'Please supply a minimal runnable two-page example showing the state assignments and navigation.',
 'DEV-7305':'Use on_change=update rather than calling update(). After that change, what happens and does the text area have a key?',
 'TEST-9603':'How do you navigate from page 1 to page 2: a URL or Markdown link, or a supported in-app navigation command?',
 'TEST-8700':'Does the failing session start directly at /mypage, and can you separate that from the terminal-only occurrence?',
 'TEST-9841':'Compare Ctrl+Enter with clicking Submit in the same minimal form. What does each submit?'}
resolved={
 'DEV-7044':['prior_state_assignment'],
 'DEV-6166':[],
 'DEV-5591':['minimal_two_page_code'],
 'DEV-7305':[],
 'TEST-9603':['navigation_method'],
 'TEST-8700':['navigation_entry_path'],
 'TEST-9841':['mouse_submit_comparison']}
for s in mt:
 cid=s['case_id'];g=bygold[cid];follow=s['turns'][1]
 cleared=resolved.get(cid,[]);remaining=[x for x in g['required_missing_information'] if x['field'] not in cleared]
 s['schema_version']='6.0';s['steps']=[
  {'step_id':s['scenario_id']+':1','kind':'deliver_input','input_ref':cid,'release_condition':'session_start'},
  {'step_id':s['scenario_id']+':2','kind':'assistant_checkpoint','response_index':1,'rubric_ref':cid,'required_known_facts':g['known_facts'],'required_missing_information':g['required_missing_information'],'expected_next_step':g['acceptable_next_step'],'forbidden_claims':g['forbidden_claims']},
  {'step_id':s['scenario_id']+':3','kind':'deliver_followup','turn_ref':2,'release_condition':'assistant_response_1_recorded','origin':follow['origin']},
  {'step_id':s['scenario_id']+':4','kind':'assistant_checkpoint','response_index':2,'expected_next_step':s['expected_decision_after_followup'],'newly_supplied_fact':follow['body'],'resolved_information_ids':[cid+':'+f for f in cleared],'required_missing_information':remaining,'forbidden_claims':[g['unacceptable_conclusion'],'Ask again for information supplied in the released follow-up.','Claim a previously attempted failed or completed check has not been tried.'],'evidence_ids':[e['evidence_id'] for e in g['acceptable_retrievable_evidence']]}
 ]
 s['workflow_event_steps']=[{'step_id':s['scenario_id']+':event:'+str(i+1),'instruction':v,'origin':s['event_origin'],'execution_status':'requires_tracker_adapter_not_executed'} for i,v in enumerate(s['evaluator_events'])]
 s['delivery_adapter']='scripts/evaluation_session.py';s['conversation_delivery_verified_offline']=True
 if cid in hist_questions:
  a=byinput[cid];hid='HIST-'+cid
  variant={'case_id':hid,'base_case_id':cid,'split':s['split'],'scenario_id':s['scenario_id'],'schema_version':'6.0','title':'Resume a conversation: '+a['title'],'user_report':follow['body'],'user_report_sha256':sha(follow['body']),'initial_conversation':[{'role':'user','content':a['user_report'],'origin':'original_issue_report'},{'role':'assistant','content':hist_questions[cid],'origin':'designed_assistant_fixture_not_actual_model_output'}],'reported_environment':a['reported_environment'].copy(),'created_at':a['created_at'],'source_url':a['source_url'],'history_variant':True,'followup_origin':follow['origin'],'followup_source_id':follow.get('source_id'),'family_group_ref':cid,'input_origin':'explicitly_designed_history_variant_with_recorded_followup'}
  if cid=='TEST-9841':variant['reported_environment']['streamlit_version']='1.40.0'
  hist_inputs.append(variant)
  hist_gold.append({'case_id':hid,'base_case_id':cid,'split':s['split'],'history_is_designed_fixture':True,'expected_next_step':s['expected_decision_after_followup'],'required_missing_information':remaining,'resolved_information_ids':[cid+':'+f for f in cleared],'forbidden_claims':s['steps'][3]['forbidden_claims'],'acceptable_retrievable_evidence':g['acceptable_retrievable_evidence'],'not_independent_case_for_sample_size_reporting':True})
writej(ROOT/'evaluator_only/multi_turn_scenarios.json',mt)
writel(ROOT/'agent_inputs/history_variant_inputs.jsonl',hist_inputs);writel(ROOT/'evaluator_only/history_variant_rubrics.jsonl',hist_gold)

source_specs=[
 ('SRC-DEV-ISSUE','development','issue:7851','Does the multithreading issue establish an implemented API or a requested capability, and which related issues does it explicitly mention?','Treat the text as a historical feature request and identify 1906, 7290 and 7655 as mentions, not proven same-cause cases.','An implemented or certified thread-safe Session State API exists because the issue requests it.'),
 ('SRC-TEST-ISSUE','test','issue:6970','Does the custom-object report prove Session State cannot store custom objects? What in the supplied snippet could recreate the object?','Identify the unconditional Dataset() assignment on each rerun; preserve the reported symptom and do not certify the malformed source snippet as runnable.','Custom objects are universally unsupported, or the provided example was executed successfully.'),
 ('SRC-DEV-COMMENT','development','comment:998909519','Find the proposed method for clearing a text_input before it is rendered again, and state the status of that proposal.','Describe keyed state and an on_click callable that clears the key before rerun; label the comment a historical proposed explanation and support callback order with official evidence.','The community example was independently executed or universally proven.'),
 ('SRC-TEST-COMMENT','test','comment:884380879','What workaround was proposed for the range slider, and what limitation did the author report?','Describe initializing slider defaults and assigning keyed state on checkbox activation, and retain the duplicate-default warning as a reported limitation.','A warning-free or universally verified fix was established.'),
 ('SRC-DEV-RELEASE','development','release:0.84.0:section:2','Which release announced st.session_state and widget callbacks, and what can that release statement establish?','Identify 0.84.0 on 2021-07-01 as the feature announcement; do not extrapolate a specific bug fix or modern callback modes.','The announcement proves every later or earlier widget bug is fixed.'),
 ('SRC-TEST-RELEASE','test','release:0.83.0:section:4','Which release added clear_on_submit to forms and what behavior does that note describe?','Identify 0.83.0 on 2021-06-17 and resetting form widgets on submission; do not infer an unrelated historical fix.','Every state-persistence problem is fixed by clear_on_submit.'),
 ('SRC-DEV-LINK','development','linked:8242bee435fd5f31062c','What did the linked discussion propose for clearing a form after submit, and what scope does that evidence have?','Report clear_on_submit=True with st.form and st.form_submit_button as a historical community suggestion, and corroborate form semantics with official documentation.','The linked forum post proves an independently executed universal solution.'),
 ('SRC-TEST-LINK','test','linked:8242bee435fd5f31062c','Does the linked form-clearing suggestion address a requirement to keep search terms between submissions?','Explain that unconditional submit-clearing conflicts with retaining terms; distinguish the post\'s proposed behavior from the user\'s desired workflow.','Clear_on_submit is always appropriate even when the user needs to retain fields.')]
source_specs[-1]=('SRC-TEST-LINK','test','linked:dc9503cd41b0a1f3eed2','What callback approach does the linked select-all discussion propose, and can its source example be described as independently verified?','Describe keyed multiselect and checkbox callbacks that synchronize their values before the rest of the script. Preserve the historical proposed-example status; it was not executed during preparation.','The forum example was independently executed or certified across all Streamlit versions.')
src_inputs=[];src_gold=[]
for cid,split,sid,q,answer,no in source_specs:
 r=by[sid];t=r['text'];a=t.find('\n\n')+2 if r['source_type']!='release_section' else 0
 e=evidence(cid,sid,t[a:],'Selected source content')
 refs=[e]
 if sid=='comment:998909519':
  official=next(e for g in rubrics for e in g['acceptable_retrievable_evidence'] if e['heading']=='### Use Callbacks to update Session State')
  refs.append(evidence(cid,official['source_id'],official['text'],official['heading']))
 if sid=='linked:8242bee435fd5f31062c':
  d=by['doc:rendered:9f138256cdc71bb8'];pos=d['text'].find('clear_on_submit');a=max(0,pos-50);b=min(len(d['text']),pos+850);refs.append(evidence(cid,d['source_id'],d['text'][a:b],'Form API clearing parameter'))
 if sid=='linked:dc9503cd41b0a1f3eed2':
  official=next(e for g in rubrics for e in g['acceptable_retrievable_evidence'] if e['heading']=='### Use Callbacks to update Session State')
  refs.append(evidence(cid,official['source_id'],official['text'],official['heading']))
 src_inputs.append({'case_id':cid,'split':split,'task_kind':'supplementary_source_use_not_issue_resolution','user_report':q,'initial_conversation':[],'schema_version':'6.0'})
 src_gold.append({'case_id':cid,'split':split,'target_source_type':r['source_type'],'acceptable_next_step':answer,'unacceptable_conclusion':no,'acceptable_retrievable_evidence':refs,'evaluation_dimensions':['retrieval_relevance','evidence_role','version_scope','claim_support'],'runtime_solution_verification_required':False,'sample_size_reporting':'Separate supplemental source-use tasks; not part of the 15+15 issue cases.'})
writel(ROOT/'agent_inputs/source_usage_inputs.jsonl',src_inputs);writel(ROOT/'evaluator_only/source_usage_rubrics.jsonl',src_gold)
metrics={'version':'v06','source_records':len(records),'source_texts_changed':0,'source_counts':dict(collections.Counter(r['source_type'] for r in records)),'base_cases':len(inputs),'base_split_counts':dict(collections.Counter(a['split'] for a in inputs)),'mapped_inherited_evidence_spans':len(mapping),'base_evidence_spans':sum(len(g['acceptable_retrievable_evidence']) for g in rubrics),'additional_official_spans':4,'coverage_cases_reviewed':len(coverage),'bounded_next_step_supported':len(coverage),'confirmed_historical_fixes':0,'missing_information_items':sum(len(g['required_missing_information']) for g in rubrics),'environment_metadata_changes':len(env_changes),'case_dates_added':len(inputs),'metadata_source_id_duplicates_removed':len(records),'cleanup_audit_id_duplicates_removed':sum('metadata.text_cleanup.audit_source_id' in r['removed_runtime_fields'] for r in removed),'related_reference_edges':len(relations),'issues_with_related_issue_numbers':sum(bool(r['metadata'].get('related_issue_numbers')) for r in records if r['source_type']=='issue'),'multi_turn_scenarios':len(mt),'history_variants':len(hist_inputs),'history_split_counts':dict(collections.Counter(a['split'] for a in hist_inputs)),'supplementary_source_use_tasks':len(src_inputs),'supplementary_split_counts':dict(collections.Counter(a['split'] for a in src_inputs)),'chunks_built':False,'embeddings_built':False,'runtime_solution_tests_performed':False,'paid_api_calls':0,'human_reviewed':False}
writej(ROOT/'audit/alignment_v06/summary.json',metrics)
manifest=json.loads((baseline/'manifest.json').read_text());manifest['schema_version']='6.0';manifest['status']='corpus_evidence_alignment_and_evaluation_schema_reviewed_not_runtime_certified';manifest['alignment_v06']=metrics;manifest['evaluation_counts']={'base_development':15,'base_test':15,'history_development':3,'history_test':3,'source_use_development':4,'source_use_test':4};manifest['index_allowlist']=['retrieval_corpus/'+n+'.jsonl' for n in NAMES]
manifest['baseline_collection_counts']=manifest.pop('counts')
manifest['counts']={'source_records':len(records),'source_by_type':metrics['source_counts'],'base_development_cases':15,'base_test_cases':15,'history_variants':6,'supplementary_source_use_tasks':8,'multi_turn_scenarios':10,'base_retrievable_evidence_spans':64}
manifest['preparation_history']={'cleanup_v04':manifest.pop('text_preparation'),'source_volume_v05':manifest.pop('source_volume')}
manifest['source_volume']={'source_records':len(records),'text_characters':sum(len(r['text']) for r in records),'text_utf8_bytes':sum(len(r['text'].encode()) for r in records),'text_whitespace_words':sum(len(r['text'].split()) for r in records),'exact_model_token_count':None,'validated_base_evidence_spans':64,'comparison_baseline':'v05','source_texts_changed':0}
manifest['latest_preparation_version']='v06';manifest['preparation_date']='2026-10-09'
manifest['remaining_gates']=['Human review of the revised corpus-linked labels and supplemental fixtures.','Run actual chunking and map supplied chunks with exact source hashes.','Implement linked-neighbor retrieval explicitly.','Run the agent on the 15+15 cases and staged scenarios, including local tracker adapters.','Check historical-version applicability before declaring a fix.']
writej(ROOT/'manifest.json',manifest)
contract={'schema_version':'6.0','source_envelope':{'required':['source_id','source_type','text','metadata','text_sha256'],'source_id_location':'top level only','embedding_text':'text only; metadata and audit are not embedded'},'index_allowlist':manifest['index_allowlist'],'evidence_contract':{'source_id':'Must resolve to one delivered corpus record. URLs and snapshot provenance come from that record metadata.','start_char':'Inclusive Python Unicode code-point offset into delivered source text.','end_char':'Exclusive Python Unicode code-point offset into delivered source text.','invariant':'corpus[source_id].text[start_char:end_char] == evidence.text','source_text_sha256':'Detects changes to the source before mapping or scoring.','evidence_sha256':'Detects changes to the exact selected span.','archived_offsets':'Retained only in audit/alignment_v06/evidence_span_mapping.json, with the original source representation.'},'related_reference_contract':{'relation_type':'Explicit mention only; never infer same cause, fix, or version applicability.','unresolved_target':'Keep number and status; do not invent an index source ID for held-out or absent records.'},'evaluation_contract':{'base_cases':'15 development + 15 test; original user reports unchanged','missing_information':'Evaluator-only structured discriminating questions; empty list is valid for bounded answerable cases.','history_variants':'Six explicit fixtures, same split and family as their base case; do not double-count as independent cases.','multi_turn':'Ten staged scenarios; follow-up is released only after response 1 is recorded.','source_usage':'Eight separate evidence-role and retrieval tasks; not held-out issue-resolution tests.','workflow_events':'Local tracker integration specifications still require an adapter and actual system execution.'},'chunk_contract':{'start_char':'Source text offset, inclusive','end_char':'Source text offset, exclusive','source_text_sha256':'Must equal parent source hash','mapping_helper':'scripts/map_evidence_to_chunks.py maps supplied chunks and reports interval coverage; no real chunks have been built.'}}
writej(ROOT/'Embedding_Structure_v06.json',contract)
readme='''# Streamlit Session State v06

540 selected source records. Every v05 text and text hash is unchanged. Only runtime metadata and evaluator schemas were revised. No chunking, embedding, network collection, paid API call or execution of source examples was performed.

## Index input
Use only the five explicit files in retrieval_corpus. scripts/load_retrieval_sources.py loads them. Embed text only. Do not recursively load audit, evaluator_only, config, reports or agent_inputs. Top-level source_id is authoritative; duplicated metadata IDs were removed.

## Labels and offsets
All inherited 60 evidence spans were remapped from archived canonical document bodies to exact delivered corpus text; four additional official witnesses were selected from existing sources. Each evidence span has source_id, start_char, end_char, exact text, evidence hash and parent text hash. Offsets are Python Unicode code points, with an exclusive end. Source URLs and snapshot versions are obtained from corpus metadata. Legacy evidence and offsets remain in audit/alignment_v06/evidence_span_mapping.json.

After actual chunking, run scripts/map_evidence_to_chunks.py with a supplied chunks.jsonl. It checks each chunk against the parent text and hash, reports overlap and complete/partial interval coverage, and fails on stale or mismatched data. No chunk IDs are fabricated in this source-only delivery.

## Evaluation
agent_inputs/development_inputs.jsonl and test_inputs.jsonl retain 15 cases each and all original report text. They now include dates and exact explicit environment lines. The corresponding evaluator rubrics contain required_missing_information, known_facts, verification_tasks, permitted answer scope and claim support. Historical facts must not be requested again as though missing.

Ten multi_turn_scenarios have explicit steps and response checkpoints. scripts/evaluation_session.py provides staged conversation delivery and never returns gold, oracle fields, or unreleased follow-ups in its agent view. Tracker approval/restart/timeout specifications still need actual integration; no agent or tracker behavior score is claimed.

Six history_variant_inputs are separate fixtures, three per split, with an explicitly designed assistant question and a recorded or designed user follow-up. Their evaluator-only rubric asks the system to resume with updated facts. They are derivatives, not six independent new incidents.

Eight source_usage_inputs are separate tasks for historical reports, comment proposals, release statements and linked discussions. Their rubrics require correct evidence roles and version scope. They do not import the answers to held-out issues into the index.

## Verification and rebuilding
Run python3 scripts/validate_dataset.py for offline differential and schema checks. Run python3 scripts/build_dataset.py to regenerate v06 sources and evaluator data from the immutable v05 baseline and recorded case review config; then validate. Run python3 scripts/finalize_v06.py after any dataset edits to validate and refresh the package SHA256SUMS inventory. scripts/test_v06_contracts.py checks staged disclosure, Unicode offsets, cross-boundary chunk mapping and stale-hash rejection using fixtures only. Suffixed older builders/validators are archived tools, not active v06 entrypoints.

## Review limits
The 30 cases were reviewed for a bounded response, discriminating question or escalation summary. This does not certify an exact historical fix, every technical claim, or end-to-end RAG performance. Cases that require more observation retain that uncertainty. Human signoff remains pending. Older CHANGE_REPORT_v04_FA.md and CHANGE_REPORT_v05_FA.md are historical snapshots; use ALIGNMENT_REPORT_v06_FA.md and the current manifest, README and validation as the v06 contract.
'''
(ROOT/'README.md').write_text(readme)
(ROOT/'REVIEW_REPORT.md').write_text('# Corpus and evaluation alignment v06\n\n'+readme.split('## Index input')[0].split('\n\n',1)[1]+'\n\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n\nThe exact 60 inherited spans are mapped to corpus text and four existing official API spans supplement the cases for data-editor keys, date bounds, chat callbacks and scheduled fragments. All 30 cases permit a bounded next step; none requires an unavailable exact fixed release. Source-use tasks and history fixtures are reported separately from the 30 base cases. Runtime solution verification and tracker integration remain unexecuted.\n')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
