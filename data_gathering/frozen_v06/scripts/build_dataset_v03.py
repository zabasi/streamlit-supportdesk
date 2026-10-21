#!/usr/bin/env python3
"""Offline synthesis of two supplied snapshots. No API or model calls.

Rules propose relevance and review dispositions; they do not certify truth.
Only retrieval_corpus is an index input. Evaluation remains a draft.
"""
import hashlib,json,re,shutil
from collections import Counter,defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'audit/input_data'
COMMIT='695e6ce2fb1b23afc6df66b40420ab0013ad72fd'
PILOTS={5813,6079,5498,9100}
EMAIL=re.compile(r'(?<![\w/])\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
STATE=re.compile(r'session_state|session[ -]state|sessionstate|widget[ -]state',re.I)
FENCE=re.compile(r'(```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~)',re.S)
MENTION=re.compile(r'https://github\.com/streamlit/streamlit/(?:issues|pull)/(\d+)|(?<![\w/#])#(\d{3,6})\b|\bgh-(\d+)\b')
TOPICS={'multipage':r'multipage|multi-page|st\.navigation|st\.page|switch.{0,20}page',
 'callbacks':r'callback|on_change|on_click|on_submit',
 'forms':r'st\.form|form_submit|inside.{0,12}form',
 'reset_or_persistence':r'reset|persist|disappear|los[et]|clear.{0,15}state',
 'initialization_or_update':r'initializ|keyerror|attributeerror|setdefault|updat',
 'widget_identity':r'duplicatewidget|duplicateelement|widget.{0,12}key|widget.{0,12}identit',
 'serialization':r'serializ|pickle|class identity',
 'session_lifecycle':r'websocket|reconnect|browser.{0,12}reload|refresh|disconnect',
 'testing':r'AppTest|testing'}

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in d),encoding='utf-8')
def outside_code(s):return FENCE.sub('',s)
def refs(s):return {int(next(v for v in match if v)) for match in MENTION.findall(s)}
def transform_outside(s,fn):return ''.join(x if i%2 else fn(x) for i,x in enumerate(FENCE.split(s)))

def clean(s):
 s=(s or '').replace('\r\n','\n').replace('\r','\n')
 def part(t):
  t=re.sub(r'(?im)^\s*[-*]\s*\[[ xX]\]\s*I (?:have searched.*existing issues.*|added a very descriptive title.*|have provided sufficient information.*)\s*$','',t)
  t=re.sub(r'(?im)^.*(?:Community voting on feature requests|Your vote helps us identify|please use the.*thumbs up|To help Streamlit prioritize this feature).*$','',t)
  return t
 return re.sub(r'\n{4,}','\n\n\n',transform_outside(EMAIL.sub('[REDACTED_EMAIL]',s),part)).strip()

def media(s):
 links=[]
 def part(t):
  def image(m):links.append({'kind':'image','reference':m[0]});return ''
  def video(m):links.append({'kind':'video','reference':m[0]});return ''
  t=re.sub(r'!\[[^\]]*\]\([^\n]+?\)',image,t)
  t=re.sub(r'<(?:img|Image)\b[^>]*?/?>',image,t,flags=re.I)
  t=re.sub(r'<(?:video|YouTube)\b[^>]*(?:/>|>.*?</video>)',video,t,flags=re.I|re.S)
  t=re.sub(r'https?://(?:www\.)?(?:youtu\.be/|youtube\.com/(?:watch\?|shorts/|embed/)|vimeo\.com/)[^\s<>)]*',video,t,flags=re.I)
  # Assets without an extension may be images or videos: do not guess.
  def attachment(m):links.append({'kind':'attachment_type_unknown','reference':m[0]});return ''
  t=re.sub(r'https://github\.com/user-attachments/(?:assets|files)/[^\s)<>]+',attachment,t)
  t=re.sub(r'https://github\.com/[^/\s]+/[^/\s]+/assets/[^\s)<>]+',attachment,t)
  t=re.sub(r'https?://[^\s<>]+\.(?:png|jpe?g|gif|webp|mp4|webm|mov)(?:\?[^\s<>]*)?',attachment,t,flags=re.I)
  t=re.sub(r'\[[^\]]*\]\(\s*\)','',t)
  return t
 text=transform_outside(s,part)
 return text.strip(),links

def document_text(s):
 s=media(clean(s))[0]
 def part(t):
  t=re.sub(r'(?im)^.*(?:tutorial video by|basics tutorial video|Check out this.*video).*\n?','',t)
  t=re.sub(r'<(?:/?(?:Note|Warning|Important|Tip|Cloud|NotCloud)|GitHubIssuesHint)\b[^>]*>','',t)
  t=re.sub(r'(?m)^import .* from [\'\"](?:@|\./).*?;?\s*$','',t)
  return t
 return transform_outside(s,part).strip()

def issue_text(s):
 def part(t):
  # Drop unfilled template sections only; meaningful debug/code stays intact.
  t=re.sub(r'(?ms)^#{2,3} (?:Checklist|Reproducible Code Example|Steps To Reproduce|Expected Behavior|Current Behavior|Additional Information|Additional Context|Are you willing to submit a PR\?)\s*\n(?:(?!^#{2,3} ).)*?(?=^#{2,3} |\Z)',lambda m:'' if not re.sub(r'(?im)^\s*(?:_No response_|[-*] \[[ xX]\].*|#+ .*|\s*)$','',m[0]).strip() else m[0],t)
  t=re.sub(r'(?im)^\s*[-*] \[[ xX]\] (?:I added a descriptive title and summary to this issue\.|Yes, I am willing to submit a PR!).*\n?','',t)
  return re.sub(r'\n{3,}','\n\n',t)
 return transform_outside(s,part).strip()

def env(body,product):
 # Preserve only explicit statements, including common templates used by both inputs.
 pattern=r'(?im)^\s*(?:[-*]\s*)?(?:\*\*)?'+product+r'(?:\s+version)?(?:\*\*)?\s*:\s*([^\n]+)'
 m=re.search(pattern,body)
 if not m:return None
 value=m[1].strip().strip('`*_ ')
 if not value or value.lower() in ['n/a','na','unknown','none','_no response_','no response']:return None
 return {'value':value,'evidence_text':m[0].strip(),'start_char':m.start(),'end_char':m.end(),'method':'explicit_environment_line_regex'}

def enrich_issue(x):
 body=clean(x.get('body'));x=dict(x);x['body']=body;x['body_sha256']=digest(body)
 x['source_id']=f'issue:{x["number"]}';x['source_type']='issue'
 x['repository']='streamlit/streamlit'
 x['streamlit_version']=env(body,'Streamlit');x['python_version']=env(body,'Python')
 x['operating_system']=env(body,r'(?:Operating System|OS)');x['browser']=env(body,r'Browser')
 x['topic_tags_heuristic']=[k for k,p in TOPICS.items() if re.search(p,x['title']+'\n'+body,re.I)]
 x['scope_match_basis']='title_or_label' if STATE.search(x['title']) or 'feature:st.session_state' in x.get('labels',[]) else 'body_narrative_requires_review'
 x['in_scope_heuristic']=('2021-07-01'<=x['created_at'][:10]<='2026-10-06' and bool(STATE.search(x['title']+'\n'+outside_code(body)) or 'feature:st.session_state' in x.get('labels',[])))
 x['verification_status']='reported_not_independently_verified';return x

def graph(issues,comments):
 ids={x['number'] for x in issues};by={x['number']:x for x in issues};edges=[];parent={n:n for n in ids}
 def root(n):
  while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
  return n
 def union(a,b):
  a,b=root(a),root(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 # Explicit duplicate-language edges are conservative blockers, not certified families.
 for x in issues+comments:
  n=x.get('number',x.get('issue_number'));body=outside_code(x['body'])
  for m in re.finditer(r'(?:duplicate\s+of|duplicated\s+by|same\s+(?:issue|problem)\s+as|tracked\s+(?:in|by)|superseded\s+by)[^\n]{0,150}',body,re.I):
   for k in refs(m[0]):
    if k in ids and k!=n:edges.append({'issue_a':n,'issue_b':k,'basis':'explicit_related_or_duplicate_language','evidence_source_id':x['source_id'],'evidence_text':m[0],'human_verified':False});union(n,k)
 # A supplied concrete cross-split duplicate candidate is blocked conservatively.
 if 15892 in ids and 15893 in ids:
  edges.append({'issue_a':15892,'issue_b':15893,'basis':'same_version_tabs_regression_and_duplicate_state_reason','evidence_source_id':'issue:15893','evidence_text':'1.59.0 tab collapse after widget interactions; issue 15893 has state_reason duplicate','human_verified':False});union(15892,15893)
 scoped=[x for x in issues if x['in_scope_heuristic']]
 for i,a in enumerate(scoped):
  ta=re.sub(r'[^a-z0-9]+',' ',a['title'].lower()).strip()
  for b in scoped[i+1:]:
   tb=re.sub(r'[^a-z0-9]+',' ',b['title'].lower()).strip()
   ratio=SequenceMatcher(None,ta,tb).ratio()
   if min(len(ta),len(tb))>35 and ratio>=.91:
    edges.append({'issue_a':a['number'],'issue_b':b['number'],'basis':'high_title_similarity_conservative_block','title_similarity':round(ratio,4),'human_verified':False});union(a['number'],b['number'])
 groups=defaultdict(set)
 for n in ids:groups[root(n)].add(n)
 family={n:'family:'+digest(','.join(map(str,sorted(g))))[:16] for g in groups.values() for n in g}
 return edges,family,{n:groups[root(n)] for n in ids}

def envelope(x,text,kind,extra=None):
 fields={
 'issue':['source_id','id','number','html_url','title','repository','created_at','updated_at','closed_at','state','state_reason','labels','author_association','streamlit_version','python_version','operating_system','browser','body_sha256','family_id','verification_status'],
 'comment':['source_id','id','parent_source_id','issue_number','html_url','created_at','updated_at','author_association','body_sha256','verification_status'],
 'documentation':['source_id','title','html_url','source_url','repository','commit','content_path','representation','retrieved_at','content_sha256','raw_source_sha256','retrieval_body_sha256'],
 'release_section':['source_id','parent_source_id','version','release_date','section_heading','html_url','source_url','commit','retrieved_at','body_sha256']}
 meta={k:x.get(k) for k in fields[kind]};meta['source_type']=kind
 if extra:meta.update(extra)
 return {'source_id':x['source_id'],'source_type':kind,'text':text,'metadata':meta,'text_sha256':digest(text)}

def run():
 primary=IN/'primary';col=IN/'colleague';previous=IN/'previous_evaluation'
 oi=rows(primary/'issues.jsonl');oc=rows(primary/'comments.jsonl');od=rows(primary/'documents.jsonl');releases=rows(primary/'release_notes_all.jsonl')
 ci_bundle=load(col/'issues.json');cc_bundle=load(col/'comments.json');cd_bundle=load(col/'documents.json')
 ci=[];cc=[]
 for z in ci_bundle['issues']:
  x={k:z.get(k) for k in ['title','body','html_url','created_at','updated_at','closed_at','state','state_reason','labels','author_association']}
  x.update({'id':z['github_id'],'number':z['issue_number'],'comment_count':z['comments_count'],'metadata_retrieved_at':None,'bundle_collected_at':ci_bundle['collected_at'],'provenance_bundle':'colleague','comments_retrieval_method':'github_rest_comments_reported_by_colleague'})
  ci.append(x)
 for z in cc_bundle['comments']:
  x={k:z.get(k) for k in ['body','html_url','created_at','updated_at','author_association','issue_number']}
  x.update({'id':z['comment_id'],'source_id':f'comment:{z["comment_id"]}','parent_source_id':f'issue:{z["issue_number"]}','source_type':'comment','retrieved_at':None,'bundle_collected_at':cc_bundle['collected_at'],'retrieval_method':'github_rest_comments_reported_by_colleague','verification_status':'unreviewed'})
  cc.append(x)
 issue_variants=defaultdict(list);comment_variants=defaultdict(list)
 for label,items in [('primary',oi),('colleague',ci)]:
  for x in items:x=dict(x);x['provenance_bundle']=label;issue_variants[x['number']].append(x)
 for label,items in [('primary',oc),('colleague',cc)]:
  for x in items:x=dict(x);x['provenance_bundle']=label;comment_variants[x['id']].append(x)
 issues=[];comments=[];merge_audit=[]
 for n,variants in sorted(issue_variants.items()):
  # Pick the latest edited issue version; prefer primary on a tie for richer provenance.
  variants.sort(key=lambda x:(x.get('updated_at') or '',x['provenance_bundle']=='primary'),reverse=True)
  x=enrich_issue(variants[0]);x['available_snapshot_bundles']=sorted({v['provenance_bundle'] for v in variants});issues.append(x)
  merge_audit.append({'source_id':x['source_id'],'selected_bundle':x['provenance_bundle'],'selection_rule':'latest updated_at; primary preferred on tie','variants':[{'bundle':v['provenance_bundle'],'updated_at':v.get('updated_at'),'original_input_body_sha256':digest(v.get('body') or '')} for v in variants]})
 for cid,variants in sorted(comment_variants.items()):
  variants.sort(key=lambda x:(x.get('updated_at') or '',x['provenance_bundle']=='primary'),reverse=True)
  x=dict(variants[0]);x['body']=clean(x.get('body'));x['body_sha256']=digest(x['body']);x['verification_status']='unreviewed';x['available_snapshot_bundles']=sorted({v['provenance_bundle'] for v in variants});comments.append(x)
 counts=Counter(x['issue_number'] for x in comments)
 for x in issues:x['collected_comment_count']=counts[x['number']];x['comments_complete_count_check']=counts[x['number']]==x['comment_count']
 edges,families,groups=graph(issues,comments)
 for x in issues:x['family_id']=families[x['number']]
 old_cases={s:load(col/f'{s}.json') for s in ['dev','test']}
 old_roots={int(x['family_id'].split('-')[-1]) for d in old_cases.values() for x in d}
 inherited=load(previous/'audit/exclusion_manifest.json')
 new_roots=set(inherited['selected_evaluation_issues'])
 inherited_blocked=set(inherited['quarantined_collected_issues'])
 blocked=set().union(inherited_blocked,*(groups[n] for n in old_roots|new_roots|PILOTS|inherited_blocked if n in groups))
 blocked_references=blocked|set(inherited['blocked_reference_numbers_including_external_issue_or_PR_targets'])
 reasons={n:[] for n in blocked}
 for n in blocked:
  if n in old_roots:reasons[n].append('colleague_evaluation_root_not_adopted')
  if n in new_roots:reasons[n].append('retained_fixed_evaluation_root')
  if n in inherited_blocked:reasons[n].append('inherited_conservative_quarantine')
  if n in PILOTS:reasons[n].append('previously_inspected_pilot')
  if not reasons[n]:reasons[n].append('additional_conservative_related_family_block')
 scope_reviews={x['number']:x for x in load(ROOT/'config/scope_reviews.json')}
 exclusions=[];ri=[];pending=[];discard=[]
 for x in issues:
  n=x['number'];reason=None
  if not x['in_scope_heuristic']:reason='outside_focused_scope_or_native_api_date_range'
  elif n in scope_reviews and scope_reviews[n]['decision']=='exclude':reason='scope_review_excluded_broader_topic'
  elif n in blocked:reason='heldout_or_pilot_family'
  elif not x['comments_complete_count_check']:reason='incomplete_comment_thread'
  elif refs(x['title']+'\n'+x['body'])&blocked_references:reason='mentions_heldout_or_pilot_issue'
  text,assets=media(x['body'])
  selected_scope=scope_reviews.get(n,{})
  if selected_scope.get('excerpt_before'):
   original_text=text
   text=text[:text.index(selected_scope['excerpt_before'])]
   if selected_scope.get('append_from'):text+='\n\n'+original_text[original_text.index(selected_scope['append_from']):]
  text=issue_text(text)
  if not reason and assets and len(re.findall(r'\b\w+\b',text))<25:reason='media_dependent_short_issue_needs_review'
  if reason:exclusions.append({'source_id':x['source_id'],'reason':reason,'references_to_blocked':sorted(refs(x['body'])&blocked_references)});continue
  enhancement=any('enhancement' in z.lower() for z in x.get('labels',[]))
  notice='Historical feature request: proposed APIs are not evidence of implemented behavior.' if enhancement else 'Historical problem report: the example may reproduce a failure; it is not a verified solution or cause.'
  ri.append(envelope(x,'Title: '+x['title']+'\nEvidence role: '+notice+'\n\n'+text,'issue',{'scope_match_basis':x['scope_match_basis'],'media_present':bool(assets),'evidence_role':'reported_problem_not_verified_solution','source_intent_hint':'feature_request' if enhancement else 'reported_behavior','usage_constraint':notice,'reviewer_kind':'ai_assistant','scope_review':scope_reviews.get(n),'code_usage':'reported_reproduction_not_executed','runtime_verified':False,'human_reviewed':False}))
 allowed_issue_ids={r['metadata']['number'] for r in ri}
 technical=re.compile(r'```|Traceback|\bst\.|session.?state|widget|callback|rerun|reproduc|workaround|regression|exception|error|fix|version|commit|pull request|python',re.I)
 for x in comments:
  n=x['issue_number'];text,assets=media(x['body']);reason=None
  if n not in allowed_issue_ids:reason='parent_issue_not_in_retrieval'
  elif refs(text)&blocked_references:reason='mentions_heldout_or_pilot_issue'
  # Content decisions are recorded explicitly. Length and keywords are flags,
  # never rejection rules: short confirmations/corrections may be valuable.
  if reason:discard.append({'source_id':x['source_id'],'issue_number':n,'reason':reason,'media_present':bool(assets)});continue
  flags=[]
  if assets:flags.append('media_present_text_independence_unreviewed')
  if len(text)<180:flags.append('short_comment_may_require_conversation_context')
  if re.search(r'\b(?:wrong|incorrect|doesn.t work|did not work|not fixed|does not fix|still fails|didn.t help|disagree|correction)\b',outside_code(text),re.I):flags.append('possible_correction_or_failed_attempt')
  if re.search(r'\b(?:maybe|perhaps|I think|might|probably)\b',outside_code(text),re.I):flags.append('explicit_uncertainty')
  flags.append('truth_not_verified')
  pending.append(envelope(x,text,'comment',{'review_flags':flags,'media_present':bool(assets),'embedding_allowed':False,'allowed_use_after_review':None,'evidence_role':'unreviewed_community_statement'}))
 # Documentation deduplication uses source path+commit, not unstable upstream IDs.
 prepared_docs=rows(previous/'retrieval_corpus/documents.jsonl')
 doc_by={x['source_id']:dict(x) for x in od};aliases=[]
 for x in prepared_docs:doc_by[x['source_id']]=dict(x)
 for z in cd_bundle['documents']:
  path=z['metadata']['path'];sid='doc:pinned:'+path
  aliases.append({'upstream_id':z['id'],'upstream_corpus_id':'document-'+z['id'],'canonical_source_id':sid})
  if sid in doc_by:continue
  text=re.sub(r'^---\s*\n.*?\n---\s*\n','',z['text'],count=1,flags=re.S).strip()
  slug=re.search(r'(?m)^slug:\s*([^\n]+)',z['text'])
  doc_by[sid]={'source_id':sid,'source_type':'documentation','title':z['title'],'body':text,'html_url':'https://docs.streamlit.io'+slug[1].strip() if slug else None,'source_url':z['url'],'repository':'streamlit/docs','commit':z['metadata']['source_version'],'content_path':path,'representation':'source_markdown','retrieved_at':None,'provenance_bundle':'colleague','content_sha256':digest(text),'provided_source_text_sha256':digest(z['text']),'raw_source_sha256':None,'requires_rendered_api_content':bool(re.search(r'<Auto(?:function|class|data)',text))}
 docs=list(doc_by.values());preferred_pages={x['html_url']:x['source_id'] for x in prepared_docs}
 document_reviews={x['source_id']:x for x in load(ROOT/'config/document_reviews.json')}
 rd=[];doc_exclusions=[]
 for x in docs:
  reason=None;path=x.get('content_path','')
  if x.get('html_url') in preferred_pages and preferred_pages[x['html_url']]!=x['source_id']:reason='preserve_previously_selected_canonical_representation'
  elif path.endswith('_index.md') or '/command-line/' in path:reason='navigation_index_or_outside_scope'
  elif x.get('requires_rendered_api_content'):reason='autogenerated_api_placeholder_requires_rendered_snapshot'
  elif refs(x['body'])&blocked_references:reason='documentation_mentions_blocked_issue'
  elif x['source_id'] not in document_reviews or document_reviews[x['source_id']]['decision']!='approve':reason='no_approved_relevant_document_sections'
  if reason:doc_exclusions.append({'source_id':x['source_id'],'reason':reason});continue
  review=document_reviews[x['source_id']];selected=[]
  for span in review['approved_spans']:
   assert x['body'][span['start_char']:span['end_char']]==span['source_slice']
   selected.append(document_text(span['source_slice']))
  text='\n\n'.join(selected);assets=media(x['body'])[1]
  rd.append(envelope(x,x['title']+'\n\n'+text,'documentation',{'media_present':bool(assets),'evidence_role':'official_documentation','version_applicability':'must_check_against_user_version','source_body_sha256':digest(x['body']),'reviewer_kind':'ai_assistant','section_review_rationale':review['rationale'],'selected_source_spans':[{k:s[k] for k in ['start_char','end_char','heading','sha256']} for s in review['approved_spans']]}))
 sections=rows(previous/'retrieval_corpus/release_sections.jsonl');rr=[];note_exclusions=[]
 release_reviews={x['source_id']:x for x in load(ROOT/'config/release_reviews.json')}
 release_by_version={x['version']:x for x in releases}
 for x in sections:
  if refs(x['body'])&blocked_references:note_exclusions.append({'source_id':x['source_id'],'reason':'release_section_mentions_blocked_issue','references_to_blocked':sorted(refs(x['body'])&blocked_references)});continue
  review=release_reviews.get(x['source_id'],{})
  if review.get('decision')!='approve':note_exclusions.append({'source_id':x['source_id'],'reason':'LLM_topic_review_excluded','rationale':review.get('rationale')});continue
  body=review['approved_text'];parent=release_by_version[x['version']]
  assert parent['body'][review['start_char']:review['end_char']]==review['source_slice']
  if refs(body)&blocked_references:note_exclusions.append({'source_id':x['source_id'],'reason':'expanded_release_context_mentions_blocked_issue'});continue
  text,assets=media(clean(body));rr.append(envelope(x,f'Streamlit {x["version"]} ({x["release_date"]})\n{x.get("section_heading","Changes")}\n\n{text}','release_section',{'evidence_role':'official_version_change','relevance_basis':review['rationale'],'reviewer_kind':'ai_assistant','selected_source_span':{k:review[k] for k in ['parent_source_id','start_char','end_char','sha256']},'nested_list_context_preserved':True}))
 for name,data in [('issues',ri),('documents',rd),('release_sections',rr),('comments',[])]:lines(ROOT/f'retrieval_corpus/{name}.jsonl',data)
 lines(ROOT/'review_pending/comments.jsonl',pending)
 # Review provenance is explicit; an agent review must never be called human review.
 approval=ROOT/'config/comment_reviews.json'
 if not approval.exists():put(approval,[])
 by_comment={x['source_id']:x for x in pending};approved=[]
 for review in load(approval):
  sid=review['source_id'];assert sid in by_comment
  assert review['decision'] in ['approve','exclude','pending']
  if review['decision']!='approve':continue
  assert review.get('reviewer') and review.get('rationale') and review.get('evidence_source_ids') and review.get('approved_text')
  assert review.get('version_applicability'), 'Record the version applicability review, including unknowns.'
  if by_comment[sid]['metadata'].get('media_present'):assert review.get('media_independence')=='selected_excerpt_is_self_contained_without_media', 'Media-containing comments require explicit excerpt independence review.'
  assert review.get('allowed_use') in ['reported_observation','diagnostic_context','corroborated_claim']
  for e in review['evidence_source_ids']:assert e in {x['source_id'] for x in ri+rd+rr}
  assert review['approved_text'] in by_comment[sid]['text'], 'Approved text must be an exact selected source excerpt.'
  if review['allowed_use']=='corroborated_claim':assert set(review['evidence_source_ids'])&{x['source_id'] for x in rd+rr}, 'Corroborated claims require official-source evidence.'
  n=by_comment[sid]['metadata']['issue_number'];title=next(x['title'] for x in issues if x['number']==n)
  contexts=[]
  for context in review.get('context_excerpts',[]):
   other=by_comment[context['source_id']]
   assert other['metadata']['issue_number']==n
   assert context['text'] in other['text']
   contexts.append('Thread context ('+context['source_id']+'): '+context['text'])
  role=review['allowed_use'];notice={'reported_observation':'Historical reported observation; not independently reproduced.','diagnostic_context':'Historical diagnostic discussion or proposed workaround; not a verified fix.','corroborated_claim':'Only the selected claim is supported by the cited official snapshot; no runtime test was performed.'}[role]
  r=dict(by_comment[sid]);r['text']='Issue context: '+title+'\nEvidence role: '+notice+'\n\n'+review['approved_text']
  if contexts:r['text']+='\n\n'+'\n\n'.join(contexts)
  if review.get('limitation'):r['text']+='\n\nReview limitation: '+review['limitation']
  r['text_sha256']=digest(r['text']);r['metadata']=dict(r['metadata']);r['metadata'].update({'embedding_allowed':True,'verification_status':'agent_reviewed_for_retrieval_not_runtime_verified','evidence_role':'agent_reviewed_community_excerpt','allowed_use_after_review':role,'allowed_use':role,'supporting_source_ids':review['evidence_source_ids'],'reviewer':review['reviewer'],'reviewer_kind':'ai_assistant','review_rationale':review['rationale'],'version_applicability_review':review['version_applicability'],'media_independence_review':review.get('media_independence'),'context_source_ids':[c['source_id'] for c in review.get('context_excerpts',[])],'runtime_verified':False,'human_reviewed':False,'code_usage':'historical_example_not_executed'});approved.append(r)
 lines(ROOT/'retrieval_corpus/comments.jsonl',approved)
 review_by={x['source_id']:x for x in load(approval)}
 remaining=[x for x in pending if review_by.get(x['source_id'],{}).get('decision','pending')=='pending']
 lines(ROOT/'review_pending/comments.jsonl',remaining)
 lines(ROOT/'audit/comment_review_outcomes.jsonl',load(approval))
 lines(ROOT/'audit/source_archive/issues.jsonl',issues);lines(ROOT/'audit/source_archive/comments.jsonl',comments);lines(ROOT/'audit/source_archive/documents.jsonl',docs);lines(ROOT/'audit/source_archive/release_versions.jsonl',releases);lines(ROOT/'audit/source_archive/release_sections.jsonl',sections)
 lines(ROOT/'audit/merge_decisions.jsonl',merge_audit);lines(ROOT/'audit/document_id_aliases.jsonl',aliases);lines(ROOT/'audit/family_edges.jsonl',edges)
 put(ROOT/'audit/heldout_issue_numbers.json',{'blocked_reference_numbers_including_external_PRs':sorted(blocked_references),'issue_numbers':sorted(blocked),'reasons':reasons,'family_resolution':'conservative_rules_not_final_human_verified_families'})
 lines(ROOT/'audit/issue_exclusions.jsonl',exclusions);lines(ROOT/'audit/comment_exclusions.jsonl',discard);lines(ROOT/'audit/document_exclusions.jsonl',doc_exclusions);lines(ROOT/'audit/release_exclusions.jsonl',note_exclusions)
 lines(ROOT/'audit/media_manifest.jsonl',[{'source_id':x['source_id'],'references':media(x['body'])[1],'downloaded':False,'interpreted':False} for x in issues+comments+docs if media(x['body'])[1]])
 for folder in ['agent_inputs','evaluator_only']:
  shutil.copytree(previous/folder,ROOT/folder,dirs_exist_ok=True)
 put(ROOT/'evaluator_only/review_status.json',{'selection':'preserved_from_previous_fixed_30_case_package','rubric_status':'team_human_signoff_pending','scenario_status':'specifications_not_executed','followup_visibility':'evaluator releases one due message; future turns must not be preloaded into agent input'})
 lines(ROOT/'evaluator_only/additional_quarantined_issues.jsonl',[x for x in issues if x['number'] in blocked and x['number'] not in new_roots])
 lines(ROOT/'evaluator_only/additional_quarantined_comments.jsonl',[x for x in comments if x['issue_number'] in blocked and x['issue_number'] not in new_roots])
 put(ROOT/'audit/inherited_exclusion_manifest.json',inherited)
 # Old case files are evidence of the review, not validated gold or agent inputs.
 for split,cases in old_cases.items():put(ROOT/f'audit/colleague_evaluation/{split}_original.json',cases)
 corpus=load(col/'corpus.json');record_ids={x['id'] for x in corpus['records']}
 old_evidence=Counter(e for d in old_cases.values() for c in d for e in c['turns'][0]['gold_evidence_ids'] if e.startswith('release-note-'))
 overlap=set(x['number'] for x in oi)&set(x['issue_number'] for x in ci_bundle['issues'])
 input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(IN.rglob('*')) if p.is_file()}
 manifest={'schema_version':'2.0','status':'source_preparation_complete_evaluation_and_comment_review_pending','snapshot_source_date':'2026-10-06','documentation_commit':COMMIT,'input_hashes':input_hashes,
 'counts':{'primary_issues':len(oi),'colleague_issues':len(ci),'overlapping_issues':len(overlap),'archived_unique_issues':len(issues),'archived_unique_comments':len(comments),'scope_matching_issue_candidates':sum(x['in_scope_heuristic'] for x in issues),'retrieval_issues':len(ri),'retrieval_comments_approved':len(approved),'comments_pending_review':len(pending),'documentation_representations_archived':len(docs),'retrieval_documentation_pages':len(rd),'release_versions_archived':len(releases),'release_sections_before_leakage_filter':len(sections),'retrieval_release_sections':len(rr),'heldout_or_pilot_issue_numbers':len(blocked),'fixed_development_cases':15,'fixed_test_cases':15,'multi_turn_scenario_specifications':len(load(previous/'evaluator_only/multi_turn_scenarios.json'))},
 'colleague_findings':{'comment_count_mismatches':[{'issue_number':x['issue_number'],'expected':x['comments_count'],'received':Counter(c['issue_number'] for c in cc)[x['issue_number']]} for x in ci_bundle['issues'] if Counter(c['issue_number'] for c in cc)[x['issue_number']]!=x['comments_count']], 'evaluation_issue_roots_in_original_corpus':sum(f'issue-{n}' in record_ids for n in old_roots),'evaluation_comments_in_original_corpus':sum(c['issue_number'] in old_roots for c in cc),'2020_annual_page_used_as_gold':old_evidence['release-note-d121bacf4c2f3fd5bbb252009bf562c0ebb62045'],'2019_annual_page_used_as_gold':old_evidence['release-note-13c1925ba0243abcef95f9df12086fa2ff3020ae'],'same_tabs_regression_cross_split_candidate':[15892,15893],'in_scope_old_case_candidates':sum(next(x for x in issues if x['number']==n)['in_scope_heuristic'] for n in old_roots)},
 'model_api_calls':0,'embedding_api_calls':0,'model_api_cost_usd':0,'embeddings_built':False,'chunks_built':False,'final_evaluation_approved':False,
 'remaining_gates':['human review of scope and related families','team signoff on preserved case-specific rubrics and multi-turn scenarios','comment claim and media-dependence review','review retained release section topic relevance','check documentation applicability to the user version'],
 'runtime_allowed_paths':['retrieval_corpus/issues.jsonl','retrieval_corpus/comments.jsonl','retrieval_corpus/documents.jsonl','retrieval_corpus/release_sections.jsonl']}
 manifest['schema_version']='3.0'
 manifest['status']='selected_text_and_code_sources_ready_for_chunking_not_runtime_verified'
 manifest['counts']['comments_screened_for_review']=len(pending)
 manifest['counts']['comments_pending_review']=len(remaining)
 manifest['counts']['comments_excluded_after_screening']=sum(x['decision']=='exclude' for x in load(approval))
 manifest['counts']['comments_approved_by_use']=dict(Counter(x['metadata']['allowed_use'] for x in approved))
 manifest['comment_review_provenance']={'reviewer_kind':'ai_assistant','human_reviewed':False,'runtime_tests_performed':False,'source_text_and_thread_context_review':True,'approval_means':'selected excerpts allowed for a specified evidence role, not certification of all statements in a comment'}
 manifest['remaining_gates']=['optional further verification of excluded-from-index pending comments','team signoff on preserved evaluation rubrics and scenarios','match documentation and historical reports to the user version before applying claims']
 manifest['project_budget']={'team_total_api_budget_usd':5,'gateway':'Metis','covers':['development','corpus and query embedding','tests','model evaluation','final runs','bonus work'],'model_selection':'deferred_by_user','external_project_spend_so_far_usd':None,'new_metis_api_calls_during_this_preparation':0,'embedding_expenditure_during_this_preparation_usd':0,'cost_guaranteed':False,'next_cost_gate':'Count selected text with the chosen tokenizer and measure a small pilot using actual Metis rates and remaining balance; do not re-embed unchanged text.'}
 put(ROOT/'manifest.json',manifest)
 print(json.dumps(manifest['counts'],indent=2))

if __name__=='__main__':run()
