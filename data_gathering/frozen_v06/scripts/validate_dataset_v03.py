#!/usr/bin/env python3
"""Independent offline structural, provenance, and leakage-boundary checks.

PASS is not a semantic truth certificate or an agent evaluation score.
"""
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from load_retrieval_sources import load_sources
from build_dataset import media, document_text

ROOT=Path(__file__).resolve().parents[1]
def rows(rel):return [json.loads(l) for l in (ROOT/rel).read_text().splitlines() if l.strip()]
def read(rel):return json.loads((ROOT/rel).read_text())
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def run():
 m=read('manifest.json');records=load_sources();by={r['source_id']:r for r in records}
 assert len(by)==len(records),'Duplicate source identifiers in retrieval.'
 counts=Counter(r['source_type'] for r in records)
 for typ,key in [('issue','retrieval_issues'),('comment','retrieval_comments_approved'),('documentation','retrieval_documentation_pages'),('release_section','retrieval_release_sections')]:assert counts[typ]==m['counts'][key]
 archive_issues=rows('audit/source_archive/issues.jsonl');archive_comments=rows('audit/source_archive/comments.jsonl');archive_docs=rows('audit/source_archive/documents.jsonl')
 ids={i['number'] for i in archive_issues};assert len(ids)==len(archive_issues)==m['counts']['archived_unique_issues']
 assert len({c['id'] for c in archive_comments})==len(archive_comments)==m['counts']['archived_unique_comments']
 assert all(c['issue_number'] in ids and c['parent_source_id']==f'issue:{c["issue_number"]}' for c in archive_comments)
 for r in archive_issues+archive_comments:assert digest(r['body'])==r['body_sha256']
 info=read('audit/heldout_issue_numbers.json');heldout=set(info['issue_numbers']);blocked=set(info['blocked_reference_numbers_including_external_PRs'])
 issue_numbers={r['metadata']['number'] for r in records if r['source_type']=='issue'}
 assert not issue_numbers&heldout
 comment_counts=Counter(c['issue_number'] for c in archive_comments)
 archive_by_number={r['number']:r for r in archive_issues}
 ref=re.compile(r'(?:github\.com/streamlit/streamlit/(?:issues|pull)/|(?<!\w)#|gh-)(\d+)\b')
 for r in records:
  assert r['text'] and digest(r['text'])==r['text_sha256']
  assert r['metadata']['source_id']==r['source_id']
  assert not {int(n) for n in ref.findall(r['text'])}&blocked,r['source_id']
  if r['source_type']=='issue':
   n=r['metadata']['number'];a=archive_by_number[n]
   assert '2021-07-01'<=a['created_at'][:10]<='2026-10-06'
   assert comment_counts[n]==a['comment_count'],f'Incomplete source thread {n}'
  if r['source_type']=='comment':assert r['metadata']['issue_number'] in issue_numbers
 assert len({r['metadata']['html_url'] for r in records if r['source_type']=='documentation'})==counts['documentation']
 assert all(r['metadata'].get('version') and r['metadata'].get('release_date') for r in records if r['source_type']=='release_section')
 pending=rows('review_pending/comments.jsonl');assert len(pending)==m['counts']['comments_pending_review']
 assert all(r['metadata']['embedding_allowed'] is False and r['metadata']['verification_status']=='unreviewed' for r in pending)
 assert all(r['source_id'] not in by for r in pending if not any(x['source_id']==r['source_id'] and x['decision']=='approve' for x in read('config/comment_reviews.json')))
 reviews=read('config/comment_reviews.json');review_by={x['source_id']:x for x in reviews};comments_by={x['source_id']:x for x in archive_comments}
 assert len(review_by)==len(reviews)==m['counts']['comments_screened_for_review']
 assert Counter(x['decision'] for x in reviews)=={'approve':counts['comment'],'exclude':m['counts']['comments_excluded_after_screening'],'pending':len(pending)}
 assert {x['source_id'] for x in reviews if x['decision']=='approve'}=={x['source_id'] for x in records if x['source_type']=='comment'}
 assert {x['source_id'] for x in reviews if x['decision']=='pending'}=={x['source_id'] for x in pending}
 for r in records:
  if r['source_type']!='comment':continue
  d=review_by[r['source_id']];source=comments_by[r['source_id']]
  assert d['approved_text'] in media(source['body'])[0]
  assert r['metadata']['human_reviewed'] is False and r['metadata']['runtime_verified'] is False
  assert r['metadata']['reviewer_kind']=='ai_assistant' and r['metadata']['embedding_allowed'] is True
  assert all(s in by for s in d['evidence_source_ids'])
  if r['metadata']['media_present']:assert d['media_independence']=='selected_excerpt_is_self_contained_without_media'
  for c in d.get('context_excerpts',[]):
   a=comments_by[c['source_id']];assert a['issue_number']==source['issue_number'] and c['text'] in media(a['body'])[0]
  for e in d.get('official_evidence_spans',[]):
   a=next(x for x in archive_docs if x['source_id']==e['source_id'])
   assert a['body'][e['start_char']:e['end_char']]==e['text'] and digest(e['text'])==e['sha256']
 docs={d['source_id']:d for d in archive_docs};cases=set();family_splits=defaultdict(set);categories={};topics={};evidence_count=0
 inherited=read('audit/inherited_exclusion_manifest.json')
 # Preserve pre-existing conservative graph grouping and check new grouping too.
 inherited_family={n:k for k,v in inherited['quarantine_family_members'].items() for n in v}
 old_family_splits=defaultdict(set)
 for split in ['development','test']:
  agent=rows(f'agent_inputs/{split}_inputs.jsonl');gold=rows(f'evaluator_only/{split}_rubrics.jsonl')
  assert len(agent)==len(gold)==15
  assert {a['case_id'] for a in agent}=={g['case_id'] for g in gold}
  categories[split]=dict(Counter(g['case_category'] for g in gold));topics[split]=dict(Counter(g['primary_topic'] for g in gold))
  assert Counter(g['case_category'] for g in gold)=={'answerable':6,'needs_clarification':6,'needs_escalation':3}
  assert set(Counter(g['primary_topic'] for g in gold).values())=={3}
  for a in agent:
   cid=a['case_id'];assert cid not in cases;cases.add(cid)
   n=int(cid.split('-')[-1]);assert n in heldout and n not in issue_numbers
   assert not {'reference_reasoning','must_ask','gold_evidence_ids','turns','future_turns','labels','state','state_reason','closed_at','comments'}&a.keys()
   assert a['initial_conversation']==[]
   family_splits[archive_by_number[n]['family_id']].add(split);old_family_splits[inherited_family[n]].add(split)
  for g in gold:
   assert g['reference_reasoning'] and g['acceptable_next_step'] and g['unacceptable_conclusion']
   for e in g['acceptable_retrievable_evidence']:
    sid=e['source_id'];assert sid in by
    text=docs[sid]['body'][e['start_char']:e['end_char']]
    assert text==e['text'] and digest(text)==e['evidence_sha256']
    # Original offsets remain authoritative; media/wrapper removal changes
    # formatting only. Ensure the evidence content remains in selected input.
    norm=lambda t:re.sub(r'\s+',' ',document_text(t)).strip()
    assert norm(text) in norm(by[sid]['text']),f'Lost official evidence content {sid}'
    evidence_count+=1
 assert all(len(s)==1 for s in family_splits.values())
 assert all(len(s)==1 for s in old_family_splits.values())
 scenarios=read('evaluator_only/multi_turn_scenarios.json');assert len(scenarios)==10
 assert Counter(s['split'] for s in scenarios)=={'development':5,'test':5}
 assert len({s['case_id'] for s in scenarios})==10 and {s['case_id'] for s in scenarios}<=cases
 for s in scenarios:
  assert s['execution_status']=='scenario_specification_not_executed'
  assert s['turns'][1]['release_after']=='assistant_response_to_turn_1'
 required={'normal_approved_action','maintainer_rejects','maintainer_edits','restart_while_waiting_for_approval','repeated_request','user_corrects_a_fact','case_changes_before_execution','tool_timeout_after_commit','evidence_search_failure','interleaved_cases'}
 assert required<={s['condition'] for s in scenarios}
 for rel,sha in m['input_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 checksums=ROOT/'SHA256SUMS.json'
 if checksums.exists():
  for rel,sha in read('SHA256SUMS.json').items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 assert m['model_api_calls']==m['embedding_api_calls']==m['model_api_cost_usd']==0
 assert m['final_evaluation_approved'] is False and m['embeddings_built'] is False and m['chunks_built'] is False
 result={'structural_status':'passed','semantic_truth_certified':False,'agent_behavior_evaluated':False,'counts':m['counts'],'validated_rubric_evidence_spans':evidence_count,'category_counts':categories,'topic_counts':topics,'checks':['unique source and parent identifiers','count-reconciled archived threads for retained issues','text and source-body hashes','only explicit index allowlist loaded','evaluation and pilot roots absent from retrieval','blocked issue and PR references absent from retrieval text','inherited and additional conservative family groups do not cross evaluation splits','one canonical representation per retained documentation page','60 inherited rubric evidence spans remain exact in archive and their normalized content remains in selected input','initial agent inputs do not preload gold or future turns','10 scenario specifications include 5 test cases and all required operational conditions','unreviewed comments cannot enter the default index input','exact selected comment/context text and honest AI review provenance','complete comment decision partition','input snapshot and delivered file integrity'],'limitations':['Team signoff on evaluator rubrics remains pending.','Rule-based duplicate blocks are conservative; unknown semantic relatives may remain.','No screenshot or video content has been interpreted.','Source selection does not certify every historic report or reproduce proposed solutions.','No agent, tracker, scenarios, or RAG metrics have been executed.','Dollar cost and tokenizer totals await model selection and actual Metis rates/balance.']}
 out=ROOT/'validation.json'
 # Repeated validation does not mutate files already covered by delivery checksums.
 if not checksums.exists():out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':run()
