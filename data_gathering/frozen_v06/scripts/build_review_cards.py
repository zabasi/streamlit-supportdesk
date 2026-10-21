#!/usr/bin/env python3
"""Create audit-only review cards. No correctness or media interpretation is inferred."""
import json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text(encoding='utf-8').splitlines() if l.strip()]
def run():
 issues={x['number']:x for x in rows('audit/source_archive/issues.jsonl')}
 threads=defaultdict(list)
 for c in rows('audit/source_archive/comments.jsonl'):threads[c['issue_number']].append(c)
 for thread in threads.values():thread.sort(key=lambda c:(c['created_at'],c['id']))
 cards=[]
 for candidate in rows('review_pending/comments.jsonl'):
  n=candidate['metadata']['issue_number'];thread=threads[n];position=next(i for i,c in enumerate(thread) if c['source_id']==candidate['source_id'])
  def brief(c):return {k:c.get(k) for k in ['source_id','body','created_at','author_association','html_url']}
  cards.append({'source_id':candidate['source_id'],'index_allowed':False,'parent_issue':{k:issues[n].get(k) for k in ['source_id','title','body','html_url','streamlit_version','python_version']},'candidate_text':candidate['text'],'review_flags':candidate['metadata']['review_flags'],'preceding_comments':[brief(c) for c in thread[max(0,position-2):position]],'following_comments':[brief(c) for c in thread[position+1:position+3]],'full_thread_comment_source_ids':[c['source_id'] for c in thread],'review':{'decision':None,'reviewer':None,'approved_text':None,'allowed_use':None,'evidence_source_ids':[],'rationale':None,'version_applicability':None,'media_independence':None},'note':'Read the full archived thread when neighboring comments are insufficient. This card is not an agent or embedding input.'})
 out=ROOT/'review_pending/comment_review_cards.jsonl';out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cards),encoding='utf-8')
 print(json.dumps({'review_cards':len(cards),'automatic_approvals':0}))
if __name__=='__main__':run()
