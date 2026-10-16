"""Portable, offline prompt construction and structural validation. No API dependencies."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

OUTCOMES=['answer','request_info','escalate']
LIMITS={'max_followup_questions':5,'max_refinements':1,'max_input_characters':60000,'max_input_bytes':48000,'max_history_characters':16000}

def sha(b):return hashlib.sha256(b).hexdigest()
def display(text):return text.replace('\r\n','\n').replace('\r','\n')
def read_jsonl(path):return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
def copy_text(messages):return '\n\n'.join('['+m['role'].upper()+' MESSAGE]\n'+m['content'] for m in messages)+'\n'
def blocks(text):
    """Exact original character spans separated by blank lines, preserving entire fences."""
    result=[];start=0;offset=0;fence=None
    def add(a,b):
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:result.append({'start_char':a,'end_char':b,'text':text[a:b]})
    for line in text.splitlines(keepends=True):
        marker=re.match(r'^\s*(`{3,}|~{3,})(.*)$',line.rstrip('\r\n'))
        if not line.strip() and fence is None:add(start,offset);start=offset+len(line)
        if marker:
            token,tail=marker.groups()
            if fence is None:fence=token
            elif token[0]==fence[0] and len(token)>=len(fence) and not tail.strip():fence=None
        offset+=len(line)
    add(start,len(text));return result

def reference_map(row,history=None):
    case=row['input'];refs={'R0':{'kind':'report','text':case['title']}}
    for i,block in enumerate(blocks(case['body']),1):refs[f'R{i}']={'kind':'report',**block}
    messages=case.get('history',[]) if history is None else history
    if not isinstance(messages,list):raise ValueError('History must be a chronological JSON array')
    for i,m in enumerate(messages,1):
        if set(m)-{'role','body','content','created_at'}:raise ValueError('Unknown history fields')
        if m.get('role') not in ('user','assistant'):raise ValueError('History roles must be user or assistant')
        body=m.get('body',m.get('content'))
        if not isinstance(body,str):raise ValueError('History message needs body/content text')
        refs[f'H{i}']={'kind':'conversation','role':m['role'],'text':body}
    n=0
    for source in row['sources']:
        for passage in source['passages']:
            n+=1;refs[f'S{n}']={'kind':'evidence','source_id':source['source_id'],'passage_id':passage['span_id'],'text':passage['text']}
    return refs

def context(row,*,history=None,questions_used=0,revision_count=0):
    if not isinstance(questions_used,int) or not 0<=questions_used<=LIMITS['max_followup_questions']:raise ValueError('Invalid question count')
    if not isinstance(revision_count,int) or not 0<=revision_count<=LIMITS['max_refinements']:raise ValueError('Invalid revision count')
    refs=reference_map(row,history)
    if sum(len(v['text']) for v in refs.values() if v['kind']=='conversation')>LIMITS['max_history_characters']:raise ValueError('History exceeds allowance; do not silently truncate it')
    parts=['# Case',json.dumps({'case_id':row['case_id']},ensure_ascii=False),'# Original user report (reference data)']
    for ref,item in refs.items():
        if item['kind']=='report':parts.append(f'[{ref}]\n{item["text"]}')
    parts+=['# Known environment',json.dumps(row['input']['environment'],ensure_ascii=False),
        '# Chronological conversation (empty means no earlier messages were supplied)']
    found=False
    for ref,item in refs.items():
        if item['kind']=='conversation':parts.append(f'[{ref}] role={item["role"]}\n{item["text"]}');found=True
    if not found:parts.append('(none supplied)')
    parts.append('# Evidence sources and their selected passages')
    n=0
    for sn,source in enumerate(row['sources'],1):
        meta={k:source[k] for k in ('source_type','title','url','version','published_at','evidence_role','version_applicability','allowed_use','usage_constraint','verification_status') if source.get(k) is not None and source[k]!=''}
        parts.append('SOURCE '+str(sn)+'\n'+json.dumps(meta,ensure_ascii=False))
        parts.append('Each labelled passage is a separate verbatim selection. Intervening source text may be omitted.')
        for passage in source['passages']:
            n+=1;parts.append(f'[S{n}]\n{passage["text"]}')
    parts+=['# Controller-supplied limits (not claims about completed actions)',json.dumps({
        'questions_used':questions_used,'questions_remaining':LIMITS['max_followup_questions']-questions_used,
        'revision_count':revision_count,'max_refinements':LIMITS['max_refinements'],
        'mode':'proposal_only','tracker_execution_available':False},ensure_ascii=False)]
    return display('\n\n'.join(parts))
