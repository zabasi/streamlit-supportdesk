"""Retrieval boundary and one-call orchestration. No automatic model/network client."""
from dataclasses import asdict,is_dataclass
import hashlib,json,re
from . import generation as g
from .context import LIMITS,reference_map

def mapping(value):
    if is_dataclass(value):return asdict(value)
    if isinstance(value,dict):return value
    raise TypeError('Expected dictionary or dataclass')

def from_packet(case,packet):
    case=mapping(case);packet=mapping(packet)
    if not isinstance(case.get('case_id'),str) or not case['case_id'].strip():raise ValueError('case_id required')
    chunks=packet.get('chunks')
    if not isinstance(chunks,list) or not chunks:raise ValueError('No retrieval evidence: stop or retrieve again before generation')
    sources=[];seen=set()
    for raw in chunks:
        c=mapping(raw);cid=c['chunk_id'];text=c['text'];meta=c.get('meta') or {}
        if cid in seen:raise ValueError('Duplicate retrieval chunk')
        seen.add(cid)
        if not isinstance(text,str) or not text.strip():raise ValueError('Empty evidence text')
        # One source entry per chunk preserves original retrieval order, even
        # when several chunks have the same parent source. No reranking occurs.
        p={'span_id':cid,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        for k in ['start_char','end_char']:
            if c.get(k) is not None:p[k]=c[k]
        source={'source_id':c['source_id'],'source_type':c['source_type'],'title':c.get('title',''),
            'url':c.get('url',''),'version':c.get('version'),'published_at':c.get('as_of'),
            'source_text_sha256':c.get('source_text_sha256'),'passages':[p]}
        for k in ['evidence_role','version_applicability','allowed_use','usage_constraint','verification_status']:
            source[k]=meta.get(k)
        sources.append(source)
    return {'case_id':case['case_id'],'input':{'case_id':case['case_id'],'title':case.get('title',''),
        'body':case.get('body',''),'environment':case.get('environment') or {},'history':case.get('history') or [],
        'created_at':case.get('created_at')},'sources':sources}

def prepare(row,*,history=None,questions_used=0,revision_count=0):
    request=g.request(row,'gpt-5.4-mini',history=history,questions_used=questions_used,revision_count=revision_count)
    content=request['messages'][1]['content']
    if len(content)>LIMITS['max_input_characters'] or len(content.encode())>LIMITS['max_input_bytes']:
        raise ValueError('Evidence/conversation exceeds context allowance; explicitly select a smaller packet before calling the model')
    return request

def finalize(row,assessment,*,history=None,questions_used=0,revision_count=0):
    result=g.rendered(assessment,row,history=history,questions_used=questions_used,revision_count=revision_count)
    result['assessment']=assessment
    result['reference_map']=reference_map(row,history)
    result['content_warnings']=g.issues(assessment)
    return result

def generate_from_packet(case,packet,complete,*,history=None,questions_used=0,revision_count=0):
    """complete(request) -> raw Chat Completions response dict. Exactly one invocation.

    Caller owns API credentials, timeout, accounting and SDK retry policy. This
    adapter has no hidden retries/critic or automatic follow-up loop.
    """
    row=from_packet(case,packet)
    request=prepare(row,history=history,questions_used=questions_used,revision_count=revision_count)
    response=complete(request)
    choice=response['choices'][0];message=choice['message']
    if choice.get('finish_reason')!='stop' or message.get('refusal'):
        raise ValueError('No complete usable assessment; preserve provider usage and surface the provider error, without inventing a support decision')
    assessment=json.loads(message['content'])
    result=finalize(row,assessment,history=history,questions_used=questions_used,revision_count=revision_count)
    result['usage']=response.get('usage',{})
    return result

def to_pipeline_answer(result):
    """Optional conversion to this repository's ParsedAnswer; no label-parser reuse.

    S references use this generator's evidence order, which may differ from
    packet.labels. Map them explicitly to original chunk IDs before storage.
    """
    from supportdesk.pipelines.parsing import ParsedAnswer
    refs=result['reference_map'];reply=result['user_reply'];cited=[]
    for ref in re.findall(r'\[(S\d+)\]',reply):
        cid=refs[ref]['passage_id']
        if cid not in cited:cited.append(cid)
    return ParsedAnswer(outcome=result['outcome'],user_reply=reply,
        open_questions=[result['action']['question']] if result['outcome']=='request_info' else [],
        next_step=result['maintainer_summary']['next_step'],citations=cited,
        violations=[],confidence=None,raw_output=json.dumps(result['assessment'],ensure_ascii=False))
