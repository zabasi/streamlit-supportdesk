"""One model assessment, controller-owned routing and reply assembly. No side effects."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path
from .context import context, reference_map, LIMITS

BUNDLE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
def text(values=None):return {'type':'string',**({'enum':values} if values is not None else {'pattern':r'\S'})}
def arr(item,minimum=0,maximum=None):return {'type':'array','items':item,'minItems':minimum,**({'maxItems':maximum} if maximum is not None else {})}
def nullable(spec):return {'anyOf':[spec,{'type':'null'}]}

def schema(row,history=None):
    refs=reference_map(row,history)
    reports=[k for k,v in refs.items() if v['kind']=='report' or v.get('role')=='user']
    evidence=[k for k,v in refs.items() if v['kind']=='evidence']
    basis=text(list(refs))
    boolean={'type':'boolean'}
    return obj({
        'case_id':text([row['case_id']]),'main_need':text(),
        'reported_observation_refs':arr(text(reports),maximum=6),
        'required_human_review':nullable(obj({
            'requirement_type':text(['human_authorization','privileged_or_policy_exception','specialist_security_privacy']),
            'reason':text(),'report_refs':arr(text(reports),1,4)})),
        'information_gaps':arr(obj({
            'missing_information':text(),'material_to_main_need':boolean,
            'user_can_supply':boolean,'already_supplied_or_tested':boolean,
            'why_it_matters':text(),'basis_refs':arr(basis,1,5),
            'question':nullable(text()),
            'possible_results':arr(obj({'result':text(),'next_support_step':text()}),maximum=2),
        }),maximum=3),
        'answer_candidate':nullable(obj({'message':text(),'evidence_refs':arr(text(evidence),1,8)})),
        'internal_investigation':nullable(obj({'unresolved_behavior':text(),'why_internal_review_needed':text(),'basis_refs':arr(basis,1,8)})),
        'workflow_findings':arr(obj({'report_ref':text(reports),'problem':text(),'correction':text(),'evidence_refs':arr(text(evidence),1,8)}),maximum=4),
    })

def check(value,spec,path='$'):
    if 'anyOf' in spec:
        for branch in spec['anyOf']:
            try:check(value,branch,path);return
            except ValueError:pass
        raise ValueError(path+': no nullable branch matches')
    kind=spec['type']
    valid={'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),
        'boolean':type(value) is bool,'null':value is None}[kind]
    if not valid:raise ValueError(path+': wrong type')
    if 'enum' in spec and value not in spec['enum']:raise ValueError(path+': unknown value/reference')
    if 'pattern' in spec and not re.search(spec['pattern'],value):raise ValueError(path+': empty text')
    if kind=='object':
        if set(value)!=set(spec['required']):raise ValueError(path+': missing or extra fields')
        for key,item in value.items():check(item,spec['properties'][key],path+'.'+key)
    if kind=='array':
        if len(value)<spec.get('minItems',0) or len(value)>spec.get('maxItems',len(value)):raise ValueError(path+': array outside allowance')
        for index,item in enumerate(value):check(item,spec['items'],path+f'[{index}]')

def _limits(questions_used,revision_count):
    if type(questions_used) is not int or not 0<=questions_used<=LIMITS['max_followup_questions']:raise ValueError('Invalid trusted question count')
    if type(revision_count) is not int or not 0<=revision_count<=LIMITS['max_refinements']:raise ValueError('Invalid trusted refinement count')

def _route(value,questions_used):
    # This code never receives a model-selected action, confidence score or case label.
    if value['required_human_review'] is not None:return {'outcome':'escalate','reason':'required_human_review','gap_index':None}
    material=[(i,gap) for i,gap in enumerate(value['information_gaps']) if gap['material_to_main_need'] and not gap['already_supplied_or_tested']]
    useful=[(i,gap) for i,gap in material if gap['user_can_supply']]
    if useful:
        if questions_used>=LIMITS['max_followup_questions']:return {'outcome':'escalate','reason':'question_limit_reached','gap_index':useful[0][0]}
        return {'outcome':'request_info','reason':'material_user_providable_gap','gap_index':useful[0][0]}
    if not material and value['answer_candidate'] is not None:return {'outcome':'answer','reason':'supported_answer_candidate','gap_index':None}
    if value['internal_investigation'] is not None:return {'outcome':'escalate','reason':'internal_investigation_no_useful_question','gap_index':None}
    raise ValueError('Assessment does not establish a supported action; no default escalation or repair')

def validate(value,row,*,history=None,questions_used=0,revision_count=0):
    _limits(questions_used,revision_count);check(value,schema(row,history))
    refs=reference_map(row,history)
    for ref in re.findall(r'\[((?:S|R|H)\d+)\]',json.dumps(value,ensure_ascii=False)):
        if ref not in refs:raise ValueError('Unknown inline reference')
    selected=value['reported_observation_refs']
    if len(selected)!=len(set(selected)):raise ValueError('Duplicate observation reference')
    for gap in value['information_gaps']:
        if gap['material_to_main_need'] and gap['user_can_supply'] and not gap['already_supplied_or_tested']:
            if gap['question'] is None or len(gap['possible_results'])!=2:raise ValueError('Useful material gap needs a question and two possible next steps')
            results=gap['possible_results']
            if results[0]['result'].strip().casefold()==results[1]['result'].strip().casefold():raise ValueError('Identical hypothetical results')
            if results[0]['next_support_step'].strip().casefold()==results[1]['next_support_step'].strip().casefold():raise ValueError('Question has no distinct next-step alternatives')
    _route(value,questions_used)
    return value

def route(value,row,*,history=None,questions_used=0,revision_count=0):
    validate(value,row,history=history,questions_used=questions_used,revision_count=revision_count)
    return _route(value,questions_used)

def citations(refs):return ''.join('['+ref+']' for ref in dict.fromkeys(refs))

def rendered(value,row,*,history=None,questions_used=0,revision_count=0):
    routing=route(value,row,history=history,questions_used=questions_used,revision_count=revision_count)
    outcome=routing['outcome'];gap=value['information_gaps'][routing['gap_index']] if routing['gap_index'] is not None else None
    if outcome=='request_info':
        action={'outcome':outcome,'message':gap['why_it_matters']+' '+citations(gap['basis_refs']),'question':gap['question']}
        next_step='Await the user response before choosing between the recorded conditional next steps.'
    elif outcome=='answer':
        message=value['answer_candidate']['message']
        # Findings are part of the supported answer, not hidden integration advice.
        if value['workflow_findings']:
            edits=[finding['correction']+' '+citations([finding['report_ref'],*finding['evidence_refs']])
                for finding in value['workflow_findings']]
            message+='\n\nRequired edits:\n'+'\n'.join('- '+edit for edit in edits)
        action={'outcome':outcome,'message':message}
        next_step='Apply the proposed correction within its stated scope; it has not been executed by this controller.'
    elif routing['reason']=='required_human_review':
        need=value['required_human_review']
        action={'outcome':outcome,'message':'Human review is required: '+need['reason']+' '+citations(need['report_refs'])}
        next_step='Propose review by the appropriate authorized person; no external action has been performed.'
    elif routing['reason']=='question_limit_reached':
        action={'outcome':outcome,'message':'The follow-up question limit has been reached while relevant information remains unresolved: '+gap['missing_information']+'. Human support review is needed to choose the next step.'}
        next_step='Human support should review the unanswered information and existing conversation; no product defect is asserted.'
    else:
        need=value['internal_investigation']
        action={'outcome':outcome,'message':'Maintainer review is needed: '+need['unresolved_behavior']+' '+need['why_internal_review_needed']+' '+citations(need['basis_refs'])}
        next_step='Propose maintainer investigation within the reported environment and supplied controls; no ticket has been created.'
    refs=reference_map(row,history)
    observations=[{'reference':ref,'quote':refs[ref]['text'],'text_sha256':sha(refs[ref]['text'].encode('utf8')),
        'origin':'original_report' if refs[ref]['kind']=='report' else 'user_conversation'} for ref in value['reported_observation_refs']]
    return {'case_id':value['case_id'],'outcome':outcome,'action':action,
        'user_reply':action['message']+ ('\n\n'+action['question'] if outcome=='request_info' else ''),
        'routing':{**routing,'decision_owner':'controller','questions_used':questions_used,
            'questions_remaining':LIMITS['max_followup_questions']-questions_used,
            'semantic_flags_are_model_assessments':True},
        'reported_observations':observations,'workflow_findings':value['workflow_findings'] if outcome=='answer' else [],
        'maintainer_summary':{'problem':value['main_need'],'reported_environment':row['input']['environment'],
            'uncertainties':[gap['missing_information'] for gap in value['information_gaps'] if gap['material_to_main_need'] and not gap['already_supplied_or_tested']],
            'next_step':next_step},'tracker_writes':0,'status':'proposal_requires_content_review'}

def issues(value):
    warnings=[]
    for gap in value['information_gaps']:
        if gap['material_to_main_need'] and not gap['already_supplied_or_tested'] and value['answer_candidate'] is not None:
            warnings.append('Answer candidate suppressed because a material gap remains');break
    candidate=value['answer_candidate']
    if candidate:
        if not re.search(r'\[S\d+\]',candidate['message']):warnings.append('Answer draft has no inline source citation')
        for snippet in re.findall(r'```(?:python|py)[ \t]*\n(.*?)```',candidate['message'],re.I|re.S):
            try:ast.parse(snippet)
            except SyntaxError:warnings.append('Answer draft Python has invalid syntax; not executed')
    return warnings

def request(row,model,*,history=None,questions_used=0,revision_count=0):
    if model!='gpt-5.4-mini':raise ValueError('Model outside the selected diagnostic')
    _limits(questions_used,revision_count)
    return {'model':model,'messages':[{'role':'system','content':(BUNDLE/'prompt.txt').read_text(encoding='utf8')},
        {'role':'user','content':context(row,history=history,questions_used=questions_used,revision_count=revision_count)}],
        'response_format':{'type':'json_schema','json_schema':{'name':'supportdesk_controller_assessment_v1','strict':True,'schema':schema(row,history)}},
        'max_completion_tokens':8000,'reasoning_effort':'medium'}

def rows():
    manifest=json.loads((BUNDLE/'manifest.json').read_text(encoding='utf8'))
    for relative,digest in manifest['frozen_files'].items():
        if sha((BUNDLE/relative).read_bytes())!=digest:raise ValueError('Frozen candidate changed; create a new version')
    return [json.loads(line) for line in (BUNDLE/'inputs/model_inputs.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
