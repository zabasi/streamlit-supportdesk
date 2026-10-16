"""Synthetic controller tests: not extra development cases or model accuracy tests."""
import copy
import unittest
from supportdesk_generation import generation as g

ROW={'case_id':'UNIT-SUPPORT','input':{'title':'Synthetic support report','body':'A reported issue requiring investigation.','environment':{},'history':[]},
    'sources':[{'source_id':'unit-doc','passages':[{'span_id':'unit-passage','text':'Synthetic reference material.'}]}]}

def assessment():
    return {'case_id':ROW['case_id'],'main_need':'Resolve the reported issue','reported_observation_refs':['R1'],
        'required_human_review':None,'information_gaps':[],
        'answer_candidate':{'message':'A bounded supported example [S1].','evidence_refs':['S1']},
        'internal_investigation':None,'workflow_findings':[]}

def gap(**changes):
    value={'missing_information':'An absent diagnostic fact','material_to_main_need':True,'user_can_supply':True,
        'already_supplied_or_tested':False,'why_it_matters':'The result changes the next diagnostic step.',
        'basis_refs':['R1','S1'],'question':'Can you provide the missing diagnostic fact?',
        'possible_results':[{'result':'The first possible result','next_support_step':'Investigate the first path'},
            {'result':'The other possible result','next_support_step':'Investigate the alternative path'}]}
    value.update(changes);return value

def internal():return {'unresolved_behavior':'An unresolved product interaction','why_internal_review_needed':'The useful user controls are supplied; internal investigation remains.','basis_refs':['R1','S1']}

class Controller(unittest.TestCase):
    def test_supported_answer_without_material_gap(self):
        result=g.rendered(assessment(),ROW);self.assertEqual(result['outcome'],'answer')
        self.assertEqual(result['routing']['decision_owner'],'controller')
    def test_useful_gap_outranks_answer_and_ordinary_investigation(self):
        value=assessment();value['information_gaps']=[gap()];value['internal_investigation']=internal()
        result=g.rendered(value,ROW)
        self.assertEqual(result['outcome'],'request_info')
        self.assertNotIn(value['answer_candidate']['message'],result['user_reply'])
        self.assertEqual(result['user_reply'].count(value['information_gaps'][0]['question']),1)
    def test_human_requirement_takes_priority(self):
        for kind in ('human_authorization','privileged_or_policy_exception','specialist_security_privacy'):
            with self.subTest(kind=kind):
                value=assessment();value['information_gaps']=[gap()]
                value['required_human_review']={'requirement_type':kind,'reason':'A synthetic required review','report_refs':['R1']}
                result=g.rendered(value,ROW);self.assertEqual(result['outcome'],'escalate')
                self.assertEqual(result['routing']['reason'],'required_human_review');self.assertEqual(result['tracker_writes'],0)
    def test_product_investigation_cannot_claim_mandatory_authority_type(self):
        value=assessment();value['required_human_review']={'requirement_type':'framework_bug','reason':'Needs ordinary debugging','report_refs':['R1']}
        with self.assertRaises(ValueError):g.validate(value,ROW)
    def test_ordinary_investigation_without_useful_gap(self):
        value=assessment();value['answer_candidate']=None;value['internal_investigation']=internal()
        self.assertEqual(g.route(value,ROW)['reason'],'internal_investigation_no_useful_question')
    def test_irrelevant_or_already_supplied_gap_does_not_block_answer(self):
        for changes in ({'material_to_main_need':False},{'already_supplied_or_tested':True}):
            value=assessment();value['information_gaps']=[gap(**changes)]
            self.assertEqual(g.route(value,ROW)['outcome'],'answer')
    def test_internal_missing_fact_blocks_answer(self):
        value=assessment();value['information_gaps']=[gap(user_can_supply=False,question=None,possible_results=[])];value['internal_investigation']=internal()
        self.assertEqual(g.route(value,ROW)['outcome'],'escalate')
    def test_unestablished_route_is_an_error_not_default_escalation(self):
        value=assessment();value['answer_candidate']=None
        with self.assertRaises(ValueError):g.validate(value,ROW)
    def test_question_budget_is_controller_owned(self):
        value=assessment();value['information_gaps']=[gap()]
        self.assertEqual(g.route(value,ROW,questions_used=4)['outcome'],'request_info')
        result=g.rendered(value,ROW,questions_used=5)
        self.assertEqual(result['outcome'],'escalate');self.assertEqual(result['routing']['reason'],'question_limit_reached')
        self.assertNotIn('question',result['action'])
        for invalid in (-1,6,True,'0'):
            with self.assertRaises(ValueError):g.route(value,ROW,questions_used=invalid)
    def test_model_cannot_supply_action_or_limits(self):
        for field,value in [('action',{'outcome':'escalate'}),('questions_used',0),('confidence',.99)]:
            payload=assessment();payload[field]=value
            with self.assertRaises(ValueError):g.validate(payload,ROW)
    def test_missing_question_or_equal_alternatives_fail_validation(self):
        for changes in ({'question':None},{'possible_results':[]},{'possible_results':[{'result':'same','next_support_step':'same'},{'result':'same','next_support_step':'same'}]}):
            value=assessment();value['information_gaps']=[gap(**changes)]
            with self.assertRaises(ValueError):g.validate(value,ROW)
    def test_non_boolean_flags_rejected(self):
        for bad in (0,1,'true',None):
            value=assessment();value['information_gaps']=[gap(material_to_main_need=bad)]
            with self.assertRaises(ValueError):g.validate(value,ROW)
    def test_unknown_or_duplicate_refs_rejected(self):
        for changes in ({'reported_observation_refs':['R1','R1']},{'reported_observation_refs':['S1']},
                {'main_need':'Unsupported reference [S999]'}):
            value=assessment();value.update(changes)
            with self.assertRaises(ValueError):g.validate(value,ROW)
    def test_literal_observations_and_user_history(self):
        history=[{'role':'user','body':'A subsequent user reply'}]
        value=assessment();value['reported_observation_refs']=['R1','H1']
        result=g.rendered(value,ROW,history=history)
        self.assertEqual(result['reported_observations'][1]['quote'],history[0]['body'])
        self.assertEqual(result['reported_observations'][1]['text_sha256'],g.sha(history[0]['body'].encode('utf8')))
        history[0]['role']='assistant'
        with self.assertRaises(ValueError):g.rendered(value,ROW,history=history)
    def test_schema_contains_no_final_action_field(self):
        schema=g.schema(ROW);self.assertNotIn('action',schema['properties']);self.assertNotIn('outcome',schema['properties'])
    def test_nonanswer_findings_not_exposed_as_verified_corrections(self):
        value=assessment();value['information_gaps']=[gap()]
        value['workflow_findings']=[{'report_ref':'R1','problem':'A hypothetical issue','correction':'A conditional check','evidence_refs':['S1']}]
        self.assertEqual(g.rendered(value,ROW)['workflow_findings'],[])
        self.assertEqual(len(value['workflow_findings']),1)
    def test_refinement_limit_and_frozen_context(self):
        value=assessment()
        with self.assertRaises(ValueError):g.route(value,ROW,revision_count=2)
        request=g.request(ROW,'gpt-5.4-mini',questions_used=2)
        self.assertIn('"questions_remaining": 3',request['messages'][1]['content'])
        self.assertNotIn('outcome',request['response_format']['json_schema']['schema']['properties'])

if __name__=='__main__':unittest.main()
