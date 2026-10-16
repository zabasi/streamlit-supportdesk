import copy,json,unittest
from pathlib import Path
from unittest.mock import Mock
from supportdesk_generation import from_packet,prepare,finalize,generate_from_packet
from supportdesk_generation import generation as g

ROOT=Path(__file__).resolve().parents[1]
CASE={'case_id':'INTEGRATION-UNIT','title':'User report','body':'Observed behavior.','environment':{'streamlit_version':'reported'},'history':[]}
PACKET={'chunks':[{'chunk_id':'chunk-a','source_id':'source-a','source_type':'documentation','title':'Reference','text':'First exact evidence.','version':'snapshot','meta':{'evidence_role':'official_documentation'}},
    {'chunk_id':'chunk-b','source_id':'source-a','source_type':'documentation','title':'Reference','text':'Second disjoint evidence.','meta':{}}],
    'labels':{'S1':'chunk-b','S2':'chunk-a'}}
def assessment():
    return {'case_id':CASE['case_id'],'main_need':'Resolve the report','reported_observation_refs':['R1'],
        'required_human_review':None,'information_gaps':[],
        'answer_candidate':{'message':'Supported scoped explanation [S1].','evidence_refs':['S1']},
        'internal_investigation':None,'workflow_findings':[]}

class Integration(unittest.TestCase):
    def test_preserves_exact_retrieval_order_metadata_and_disjoint_passages(self):
        row=from_packet(CASE,PACKET)
        self.assertEqual([s['passages'][0]['text'] for s in row['sources']],['First exact evidence.','Second disjoint evidence.'])
        self.assertEqual(row['sources'][0]['version'],'snapshot')
        self.assertEqual(g.reference_map(row)['S1']['passage_id'],'chunk-a')
    def test_rejects_no_evidence_without_calling_transport(self):
        call=Mock()
        with self.assertRaises(ValueError):generate_from_packet(CASE,{'chunks':[]},call)
        call.assert_not_called()
    def test_rejects_duplicate_chunk_without_calling_transport(self):
        call=Mock();p=copy.deepcopy(PACKET);p['chunks'][1]=p['chunks'][0]
        with self.assertRaises(ValueError):generate_from_packet(CASE,p,call)
        call.assert_not_called()
    def test_context_byte_allowance_before_network(self):
        call=Mock();p=copy.deepcopy(PACKET);p['chunks'][0]['text']='a'*60001
        with self.assertRaises(ValueError):generate_from_packet(CASE,p,call)
        call.assert_not_called()
    def test_one_model_invocation_and_preserved_citations(self):
        call=Mock(return_value={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(assessment())}}],'usage':{'prompt_tokens':8}})
        result=generate_from_packet(CASE,PACKET,call)
        self.assertEqual(call.call_count,1);self.assertEqual(result['outcome'],'answer')
        self.assertEqual(result['reference_map']['S1']['passage_id'],'chunk-a');self.assertEqual(result['usage'],{'prompt_tokens':8})
    def test_truncation_has_no_retry_or_fabricated_escalation(self):
        call=Mock(return_value={'choices':[{'finish_reason':'length','message':{'content':'{}'}}]})
        with self.assertRaises(ValueError):generate_from_packet(CASE,PACKET,call)
        self.assertEqual(call.call_count,1)
    def test_question_allowance_blocks_untrusted_count(self):
        with self.assertRaises(ValueError):prepare(from_packet(CASE,PACKET),questions_used=6)
    def test_identified_edits_are_in_answer(self):
        a=assessment();a['workflow_findings']=[{'report_ref':'R1','problem':'Shown problem','correction':'Apply the identified edit.','evidence_refs':['S1']}]
        self.assertIn('Apply the identified edit. [R1][S1]',finalize(from_packet(CASE,PACKET),a)['user_reply'])
    def test_all_15_frozen_requests_and_saved_outputs_replay_identically(self):
        records=[json.loads(l) for l in (ROOT/'data/replay.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(records),15)
        for r in records:
            with self.subTest(case=r['row']['case_id']):
                self.assertEqual(prepare(r['row']),r['request'])
                result=finalize(r['row'],r['assessment'])
                self.assertEqual({k:result[k] for k in r['rendered']},r['rendered'])
if __name__=='__main__':unittest.main()
