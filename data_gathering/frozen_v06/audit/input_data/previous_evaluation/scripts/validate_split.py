#!/usr/bin/env python3
"""Validate the delivered split and source/evaluator access boundaries."""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path


def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);a=ap.parse_args();p=a.package
    summary=json.loads((p/'summary.json').read_text());ex=json.loads((p/'audit/exclusion_manifest.json').read_text())
    selected=list(csv.DictReader((p/'selected_cases.csv').open(encoding='utf-8')))
    blocked=set(ex['blocked_reference_numbers_including_external_issue_or_PR_targets'])
    assert len(selected)==30 and len({c['case_id'] for c in selected})==30
    assert collections.Counter(c['split'] for c in selected)=={'development':15,'test':15}
    families=collections.defaultdict(set)
    for c in selected:families[c['quarantine_family_id']].add(c['split'])
    assert all(len(x)==1 for x in families.values())
    for topic in {c['primary_topic'] for c in selected}:
        assert collections.Counter(c['split'] for c in selected if c['primary_topic']==topic)=={'development':3,'test':3}
    documents={d['source_id']:d for d in rows(p/'retrieval_corpus/documents.jsonl')}
    assert len(documents)==20 and len({d['html_url'] for d in documents.values()})==20
    for split in ['development','test']:
        agent=rows(p/'agent_inputs'/f'{split}_inputs.jsonl');gold=rows(p/'evaluator_only'/f'{split}_rubrics.jsonl')
        assert len(agent)==len(gold)==15
        assert {r['case_id'] for r in agent}=={r['case_id'] for r in gold}
        for r in agent:
            assert not {'labels','state','state_reason','closed_at','comments','reference_reasoning','turns','future_turns'} & r.keys()
            assert r['initial_conversation']==[] and r['historical_evaluation'] is False
        for g in gold:
            assert g['reference_reasoning'] and g['acceptable_next_step'] and g['unacceptable_conclusion']
            for e in g['acceptable_retrievable_evidence']:
                doc=documents[e['source_id']];t=doc['body'][e['start_char']:e['end_char']]
                assert t==e['text'] and digest(t)==e['evidence_sha256']
    scenarios=json.loads((p/'evaluator_only/multi_turn_scenarios.json').read_text())
    assert len(scenarios)>=10 and sum(s['split']=='test' for s in scenarios)>=5
    assert len({s['case_id'] for s in scenarios})==len(scenarios)
    for s in scenarios:
        assert len(s['turns'])>=2 and s['turns'][1]['release_after']=='assistant_response_to_turn_1'
        assert s['turns'][1]['origin'] in ['real_comment','designed'] and s['turns'][1]['body']
        assert s['execution_status']=='scenario_specification_not_executed'
    required={'normal_approved_action','maintainer_rejects','maintainer_edits','restart_while_waiting_for_approval',
              'repeated_request','user_corrects_a_fact','case_changes_before_execution','tool_timeout_after_commit',
              'evidence_search_failure','interleaved_cases'}
    assert required <= {s['condition'] for s in scenarios}
    issues=rows(p/'retrieval_corpus/issues.jsonl');comments=rows(p/'retrieval_corpus/comments.jsonl')
    releases=rows(p/'retrieval_corpus/release_sections.jsonl')
    remaining={i['number'] for i in issues}
    assert not remaining & set(ex['quarantined_collected_issues'])
    assert not remaining & {int(c['issue_number']) for c in selected}
    assert all(c['issue_number'] in remaining for c in comments)
    counts=collections.Counter(c['issue_number'] for c in comments)
    assert all(counts[i['number']]==i['comment_count'] for i in issues)
    ref=re.compile(r'(?:github\.com/streamlit/streamlit/(?:issues|pull)/|(?<!\w)#|gh-)(\d+)\b')
    for r in issues+comments+list(documents.values())+releases:
        assert not {int(m.group(1)) for m in ref.finditer(r['body'])} & blocked, r['source_id']
    assert len(issues)==summary['retrieval_issues'] and len(comments)==summary['retrieval_comments']
    assert len(releases)==summary['retrieval_release_sections']
    assert summary['model_and_embedding_api_calls']==0 and summary['retrieval_index_built'] is False
    result={'status':'passed','evaluation_cases':len(selected),'development_cases':15,'test_cases':15,
            'multi_turn_scenarios':len(scenarios),'test_multi_turn':sum(s['split']=='test' for s in scenarios),
            'retrieval_issues':len(issues),'retrieval_comments':len(comments),'documents':len(documents),
            'release_sections':len(releases),
            'checks':['case/topic/split counts','reference-quarantine components do not cross splits',
                      'evaluation threads excluded from corpus','comments complete and attached to retained issues',
                      'initial inputs omit evaluator fields and future turns','multi-turn operational specifications cover required hazards',
                      'one canonical documentation representation per URL','acceptable evidence sections match source offsets and hashes',
                      'no direct references to quarantined issues/PRs in corpus'],
            'limits':['Structural/source checks only; no agent behavior or operational scenario has been executed.',
                      'Semantic leakage review is conservative, not a proof of complete absence.',
                      'Independent team human signoff on rubrics remains pending.']}
    (p/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
