#!/usr/bin/env python3
"""Build a reproducible evaluation split from the delivered source snapshot.

No network requests, embeddings, or model calls. This prepares data and scenario
specifications, not an implemented agent or executed operational evaluation.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from case_design import TOPICS, SEMANTIC_QUARANTINE_EDGES, RUBRICS, SCENARIOS


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')


def section_evidence(doc, heading):
    text = doc['body']
    lines = text.splitlines(keepends=True)
    positions = []
    cursor = 0
    fence = False
    for line in lines:
        if line.lstrip().startswith('```') or line.lstrip().startswith('~~~'):
            fence = not fence
        if not fence and re.match(r'^#{1,6} ', line):
            positions.append((cursor, line.strip(), len(line)-len(line.lstrip('#'))))
        cursor += len(line)
    for i,(start,h,level) in enumerate(positions):
        if h != heading:
            continue
        end = len(text)
        for later,next_h,next_level in positions[i+1:]:
            if next_level <= level:
                end = later
                break
        return {'source_id':doc['source_id'], 'source_url':doc['html_url'],
                'heading':heading, 'start_char':start, 'end_char':end,
                'text':text[start:end], 'evidence_sha256':sha(text[start:end]),
                'commit':doc.get('commit'), 'retrieved_at':doc['retrieved_at'],
                'version_scope':'Snapshot documentation; check explicit version notes before applying to an older report.'}
    raise ValueError(f'Missing heading {heading!r} in {doc["source_id"]}')


def direct_reference(body, blocked):
    found = set()
    for m in re.finditer(r'(?:github\.com/streamlit/streamlit/(?:issues|pull)/|(?<!\w)#|gh-)(\d+)\b', body):
        if int(m.group(1)) in blocked:
            found.add(int(m.group(1)))
    return sorted(found)


def filter_blocks(row, blocked, audit):
    # List-item paragraphs are retained/removed as units, keeping continuation
    # lines with the referenced item. Original source body hash remains recorded.
    text = row['body']
    blocks = re.split(r'(?m)(?=^\s*[-*] )', text)
    kept = []
    for block in blocks:
        hits = direct_reference(block, blocked)
        if hits:
            audit.append({'source_id':row['source_id'], 'reason':'direct_reference_to_quarantined_case_or_linked_PR',
                          'referenced_numbers':hits, 'removed_text':block,
                          'removed_sha256':sha(block)})
        else:
            kept.append(block)
    new = dict(row)
    new['original_snapshot_body_sha256'] = sha(text)
    new['body'] = ''.join(kept)
    new['retrieval_body_sha256'] = sha(new['body'])
    new['retrieval_filter_applied'] = new['body'] != text
    # Retain original content_sha256 as provenance only; it hashes the original
    # full text, whereas retrieval_body_sha256 hashes the filtered body.
    assert not direct_reference(new['body'], blocked)
    return new


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    src=args.dataset/'data'; out=args.output
    out.mkdir(parents=True, exist_ok=True)
    issues=read_jsonl(src/'issues.jsonl'); comments=read_jsonl(src/'comments.jsonl')
    docs=read_jsonl(src/'documents.jsonl'); releases=read_jsonl(src/'release_notes_all.jsonl')
    relationships=read_jsonl(src/'relationships.jsonl')
    by_issue={r['number']:r for r in issues}; by_comment={c['source_id']:c for c in comments}
    thread=collections.defaultdict(list)
    for c in comments: thread[c['issue_number']].append(c)
    selected={n for _,dev,test in TOPICS for n in dev+test}
    assert len(selected)==30 and selected==set(RUBRICS)
    assert not any(by_issue[n]['pilot_previously_inspected'] for n in selected)
    # Reference graph is a conservative quarantine graph, not a verified
    # duplicate/root-cause graph. PR references share this numeric namespace
    # conservatively; this can remove more data than strictly necessary.
    graph=collections.defaultdict(set)
    for r in relationships:
        sid=r['source_id']
        a=int(sid.split(':')[1]) if sid.startswith('issue:') else by_comment[sid]['issue_number']
        if r['target_repository']=='streamlit/streamlit':
            b=r['target_number'];graph[a].add(b);graph[b].add(a)
    for a,b in SEMANTIC_QUARANTINE_EDGES:graph[a].add(b);graph[b].add(a)
    components={};blocked=set()
    for a in sorted(selected):
        if a in components:continue
        comp=set();stack=[a]
        while stack:
            n=stack.pop()
            if n in comp:continue
            comp.add(n);stack.extend(graph[n]-comp)
        family='quarantine-'+str(min(comp))
        for n in comp:components[n]=family
        blocked.update(comp)
    excluded={n for n in by_issue if n in blocked}
    inventory=[];inputs={'development':[],'test':[]};golds={'development':[],'test':[]};scenarios=[]
    docmap={d['source_id']:d for d in docs}
    # One canonical representation per URL: pinned text when complete, otherwise
    # rendered text. Live rendered snapshots retain their own retrieval/hash.
    grouped=collections.defaultdict(list)
    for d in docs:grouped[d['html_url']].append(d)
    canonical=[]
    for url,rows in sorted(grouped.items()):
        rows.sort(key=lambda d:(d['requires_rendered_api_content'],d['representation']!='source_markdown',d['source_id']))
        canonical.append(rows[0])
    canonical_ids={d['source_id'] for d in canonical}
    leak_audit=[]
    canonical=[filter_blocks(d,blocked,leak_audit) for d in canonical]
    canon_map={d['source_id']:d for d in canonical}
    for topic,dev,test in TOPICS:
        for split,nums in [('development',dev),('test',test)]:
            for n in nums:
                issue=by_issue[n];case_id=('DEV-' if split=='development' else 'TEST-')+str(n)
                cat,reason,next_step,forbidden,anchors=RUBRICS[n]
                env={k:(issue.get(k) or {}).get('value') for k in ['streamlit_version','python_version','operating_system','browser']}
                # The full report remains authoritative when the conservative
                # environment parser missed a value in nonstandard formatting.
                initial={'case_id':case_id,'split':split,'title':issue['title'],
                         'user_report':issue['body'],'reported_environment':env,
                         'environment_note':'The report text is authoritative; null means not extracted, not necessarily absent.',
                         'initial_conversation':[], 'input_origin':'original_issue_body_current_snapshot',
                         'historical_evaluation':False}
                inputs[split].append(initial)
                evidence=[]
                for sid,heading in anchors:
                    assert sid in canonical_ids
                    evidence.append(section_evidence(canon_map[sid],heading))
                oracle=[]
                for c in thread[n]:
                    if c['is_bot'] or 'automated triage comment' in c['body'].lower() or ('react with a' in c['body'].lower() and 'prioritize' in c['body'].lower()):
                        continue
                    oracle.append({'source_id':c['source_id'],'source_url':c['html_url'],
                                   'body':c['body'],'author_association':c['author_association'],
                                   'role':'withheld_context_for_curator_review_not_retrievable_evidence'})
                # Release-note direct answers are retained ONLY for the evaluator.
                release_oracle=[]
                for r in releases:
                    if direct_reference(r['body'],{n}):
                        release_oracle.append({'source_id':r['source_id'],'version':r['version'],
                                               'release_date':r['release_date'],'source_url':r['html_url'],
                                               'body':r['body'],'role':'withheld_direct_answer_do_not_require_the_agent_to_retrieve_it'})
                gold={'case_id':case_id,'split':split,'primary_topic':topic,
                      'case_category':cat,'reference_reasoning':reason,'acceptable_next_step':next_step,
                      'unacceptable_conclusion':forbidden,'acceptable_retrievable_evidence':evidence,
                      'oracle_only_comments':oracle,'oracle_only_direct_release_notes':release_oracle,
                      'output_requirements':['User-facing proposed response','Maintainer summary of known facts, performed checks, evidence, uncertainties, and next step','Clear distinction between reported fact, documentation, and hypothesis'],
                      'rubric':{'evidence_relevance_and_support':2,'version_and_uncertainty_handling':2,'decision_or_targeted_question':2,'no_needless_repeated_check':2,'user_response_and_maintainer_summary':2},
                      'review_status':'assistant_source_review_complete_team_human_signoff_pending',
                      'notes':['Multiple defensible answers are allowed; this is a rubric, not an exact-string answer.',
                               'An exact fixed version is not required when its only evidence was withheld.',
                               'Do not apply modern key-identity rules universally to historical versions.']}
                golds[split].append(gold)
                inventory.append({'case_id':case_id,'split':split,'issue_number':n,'primary_topic':topic,
                                  'case_category':cat,'multi_turn':n in SCENARIOS,
                                  'quarantine_family_id':components[n],'title':issue['title'],
                                  'source_url':issue['html_url'],'comment_count':issue['comment_count'],
                                  'review_status':gold['review_status']})
                write_json(out/'evaluator_only'/'source_threads'/f'{n}.json',{'issue':issue,'comments':thread[n]})
                if n in SCENARIOS:
                    sc=dict(SCENARIOS[n]);cid=sc.get('comment_id')
                    if cid:
                        assert by_comment[cid]['issue_number']==n
                        reply=by_comment[cid]['body']
                        if 'reply' in sc:assert sc['reply']==reply
                    else:reply=sc['reply']
                    scenarios.append({'scenario_id':'MT-'+case_id,'case_id':case_id,'split':split,
                                      'condition':sc['condition'],'initial_input_ref':case_id,
                                      'turns':[{'turn':1,'kind':'initial_user_report','input_ref':case_id},
                                               {'turn':2,'kind':'user_followup','origin':sc['origin'],
                                                'source_id':cid,'body':reply,'release_after':'assistant_response_to_turn_1'}],
                                      'expected_decision_after_followup':sc['expected'],
                                      'evaluator_events':sc['events'],'tracker_assertions':sc['assertions'],
                                      'execution_status':'scenario_specification_not_executed',
                                      'event_origin':'designed_local_tracker_test_not_real_GitHub_activity'})
    # Any quarantined reference component that contains selected cases has one split.
    family_splits=collections.defaultdict(set)
    for r in inventory:family_splits[r['quarantine_family_id']].add(r['split'])
    assert all(len(s)==1 for s in family_splits.values())
    for split in inputs:
        write_jsonl(out/'agent_inputs'/f'{split}_inputs.jsonl',inputs[split])
        write_jsonl(out/'evaluator_only'/f'{split}_rubrics.jsonl',golds[split])
    write_json(out/'evaluator_only'/'multi_turn_scenarios.json',scenarios)
    fields=list(inventory[0])
    with (out/'selected_cases.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(inventory)
    retained_issues=[i for i in issues if i['number'] not in excluded]
    retained_comments=[c for c in comments if c['issue_number'] not in excluded]
    assert not any(direct_reference(i['body'],blocked) for i in retained_issues)
    assert not any(direct_reference(c['body'],blocked) for c in retained_comments)
    retained_releases=[filter_blocks(r,blocked,leak_audit) for r in releases]
    # Release selection is still by topic keywords. Rows include only surviving
    # relevant list blocks, not the entire release text or unrelated changes.
    release_sections=[]
    kw=re.compile(r'session[ _-]?state|widget|callback|multipage|multi-page|st\.form|form_submit|st\.rerun|fragment|websocket|web.socket',re.I)
    for r in retained_releases:
        blocks=re.split(r'(?m)(?=^\s*[-*] )',r['body'])
        for j,b in enumerate(blocks):
            if not b.lstrip().startswith(('- ','* ')) or not kw.search(b):continue
            section={k:r[k] for k in ['version','release_date','html_url','source_url','commit','retrieved_at']}
            section.update({'source_id':r['source_id']+':section:'+str(j),'parent_source_id':r['source_id'],
                            'body':b.strip(),'body_sha256':sha(b.strip()),
                            'original_snapshot_body_sha256':r['original_snapshot_body_sha256'],
                            'selection_method':'surviving_topic_keyword_list_blocks_requires_relevance_review'})
            release_sections.append(section)
    write_jsonl(out/'retrieval_corpus'/'issues.jsonl',retained_issues)
    write_jsonl(out/'retrieval_corpus'/'comments.jsonl',retained_comments)
    write_jsonl(out/'retrieval_corpus'/'documents.jsonl',canonical)
    write_jsonl(out/'retrieval_corpus'/'release_sections.jsonl',release_sections)
    write_jsonl(out/'audit'/'removed_source_blocks.jsonl',leak_audit)
    write_json(out/'audit'/'exclusion_manifest.json',{
        'selected_evaluation_issues':sorted(selected),'quarantined_collected_issues':sorted(excluded),
        'additional_quarantined_collected_issues':sorted(excluded-selected),
        'blocked_reference_numbers_including_external_issue_or_PR_targets':sorted(blocked),
        'quarantine_family_members':{f:sorted(n for n,x in components.items() if x==f) for f in sorted(set(components.values()))},
        'semantic_quarantine_edges':SEMANTIC_QUARANTINE_EDGES,
        'policy':'All explicit-reference connected components of evaluation cases are conservatively withheld. These are not all confirmed duplicate families.',
        'deduplicated_document_representations':len(docs)-len(canonical),
        'excluded_release_and_document_blocks':len(leak_audit)})
    split_counts={s:dict(collections.Counter(r['case_category'] for r in inventory if r['split']==s)) for s in inputs}
    summary={'schema_version':'1.0','source_snapshot_date':'2026-10-06',
             'evaluation_cases':len(inventory),'development_cases':len(inputs['development']),'test_cases':len(inputs['test']),
             'multi_turn_scenarios':len(scenarios),'development_multi_turn':sum(s['split']=='development' for s in scenarios),
             'test_multi_turn':sum(s['split']=='test' for s in scenarios),'category_counts_by_split':split_counts,
             'topic_counts':{topic:{'development':len(dev),'test':len(test)} for topic,dev,test in TOPICS},
             'retrieval_issues':len(retained_issues),'retrieval_comments':len(retained_comments),
             'retrieval_distinct_documents':len(canonical),'retrieval_release_sections':len(release_sections),
             'retrieval_release_versions':len(set(x['version'] for x in release_sections)),
             'quarantined_collected_issues':len(excluded),'additional_quarantined_issues':len(excluded-selected),
             'removed_direct_answer_blocks':len(leak_audit),'retrieval_index_built':False,
             'model_and_embedding_api_calls':0,'additional_model_api_cost_usd':0,
             'selection_status':'fixed_reproducible_selection','rubric_status':'team_human_signoff_pending',
             'operational_evaluation_status':'scenario_specs_prepared_not_executed',
             'historical_evaluation':False,
             'source_file_sha256':{name:sha((src/name).read_bytes()) for name in ['issues.jsonl','comments.jsonl','documents.jsonl','release_notes_all.jsonl','relationships.jsonl']}}
    write_json(out/'summary.json',summary)
    readme=f'''# Streamlit Session State: 30-case Evaluation Split

This is a fixed selection from the 2026-10-06 source snapshot, not an executed evaluation or an implemented agent. All authored files are English; source text is preserved in its original language.

## What is ready

- 15 development and 15 final-test cases; five topics with three cases per split per topic.
- 10 multi-turn scenario specifications, five in each split. Four use real user-comment follow-ups and six use explicitly marked designed follow-ups. All local-tracker events are designed specifications.
- Per-case proposed evaluation rubrics, acceptable document sections with character offsets, and withheld source threads for curator review.
- A separate retrieval corpus: {summary['retrieval_issues']} issues, {summary['retrieval_comments']} comments, {summary['retrieval_distinct_documents']} distinct documentation pages, and {summary['retrieval_release_sections']} release-note sections across {summary['retrieval_release_versions']} versions.
- {len(excluded)} issue threads withheld: 30 evaluation cases plus {len(excluded-selected)} conservatively linked/semantically similar reports. Direct-answer release/document blocks are removed and recorded in the audit.

## Access boundaries

Only index `retrieval_corpus/`. Never index `agent_inputs/`, `evaluator_only/`, `audit/`, `selected_cases.csv`, or the original unsplit dataset. Never make the evaluator directory accessible through the agent's file/search tools.

`agent_inputs/development_inputs.jsonl` and `agent_inputs/test_inputs.jsonl` contain only each scenario's initial report; they omit final labels, outcomes, all comments, gold answers, and future turns. Give the agent one current case at a time. The runner must reveal one follow-up only after the preceding assistant turn. Its tool access must enforce that boundary; file layout alone is not a security control.

Development cases may be used repeatedly to adjust prompts, retrieval, and models. Final-test rubrics may be inspected by the designated dataset curator to validate the dataset, but must not be used to tune the system. Freeze the system before final testing. Compare baseline and final system on the same cases and source snapshot.

## What still needs implementation or review

The proposed rubrics have been reviewed against the source text by the assistant, not independently signed off by a human domain expert. The team should review the 30 rubrics, accept multiple defensible responses, and correct any evidence/decision ambiguity before running evaluation. This does not require line-by-line human review of all 220 source issues.

The multi-turn JSON is an evaluator specification, not a working scenario runner. Implement persistent local tracker storage, proposal-specific/revision-specific approval, idempotent updates, fault injection, sequential turn delivery, and stored-state assertions. No operational tests or LLM response scoring have run in this preparation step.

The original source bodies are current snapshots, not reconstructed historical reports. Therefore evaluation is **current-snapshot replay**, not historical evaluation. Source documentation may describe newer behavior than an old report; every rubric requires version-aware interpretation. An exact fixed release is not required when its only direct supporting evidence is withheld. Do not claim a fix version from a closure label.

Explicit references are a conservative quarantine mechanism, not proof of duplicates. Additional semantic quarantines protect closely similar prepopulation/callback reports. Some topical overlap is necessary for retrieval; direct duplicates, linked answers, and exact outcome references are withheld. The package does not prove that all possible semantic leakage has been eliminated; the team should review remaining corpus items if evaluation identifies suspicious overlaps.

The release-note subset is keyword-selected after direct-reference filtering and still benefits from relevance review. Canonical documentation selects pinned complete Markdown when available and rendered content otherwise. Original hashes and filtered retrieval hashes have separate meanings.

## Multi-turn operational coverage

Normal approval, rejection, maintainer edits, restart while awaiting approval, repeated action request, user fact correction, case changes before execution, timeout after a committed write, failed evidence search, and switching between two cases are each specified once. At least five scenarios belong to final test. Designed turns are marked as designed and must not be presented as actual historical user replies or successful executed tests.

## Rebuild and validate

Use Python 3.10+; scripts depend only on the standard library.

```bash
python scripts/build_evaluation.py --dataset /path/to/Streamlit_Session_State_Dataset_2026-10-06 --output /path/to/new_split
python scripts/validate_split.py --package /path/to/new_split
```

Review `selected_cases.csv` for assignments, `summary.json` for counts, and `audit/exclusion_manifest.json` for exact exclusions. Source collection and this split preparation incurred zero model or embedding API calls. No chunking, embeddings, retrieval index, agent, or evaluated response metrics are included.
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    # Bundle the exact design/build scripts; retain input snapshot identities.
    for name in ['case_design.py','build_evaluation.py','validate_split.py']:
        source=Path(__file__).parent/name
        if source.exists():
            (out/'scripts').mkdir(exist_ok=True)
            shutil.copyfile(source,out/'scripts'/name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
