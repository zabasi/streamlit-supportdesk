#!/usr/bin/env python3
from pathlib import Path
import json,collections
ROOT=Path(__file__).resolve().parents[1]
a=ROOT/'audit/link_review_v05';files=['issues','comments','documents','release_sections']
records=[json.loads(l) for f in files for l in (a/'baseline_v04'/f'{f}.jsonl').read_text().splitlines()]
new=json.loads((a/'added_sources.json').read_text());records+=new;by={r['source_id']:r for r in records};newids={r['source_id'] for r in new}
for d in json.loads((a/'link_decisions.json').read_text()):
 for sid in d['linked_from_source_ids']:
  r=by[sid];r['metadata'].setdefault('linked_references',[]).append({'url':d['url'],'status':d['status'],'target_source_ids':d['target_source_ids'],'canonical_url':d.get('canonical_url')})
  ids=r['metadata'].setdefault('linked_source_ids',[])
  for target in d['target_source_ids']:
   if target!=sid and target not in ids:ids.append(target)
  r['metadata']['link_preparation_version']='v05'
for typ,name in [('issue','issues'),('comment','comments'),('documentation','documents'),('release_section','release_sections'),('linked_reference','linked_references')]:
 (ROOT/'retrieval_corpus'/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records if r['source_type']==typ))
print(json.dumps({'rebuilt_sources':len(records),'network_calls':0,'evaluation_modified':False}))
