#!/usr/bin/env python3
"""Validate supplied chunks and map exact evidence intervals. Does not build chunks."""
from pathlib import Path
import argparse,json,hashlib
from load_retrieval_sources import load_sources
def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def map_evidence(sources,evidences,chunks):
 by={s['source_id']:s for s in sources};ids=set();grouped={}
 for c in chunks:
  if c['chunk_id'] in ids:raise ValueError('Duplicate chunk ID')
  ids.add(c['chunk_id']);s=by[c['source_id']];a,b=c['start_char'],c['end_char']
  if not (type(a) is int and type(b) is int and 0<=a<b<=len(s['text'])):raise ValueError('Invalid chunk offsets')
  if c['source_text_sha256']!=s['text_sha256'] or sha(s['text'])!=s['text_sha256']:raise ValueError('Stale source hash')
  if c['text']!=s['text'][a:b]:raise ValueError('Chunk text differs from parent source slice')
  grouped.setdefault(c['source_id'],[]).append(c)
 results=[]
 for e in evidences:
  s=by[e['source_id']];a,b=e['start_char'],e['end_char']
  if not (type(a) is int and type(b) is int and 0<=a<b<=len(s['text'])):raise ValueError('Invalid evidence offsets')
  if e['source_text_sha256']!=s['text_sha256'] or s['text'][a:b]!=e['text'] or sha(e['text'])!=e['evidence_sha256']:raise ValueError('Stale or invalid evidence span')
  hits=[];intervals=[]
  for c in grouped.get(e['source_id'],[]):
   x,y=max(a,c['start_char']),min(b,c['end_char'])
   if x<y:
    hits.append({'chunk_id':c['chunk_id'],'overlap_start_char':x,'overlap_end_char':y,'overlap_characters':y-x});intervals.append((x,y))
  merged=[]
  for x,y in sorted(intervals):
   if merged and x<=merged[-1][1]:merged[-1]=(merged[-1][0],max(y,merged[-1][1]))
   else:merged.append((x,y))
  covered=sum(y-x for x,y in merged)
  results.append({'evidence_id':e['evidence_id'],'source_id':e['source_id'],'source_text_sha256':s['text_sha256'],'chunk_overlaps':hits,'covered_characters':covered,'span_characters':b-a,'covered_character_fraction':covered/(b-a),'complete_span_covered':covered==b-a,'label_semantics':'Span overlap is traceability, not proof that each overlapping chunk is independently sufficient to answer.'})
 return results
def main():
 p=argparse.ArgumentParser();p.add_argument('--chunks',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
 root=Path(__file__).resolve().parents[1];sources=load_sources(root);e=[]
 for n in ['development_rubrics','test_rubrics','history_variant_rubrics','source_usage_rubrics']:
  for l in (root/'evaluator_only'/f'{n}.jsonl').read_text().splitlines():
   g=json.loads(l)
   for v in g['acceptable_retrievable_evidence']:
    if v['evidence_id'] not in {a['evidence_id'] for a in e}:e.append(v)
 chunks=[json.loads(l) for l in args.chunks.read_text().splitlines() if l.strip()]
 args.out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in map_evidence(sources,e,chunks)))
if __name__=='__main__':main()
