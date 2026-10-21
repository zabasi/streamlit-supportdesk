#!/usr/bin/env python3
"""Explicit index-input loader. Never recursively scan the whole package."""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED=("issues.jsonl","comments.jsonl","documents.jsonl","release_sections.jsonl","linked_references.jsonl")

def load_sources(root=ROOT):
 records=[]
 for name in ALLOWED:
  path=root/'retrieval_corpus'/name
  if path.resolve().parent!=(root/'retrieval_corpus').resolve():raise ValueError('Index source escaped the permitted directory.')
  for line in path.read_text(encoding='utf-8').splitlines():
   if not line.strip():continue
   r=json.loads(line)
   if not isinstance(r.get('text'),str) or not r['text'].strip():raise ValueError('Missing source text.')
   if r['source_type']=='comment' and (r['metadata'].get('verification_status')!='agent_reviewed_for_retrieval_not_runtime_verified' or r['metadata'].get('embedding_allowed') is not True or r['metadata'].get('reviewer_kind')!='ai_assistant' or r['metadata'].get('allowed_use') not in ['reported_observation','diagnostic_context','corroborated_claim']):
    raise ValueError('Unreviewed comment found in index input.')
   records.append(r)
 return records

if __name__=='__main__':
 records=load_sources()
 print(json.dumps({'source_record_counts':dict(Counter(r['source_type'] for r in records)),'total_source_records':len(records),'text_characters':sum(len(r['text']) for r in records),'chunks_built':False,'embeddings_built':False,'note':'Only each record text may be submitted for embedding. These are source records, not final chunks.'},indent=2))
