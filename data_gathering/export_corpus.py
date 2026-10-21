"""Export exactly the five reviewed retrieval files, without overwriting different data."""
from pathlib import Path
import argparse,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    script=ROOT/'frozen_v06/scripts/load_retrieval_sources.py'
    spec=importlib.util.spec_from_file_location('reviewed_sources',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    records=m.load_sources(ROOT/'frozen_v06')
    assert len(records)==len({r['source_id'] for r in records})==540
    for r in records:assert hashlib.sha256(r['text'].encode('utf-8')).hexdigest()==r['text_sha256']
    if a.output.exists():
        old=[json.loads(l) for l in a.output.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        if sorted(old,key=lambda r:r['source_id'])!=sorted(records,key=lambda r:r['source_id']):raise SystemExit('Different corpus already exists; no file was changed.')
        print('Existing corpus contains the same 540 records. Left unchanged.');return
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='\n') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Exported 540 reviewed source records.')
if __name__=='__main__':main()
