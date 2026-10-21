"""Explicit fresh collection. Default is a network-free execution plan."""
from pathlib import Path
import argparse, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
STEPS=('download_issues.py','download_comments.py','download_documents.py','download_release_notes.py','build_corpus.py')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',help='Actually download a new dataset; does not update frozen_v06')
    args=ap.parse_args()
    for name in STEPS:
        if not (ROOT/'scripts'/name).is_file():raise SystemExit('Missing collection script: '+name)
    if not args.run:
        print('PLAN ONLY: no network requests or files created.')
        for i,name in enumerate(STEPS,1):print(str(i)+'. '+name)
        print('New outputs: acquisition/data/raw and acquisition/data/corpus; frozen_v06 is unchanged.')
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    if not os.getenv('DOMAIN_TERMS','').strip():raise SystemExit('Configure DOMAIN_TERMS in acquisition/.env before downloading.')
    env=dict(os.environ,PYTHONUTF8='1')
    for name in STEPS:
        print('Running '+name,flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)
    print('Fresh early-schema corpus created. Review and quarantine before indexing. No automatic v06 promotion.')
if __name__=='__main__':main()
