from datetime import datetime,timezone
from common import *

def main():
    ensure(); p=load(RAW/'issues.json'); out=[]
    total=len(p['issues'])
    print(f'Downloading comments for {total} selected issues...',flush=True)
    for i,x in enumerate(p['issues'],1):
        n=x['issue_number']
        items=paginate(f'{API}/issues/{n}/comments')
        for c in items:
            out.append({'comment_id':c.get('id'),'issue_number':n,'body':text(c.get('body')),'created_at':c.get('created_at'),'updated_at':c.get('updated_at'),'html_url':c.get('html_url'),'author_association':c.get('author_association')})
        print(f'[{i}/{total}] {i/total:.1%} | Issue #{n}: {len(items)} comments | total comments: {len(out)}',flush=True)
    save(RAW/'comments.json',{'dataset_version':DATASET_VERSION,'source_type':'github_issue_comment','repository':f'{OWNER}/{REPO}','collected_at':datetime.now(timezone.utc).isoformat(),'count':len(out),'comments':out})
    print(f'Done. Saved {len(out)} comments.',flush=True)

if __name__=='__main__':main()
