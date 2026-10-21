import json, os, time
from pathlib import Path
from typing import Any, Callable, Optional
import requests
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; CORPUS=ROOT/'data/corpus'; EVAL=ROOT/'data/evaluation'
load_dotenv(ROOT/'.env')

OWNER=os.getenv('GITHUB_OWNER','streamlit')
REPO=os.getenv('GITHUB_REPO','streamlit')
TOKEN=os.getenv('GITHUB_TOKEN','').strip()
DATASET_VERSION=os.getenv('DATASET_VERSION','unknown')
DOCS_COMMIT=os.getenv('DOCS_COMMIT','695e6ce2fb1b23afc6df66b40420ab0013ad72fd')
MAX_ISSUES=int(os.getenv('MAX_ISSUES','500'))
DOMAIN_NAME=os.getenv('DOMAIN_NAME','runtime_state_widgets')
DOMAIN_TERMS=[x.strip().lower() for x in os.getenv('DOMAIN_TERMS','').split(',') if x.strip()]
REQUIRED_LABELS={x.strip().lower() for x in os.getenv('REQUIRED_LABELS','').split(',') if x.strip()}

API=f'https://api.github.com/repos/{OWNER}/{REPO}'
HEAD={
    'Accept':'application/vnd.github+json',
    'X-GitHub-Api-Version':'2026-03-10',
    'User-Agent':'streamlit-support-dataset-builder'
}
if TOKEN:
    HEAD['Authorization']=f'Bearer {TOKEN}'


def ensure():
    for p in (RAW,CORPUS,EVAL):
        p.mkdir(parents=True,exist_ok=True)


def save(p:Path,x:Any):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def load(p:Path):
    return json.loads(p.read_text(encoding='utf-8'))


def text(x):
    return '' if x is None else str(x).replace('\r\n','\n').replace('\r','\n').strip()


def get(url,params=None):
    for i in range(6):
        r=requests.get(url,headers=HEAD,params=params,timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (403,429):
            retry_after=r.headers.get('Retry-After')
            if retry_after and retry_after.isdigit():
                delay=min(int(retry_after),120)
            else:
                delay=min(2**i,60)
            print(f'  GitHub returned {r.status_code}; retrying in {delay}s...',flush=True)
            time.sleep(delay)
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r


def next_url_from_response(response):
    link=response.headers.get('Link','')
    for part in link.split(','):
        section=part.strip()
        if 'rel="next"' in section:
            start=section.find('<')
            end=section.find('>',start+1)
            if start >= 0 and end > start:
                return section[start+1:end]
    return None


def paginate(url,params=None,max_items=None,progress:Optional[Callable[[int,int,int],None]]=None):
    """Fetch a paginated GitHub endpoint using Link headers.

    progress(page, total_items_seen, batch_size) is called after each page.
    This avoids requesting invalid pages near GitHub's pagination boundary.
    """
    params=dict(params or {})
    params.setdefault('per_page',100)
    out=[]
    page=1
    next_url=url
    next_params=params

    while next_url:
        response=get(next_url,next_params)
        batch=response.json()
        if not isinstance(batch,list):
            raise RuntimeError(f'Expected a JSON list from {next_url}, got {type(batch).__name__}')

        out.extend(batch)
        if progress:
            progress(page,len(out),len(batch))

        if max_items and len(out) >= max_items:
            return out[:max_items]

        next_url=next_url_from_response(response)
        next_params=None
        page+=1

        if not next_url:
            break

    return out


def match_domain(item):
    labels=[text(x.get('name')) for x in item.get('labels',[]) if isinstance(x,dict)]
    if REQUIRED_LABELS and not REQUIRED_LABELS.intersection({x.lower() for x in labels}):
        return False,[]
    hay=' '.join([text(item.get('title')),text(item.get('body')),' '.join(labels)]).lower()
    hits=[t for t in DOMAIN_TERMS if t in hay]
    return bool(hits),hits
