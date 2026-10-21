from datetime import datetime,timezone
from common import *


def main():
    ensure()
    selected=[]
    prs=0
    api_items_seen=0
    pages=0

    print('Downloading GitHub Issues...',flush=True)
    print(f'Repository: {OWNER}/{REPO}',flush=True)
    print(f'Domain: {DOMAIN_NAME}',flush=True)
    print(f'Target selected issues: {MAX_ISSUES}',flush=True)
    print('Pagination: following GitHub Link headers; progress is shown per page.',flush=True)
    print('',flush=True)

    # We need access to each page's actual items, so use a page iterator here.
    url=f'{API}/issues'
    params={'state':'all','sort':'updated','direction':'desc','per_page':100}
    next_url=url
    next_params=params

    while next_url and len(selected)<MAX_ISSUES:
        response=get(next_url,next_params)
        batch=response.json()
        if not isinstance(batch,list):
            raise RuntimeError('GitHub Issues response was not a JSON list.')

        pages += 1
        api_items_seen += len(batch)
        page_prs=0
        page_selected=0

        for x in batch:
            if 'pull_request' in x:
                prs += 1
                page_prs += 1
                continue

            ok,hits=match_domain(x)
            if not ok:
                continue

            selected.append({
                'issue_number':x.get('number'),
                'github_id':x.get('id'),
                'title':text(x.get('title')),
                'body':text(x.get('body')),
                'state':x.get('state'),
                'state_reason':x.get('state_reason'),
                'created_at':x.get('created_at'),
                'updated_at':x.get('updated_at'),
                'closed_at':x.get('closed_at'),
                'comments_count':x.get('comments',0),
                'comments_url':x.get('comments_url'),
                'labels':[text(a.get('name')) for a in x.get('labels',[]) if isinstance(a,dict)],
                'author_association':x.get('author_association'),
                'html_url':x.get('html_url'),
                'domain':DOMAIN_NAME,
                'domain_matches':hits
            })
            page_selected += 1

            if len(selected)>=MAX_ISSUES:
                break

        print(
            f'  Page {pages:>3}: fetched {len(batch):>3} | '
            f'API total: {api_items_seen:>5} | '
            f'PRs here: {page_prs:>3} | '
            f'selected here: {page_selected:>3} | '
            f'total selected: {len(selected):>4}/{MAX_ISSUES}',
            flush=True
        )

        if len(selected)>=MAX_ISSUES:
            print('\nTarget reached; stopping collection.',flush=True)
            break

        next_url=next_url_from_response(response)
        next_params=None

    if len(selected) < MAX_ISSUES:
        print('No further GitHub Link-header page was provided.', flush=True)
        print('This is normal when the available result set has ended; GitHub can also impose pagination limits.', flush=True)

    now=datetime.now(timezone.utc).isoformat()
    save(RAW/'issues.json',{
        'dataset_version':DATASET_VERSION,
        'source_type':'github_issue',
        'repository':f'{OWNER}/{REPO}',
        'collected_at':now,
        'selection':{
            'domain':DOMAIN_NAME,
            'terms':DOMAIN_TERMS,
            'required_labels':sorted(REQUIRED_LABELS),
            'method':'broad Issues retrieval -> exclude PRs -> local domain selection'
        },
        'counts':{
            'api_items_seen':api_items_seen,
            'pages_fetched':pages,
            'pull_requests_excluded':prs,
            'selected':len(selected)
        },
        'issues':selected
    })
    print('\nDone.',flush=True)
    print(f'Saved: {RAW / "issues.json"}',flush=True)
    print(f'Pages fetched: {pages}',flush=True)
    print(f'API items seen: {api_items_seen}',flush=True)
    print(f'Pull Requests excluded: {prs}',flush=True)
    print(f'Issues selected: {len(selected)}',flush=True)


if __name__=='__main__':
    main()
