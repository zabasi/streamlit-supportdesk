from datetime import datetime,timezone
import re,requests
from bs4 import BeautifulSoup
from common import *
URL='https://docs.streamlit.io/develop/quick-reference/release-notes'
def main():
    ensure(); s=BeautifulSoup(requests.get(URL,timeout=60).text,'html.parser'); [z.decompose() for z in s(['script','style','noscript'])]
    out=[]; rx=re.compile(r'(?:Version|v)?\s*(\d+\.\d+\.\d+)',re.I)
    for h in s.find_all(['h2','h3']):
        title=text(h.get_text(' ',strip=True)); m=rx.search(title)
        if not m:continue
        parts=[]; n=h.find_next_sibling()
        while n and n.name not in ('h2','h3'):
            if n.name in ('p','ul','ol','pre','table'):
                t=text(n.get_text('\n',strip=True));
                if t:parts.append(t)
            n=n.find_next_sibling()
        body='\n\n'.join(parts); dm=re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})',body,re.I)
        out.append({'release_id':'release-'+m.group(1),'version':m.group(1),'release_date':dm.group(1) if dm else None,'title':title,'text':body,'url':URL,'section':title})
    unique={x['version']:x for x in out}; out=list(unique.values())
    save(RAW/'release_notes.json',{'dataset_version':DATASET_VERSION,'source_type':'release_note','source_url':URL,'collected_at':datetime.now(timezone.utc).isoformat(),'count':len(out),'release_notes':out}); print(f'Saved {len(out)} release notes.')
if __name__=='__main__':main()
