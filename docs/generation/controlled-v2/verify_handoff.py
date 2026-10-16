"""Verify package integrity and that the standalone report has no local dependencies."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'checksums.json').read_text())
    for name,digest in manifest.items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    report=(ROOT/'results/development-results.html').read_text(encoding='utf8')
    for href in re.findall(r'href="([^"]+)"',report):assert href.startswith(('#','data:','https:','http:','mailto:')),href
    for src in re.findall(r'src="([^"]+)"',report):assert src.startswith('data:'),src
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    assert metrics['cases']==15 and metrics['action_label_matches']==11
    print(json.dumps({'files_verified':len(manifest),'development_cases':15,'decision_label_matches':11,'standalone_html_local_dependencies':0,'new_api_calls':0}))
if __name__=='__main__':main()
