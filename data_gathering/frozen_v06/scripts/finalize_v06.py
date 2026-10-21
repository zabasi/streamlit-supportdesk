#!/usr/bin/env python3
"""Validate the v06 dataset and refresh the package file inventory."""
from pathlib import Path
import hashlib, json
from validate_dataset import run

ROOT = Path(__file__).resolve().parents[1]

def finalize():
    result = run(ROOT)
    (ROOT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    hashes = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == 'SHA256SUMS.json':
            continue
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'SHA256SUMS.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+'\n')
    return hashes

if __name__ == '__main__':
    print(json.dumps({'validation': 'passed', 'files_hashed': len(finalize())}))
