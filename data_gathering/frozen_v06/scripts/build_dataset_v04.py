#!/usr/bin/env python3
"""Reconstruct selected v04 sources from the fixed delivered v03 baseline.

No source fetch, new selection, evaluation mutation, code execution, or API call.
"""
from pathlib import Path
from build_dataset_v03 import media, document_text
from clean_for_chunking_v04 import run
from finalize_v04 import finalize, checksums

if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 (root/'SHA256SUMS.json').unlink(missing_ok=True)
 run(root)
 finalize()
 from validate_dataset import run as validate
 validate()
 checksums()
