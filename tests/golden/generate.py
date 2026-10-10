"""Write the baseline's golden files: its outputs on the scenarios (baseline_results.json) and the
fingerprint of everything that defines it (baseline_fingerprint.json). Run only when a change to the
baseline is intended, and review the diff of both files before committing them."""
import json

from supportdesk.pipelines.baseline import build_baseline
from tests.golden.fingerprint import FINGERPRINT, write
from tests.golden.scenarios import GOLDEN, run_scenarios

if __name__ == "__main__":
    data = run_scenarios(build_baseline)
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    write()
    print(f"wrote {GOLDEN} ({len(data)} scenarios) and {FINGERPRINT}")
