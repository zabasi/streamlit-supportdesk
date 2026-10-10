"""Write tests/golden/baseline_results.json from the current baseline. Run deliberately."""
import json

from supportdesk.pipelines.baseline import build_baseline
from tests.golden.scenarios import GOLDEN, run_scenarios

if __name__ == "__main__":
    data = run_scenarios(build_baseline)
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {GOLDEN} ({len(data)} scenarios)")
