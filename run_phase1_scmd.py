"""Run the SCMD Phase 1 pipeline from the repository root."""
import argparse
import json
from src.scmd_phase1 import run_phase1_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SCMD monthly cohort")
    parser.add_argument("--source", default=None, help="Optional SCMD CSV path")
    parser.add_argument("--min-periods", type=int, default=18)
    parser.add_argument("--min-nonzero", type=int, default=6)
    args = parser.parse_args()
    _, cohort, report = run_phase1_from_csv(
        source=args.source,
        min_periods=args.min_periods,
        min_nonzero=args.min_nonzero,
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"\nCohort rows: {len(cohort):,}")
