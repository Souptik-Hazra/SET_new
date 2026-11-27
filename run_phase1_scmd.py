"""Deprecated Phase 1 compatibility command.

Use: python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
"""
import argparse
import json
from pathlib import Path

import yaml

from src.set_forecasting.data.ingestion import run_phase1_from_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--min-periods", type=int, default=None)
    parser.add_argument("--min-nonzero", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}
    data = config.get("data", {})
    _, cohort, report = run_phase1_from_csv(
        source=args.source,
        min_periods=args.min_periods or data.get("min_periods", 18),
        min_nonzero=args.min_nonzero or data.get("min_nonzero", 6),
        train_frac=data.get("train_frac", 0.6),
        val_frac=data.get("val_frac", 0.2),
        load_series=False,
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"\nCohort rows: {len(cohort):,}")


if __name__ == "__main__":
    main()
