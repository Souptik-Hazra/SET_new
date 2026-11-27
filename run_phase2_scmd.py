"""Deprecated Phase 2 compatibility command; delegates to package features.

Use the canonical end-to-end command: python scripts/run_pipeline.py --source <CSV>
"""
from pathlib import Path

import pandas as pd

from src.set_forecasting.features.engineering import create_features, leakage_check


def main():
    source = Path("data/processed/scmd_monthly_series.parquet")
    target = Path("data/processed/scmd_features.parquet")
    if not source.is_file():
        raise FileNotFoundError(f"Run canonical Phase 1 ingestion first: {source}")
    features = create_features(pd.read_parquet(source))
    if not leakage_check(features):
        raise RuntimeError("Feature leakage check failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    print(f"Phase 2 complete: {len(features):,} rows, {len(features.columns)} columns")


if __name__ == "__main__":
    main()
