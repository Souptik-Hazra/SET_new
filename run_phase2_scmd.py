"""Run SCMD monthly feature engineering."""
import pandas as pd
from src.features import create_features, leakage_check
if __name__=="__main__":
    x=pd.read_parquet("data/processed/scmd_monthly_series.parquet")
    f=create_features(x)
    if not leakage_check(f): raise RuntimeError("Feature leakage check failed")
    f.to_parquet("data/processed/scmd_features.parquet",index=False)
    print(f"Phase 2 complete: {len(f):,} rows, {len(f.columns)} columns")
