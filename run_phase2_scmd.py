"""Run SCMD monthly feature engineering with explicit output directories."""
from pathlib import Path
import pandas as pd
from src.features import create_features,leakage_check
if __name__=="__main__":
 source=Path("data/processed/scmd_monthly_series.parquet");target=Path("data/processed/scmd_features.parquet")
 if not source.is_file(): raise FileNotFoundError(f"Run Phase 1 first: {source}")
 x=pd.read_parquet(source);f=create_features(x)
 if not leakage_check(f):raise RuntimeError("Feature leakage check failed")
 target.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(target,index=False)
 print(f"Phase 2 complete: {len(f):,} rows, {len(f.columns)} columns")
