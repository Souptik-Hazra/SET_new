"""Phase 2: leakage-safe monthly features for SCMD organisation/VMP/unit series."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.scmd_phase1 import KEY

TARGET = "MODEL_DEMAND"
def create_features(series_df: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY + ["PERIOD", TARGET, "OBSERVATION_MISSING"])
    missing = required.difference(series_df.columns)
    if missing: raise ValueError(f"Missing SCMD columns: {sorted(missing)}")
    frames=[]
    for _, g in series_df.groupby(KEY, sort=False, dropna=False):
        g=g.sort_values("PERIOD").copy()
        y=g[TARGET].where(~g["OBSERVATION_MISSING"].astype(bool))
        g["SERIES_ID"]=g[KEY].astype(str).agg("|".join, axis=1)
        for lag in (1,2,3,6,12): g[f"LAG_{lag}"]=y.shift(lag)
        prior=y.shift(1)
        for w in (3,6,12):
            r=prior.rolling(w,min_periods=1)
            g[f"ROLL_MEAN_{w}"]=r.mean()
            g[f"ROLL_STD_{w}"]=r.std()
            g[f"ROLL_MAX_{w}"]=r.max()
            g[f"ZERO_RATIO_{w}"]=r.apply(lambda x: float(np.mean(x==0)),raw=True)
            g[f"OBSERVED_RATIO_{w}"]=g["OBSERVATION_MISSING"].shift(1).rolling(w,min_periods=1).apply(lambda x:float(np.mean(x==0)),raw=True)
        g["MONTH"]=g["PERIOD"].dt.month
        g["QUARTER"]=g["PERIOD"].dt.quarter
        g["YEAR"]=g["PERIOD"].dt.year
        g["MONTH_SIN"]=np.sin(2*np.pi*g["MONTH"]/12)
        g["MONTH_COS"]=np.cos(2*np.pi*g["MONTH"]/12)
        g["TREND_INDEX"]=np.arange(len(g),dtype=int)
        g["TARGET_OBSERVED"]=~g["OBSERVATION_MISSING"].astype(bool)
        frames.append(g)
    return pd.concat(frames,ignore_index=True) if frames else series_df.copy()

def leakage_check(features_df: pd.DataFrame) -> bool:
    for _,g in features_df.groupby(KEY,sort=False,dropna=False):
        g=g.sort_values("PERIOD")
        if len(g)>1 and not np.allclose(g["LAG_1"].iloc[1:].to_numpy(dtype=float),g["MODEL_DEMAND"].where(~g["OBSERVATION_MISSING"]).shift(1).iloc[1:].to_numpy(dtype=float),equal_nan=True):
            return False
    return True
