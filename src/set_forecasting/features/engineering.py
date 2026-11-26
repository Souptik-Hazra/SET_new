"""Past-only SCMD features; no current/future target enters a predictor."""
import numpy as np
import pandas as pd
SERIES_KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]
TARGET="MODEL_DEMAND"
LAGS=(1,2,3,6,12)
WINDOWS=(3,6,12)

def create_features(data):
    missing=set(SERIES_KEYS+["PERIOD",TARGET,"OBSERVATION_MISSING"])-set(data.columns)
    if missing: raise ValueError(f"Missing SCMD columns: {sorted(missing)}")
    if data.empty:return data.copy()
    frames=[]
    for _,g in data.groupby(SERIES_KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy();g["PERIOD"]=pd.to_datetime(g.PERIOD)
        missing_obs=g.OBSERVATION_MISSING.fillna(True).astype(bool)
        y=pd.to_numeric(g[TARGET],errors="coerce").where(~missing_obs)
        past=y.shift(1)
        for lag in LAGS:g[f"LAG_{lag}"]=y.shift(lag)
        for w in WINDOWS:
            r=past.rolling(w,min_periods=1);n=past.rolling(w,min_periods=1).count()
            mean=r.mean();std=r.std(ddof=0)
            g[f"ROLL_MEAN_{w}"]=mean;g[f"ROLL_MEDIAN_{w}"]=r.median()
            g[f"ROLL_STD_{w}"]=std;g[f"ROLL_MIN_{w}"]=r.min();g[f"ROLL_MAX_{w}"]=r.max()
            g[f"ROLL_CV_{w}"]=std/mean.replace(0,np.nan)
            g[f"ZERO_RATIO_{w}"]=past.eq(0).rolling(w,min_periods=1).sum()/n.replace(0,np.nan)
            g[f"OBSERVED_RATIO_{w}"]=(~missing_obs).shift(1).astype(float).rolling(w,min_periods=1).mean()
        positive=past.gt(0).fillna(False).to_numpy();age=np.full(len(g),np.nan);last=None
        for i,flag in enumerate(positive):
            age[i]=np.nan if last is None else i-last
            if flag:last=i
        g["MONTHS_SINCE_NONZERO"]=age
        g["PRIOR_NONZERO_MEAN_6"]=past.where(past>0).rolling(6,min_periods=1).mean()
        g["MOMENTUM_3M"]=g.ROLL_MEAN_3/g.ROLL_MEAN_6.replace(0,np.nan)
        g["SEASONAL_RATIO_12"]=g.LAG_12/g.ROLL_MEAN_12.replace(0,np.nan)
        month=g.PERIOD.dt.month
        g["MONTH"]=month;g["QUARTER"]=g.PERIOD.dt.quarter;g["YEAR"]=g.PERIOD.dt.year
        g["MONTH_SIN"]=np.sin(2*np.pi*month/12);g["MONTH_COS"]=np.cos(2*np.pi*month/12)
        g["TREND_INDEX"]=np.arange(len(g),dtype=np.int32);g["TARGET_OBSERVED"]=~missing_obs
        if {"TOTAL_INDICATIVE_COST","POSITIVE_QUANTITY"}.issubset(g.columns):
            cost=pd.to_numeric(g.TOTAL_INDICATIVE_COST,errors="coerce")
            qty=pd.to_numeric(g.POSITIVE_QUANTITY,errors="coerce")
            unit=(cost/qty.where(qty>0)).replace([np.inf,-np.inf],np.nan)
            g["LAG_UNIT_COST_1"]=unit.shift(1)
            g["ROLL_UNIT_COST_3"]=unit.shift(1).rolling(3,min_periods=1).mean()
        frames.append(g)
    return pd.concat(frames,ignore_index=True)

def leakage_check(features):
    required=set(SERIES_KEYS+["PERIOD",TARGET,"OBSERVATION_MISSING","LAG_1"])
    if required-set(features.columns):return False
    for _,g in features.groupby(SERIES_KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD")
        expected=pd.to_numeric(g[TARGET],errors="coerce").where(~g.OBSERVATION_MISSING.fillna(True).astype(bool)).shift(1)
        if not np.allclose(g.LAG_1.to_numpy(float),expected.to_numpy(float),equal_nan=True):return False
    return True
