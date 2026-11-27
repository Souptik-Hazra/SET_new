import numpy as np
import pandas as pd
from src.features import create_features, leakage_check
from src.scmd_reliability import calibrate_intervals

KEY=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]
def sample():
    n=30
    d={k:[v]*n for k,v in zip(KEY,["ORG001","000123","U1"])}
    d.update({"PERIOD":pd.date_range("2023-01-01",periods=n,freq="MS"),"MODEL_DEMAND":np.arange(n,dtype=float),"OBSERVATION_MISSING":[False]*n,"SPLIT":["train"]*18+["validation"]*6+["test"]*6})
    return pd.DataFrame(d)

def test_monthly_features_are_past_only():
    x=sample(); f=create_features(x)
    assert f.loc[1,"LAG_1"]==0
    assert f.loc[0,"LAG_1"]!=f.loc[0,"MODEL_DEMAND"]
    assert leakage_check(f)
    assert "MONTH_SIN" in f and "LAG_12" in f

def test_missing_month_target_is_not_fabricated():
    x=sample();x.loc[5,"OBSERVATION_MISSING"]=True
    f=create_features(x)
    assert pd.isna(f.loc[5,"LAG_1"]) is False
    assert pd.isna(f.loc[6, "LAG_1"])

def test_calibration_uses_prior_residuals():
    x=sample().tail(6).copy()
    x["Q10"]=x.MODEL_DEMAND-1;x["Q50"]=x.MODEL_DEMAND-0.5;x["Q90"]=x.MODEL_DEMAND+1
    c=calibrate_intervals(x,min_history=2,window=3)
    assert c.loc[c.index[0],"CAL_Q10"]>=0
    assert c.loc[c.index[0],"ROLLING_MAE"]!=c.loc[c.index[0],"ABS_ERROR"]
    assert set(c.DATA_TYPE) == {"OBSERVED_DEMAND"}
    assert "OUTCOME_TYPE" not in c.columns
