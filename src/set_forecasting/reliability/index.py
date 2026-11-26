"""Prequential four-signal forecast-unreliability index.

Higher values mean worse forecast reliability (greater error, bias, coverage
miscalibration or drift). ``R_UNRELIABILITY`` is the canonical output column;
``R_REL`` is retained as a backwards-compatible alias used by RADCSR.
Q10/Q90 form a nominal central 80% interval. The coverage signal uses a separate
past-only conformalised interval around Q50 to match the configured target.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def build_reliability_index(predictions,target_coverage=.90,window=8,weights=None,min_history=3):
    required=KEYS+["PERIOD","MODEL_DEMAND","Q10","Q50","Q90"]
    missing=set(required)-set(predictions.columns)
    if missing:raise ValueError(f"Missing reliability inputs: {sorted(missing)}")
    w=weights or {"error":.35,"bias":.25,"coverage":.25,"drift":.15}
    expected={"error", "bias", "coverage", "drift"}
    if set(w) != expected or any(not np.isfinite(float(v)) or float(v) < 0 for v in w.values()) or not np.isclose(sum(w.values()), 1):
        raise ValueError("Weights must define error, bias, coverage and drift; be finite, non-negative and sum to one")
    if predictions.duplicated(KEYS + ["PERIOD"]).any():
        raise ValueError("Reliability input must have one row per series and period")
    if not 0 < float(target_coverage) < 1 or int(window) < 1 or int(min_history) < 1:
        raise ValueError("target_coverage must be in (0,1); window and min_history must be positive")
    output=[]
    for _,g in predictions.groupby(KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy()
        y=g.MODEL_DEMAND.to_numpy(float);q10=g.Q10.to_numpy(float)
        q50=g.Q50.to_numpy(float);q90=g.Q90.to_numpy(float)
        err=y-q50;ae=np.abs(err);n=len(g)
        re=np.zeros(n);rb=np.zeros(n);rc=np.zeros(n);rd=np.zeros(n)
        mae=np.full(n,np.nan);bias=np.full(n,np.nan);coverage=np.full(n,np.nan)
        cal_lo=np.zeros(n);cal_hi=np.zeros(n);cal_covered=np.zeros(n,dtype=bool)
        pos=neg=0.;res_history=[]
        for t in range(n):
            base_radius=max((q90[t]-q10[t])/2*1.282,1e-6)
            prior_abs=np.asarray(ae[max(0,t-window):t],float)
            radius=float(np.quantile(prior_abs,target_coverage,method="higher")) if len(prior_abs)>=min_history else base_radius
            cal_lo[t]=max(0.,q50[t]-radius);cal_hi[t]=q50[t]+radius
            cal_covered[t]=(y[t]>=cal_lo[t]) and (y[t]<=cal_hi[t])
            if t:
                start=max(0,t-window)
                pm=float(np.mean(ae[start:t]));pb=float(np.mean(err[start:t]))
                pc=float(np.mean(cal_covered[start:t]))
                scale=max(float(np.median(np.abs(y[start:t]))),float(np.mean(ae[start:t])),1e-6)
                mae[t]=pm;bias[t]=pb;coverage[t]=pc
                re[t]=np.clip(pm/scale,0,1)
                rb[t]=np.clip(2*(expit(2*abs(pb)/scale)-.5),0,1)
                rc[t]=np.clip(abs(pc-target_coverage)/max(1-target_coverage,1e-6),0,1)
                sigma=max(float(np.std(res_history[-window:])) if len(res_history)>1 else 1.,1e-6)
                z=err[t-1]/sigma
                pos=max(0.,pos+z-.5);neg=min(0.,neg+z+.5)
                rd[t]=float(pos>4 or abs(neg)>4)
            res_history.append(err[t])
        g["CAL_LOWER_90"]=cal_lo;g["CAL_UPPER_90"]=cal_hi
        g["CAL_COVERED_90"]=cal_covered.astype("int8")
        g["R_E"]=re;g["R_B"]=rb;g["R_C"]=rc;g["R_D"]=rd
        g["R_UNRELIABILITY"]=np.clip(w["error"]*re+w["bias"]*rb+w["coverage"]*rc+w["drift"]*rd,0,1)
        g["R_REL"]=g["R_UNRELIABILITY"]  # Legacy alias: higher means less reliable.
        g["ROLLING_MAE_PRIOR"]=mae;g["ROLLING_BIAS_PRIOR"]=bias;g["ROLLING_COVERAGE_PRIOR"]=coverage
        shift=np.zeros(n,dtype=bool)
        for t in range(2*window,n):
            a=err[t-2*window:t-window];b=err[t-window:t]
            scale=max(float(np.std(np.r_[a,b])),1e-6)
            shift[t]=abs(float(np.mean(b)-np.mean(a)))>1.5*scale
        g["SHIFT_DETECTOR_FLAG"]=shift
        output.append(g)
    return pd.concat(output,ignore_index=True) if output else predictions.copy()
