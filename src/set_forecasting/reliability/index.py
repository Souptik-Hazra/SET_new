"""Prequential four-signal RADCSR reliability index."""
import numpy as np
import pandas as pd
from scipy.special import expit
KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def build_reliability_index(predictions,target_coverage=.90,window=8,weights=None):
    required=KEYS+["PERIOD","MODEL_DEMAND","Q10","Q50","Q90"]
    missing=set(required)-set(predictions.columns)
    if missing:raise ValueError(f"Missing reliability inputs: {sorted(missing)}")
    w=weights or {"error":.35,"bias":.25,"coverage":.25,"drift":.15}
    if not np.isclose(sum(w.values()),1) or any(v<0 for v in w.values()):raise ValueError("Weights must be non-negative and sum to one")
    output=[]
    for _,g in predictions.groupby(KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy();y=g.MODEL_DEMAND.to_numpy(float)
        err=y-g.Q50.to_numpy(float);ae=np.abs(err)
        covered=((y>=g.Q10.to_numpy(float))&(y<=g.Q90.to_numpy(float))).astype(float)
        n=len(g);re=np.zeros(n);rb=np.zeros(n);rc=np.zeros(n);rd=np.zeros(n)
        mae=np.full(n,np.nan);bias=np.full(n,np.nan);coverage=np.full(n,np.nan)
        # CUSUM at t uses only residuals strictly before t.
        pos=neg=0.;res_hist=[]
        for t in range(n):
            start=max(0,t-window);prior=slice(start,t)
            if t:
                pm=float(np.mean(ae[prior]));pb=float(np.mean(err[prior]));pc=float(np.mean(covered[prior]))
                scale=max(float(np.median(np.abs(y[prior]))),float(np.mean(ae[prior])),1e-6)
                mae[t]=pm;bias[t]=pb;coverage[t]=pc
                re[t]=np.clip(pm/scale,0,1)
                rb[t]=np.clip(2*(expit(2*abs(pb)/scale)-.5),0,1)
                rc[t]=np.clip(abs(pc-target_coverage)/max(1-target_coverage,1e-6),0,1)
                sigma=max(float(np.std(res_hist[-window:])) if len(res_hist)>1 else 1.,1e-6)
                z=err[t-1]/sigma
                pos=max(0.,pos+z-.5);neg=min(0.,neg+z+.5)
                rd[t]=float(pos>4 or abs(neg)>4)
            res_hist.append(err[t])
        g["R_E"]=re;g["R_B"]=rb;g["R_C"]=rc;g["R_D"]=rd
        g["R_REL"]=np.clip(w["error"]*re+w["bias"]*rb+w["coverage"]*rc+w["drift"]*rd,0,1)
        g["ROLLING_MAE_PRIOR"]=mae;g["ROLLING_BIAS_PRIOR"]=bias;g["ROLLING_COVERAGE_PRIOR"]=coverage
        # A transparent two-window mean-shift detector, not ADWIN.
        shift=np.zeros(n,dtype=bool)
        for t in range(2*window,n):
            a=err[t-2*window:t-window];b=err[t-window:t]
            scale=max(float(np.std(np.r_[a,b])),1e-6)
            shift[t]=abs(float(np.mean(b)-np.mean(a)))>1.5*scale
        g["SHIFT_DETECTOR_FLAG"]=shift
        output.append(g)
    return pd.concat(output,ignore_index=True) if output else predictions.copy()
