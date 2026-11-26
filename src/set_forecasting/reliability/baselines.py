"""Static, rolling-quantile, EWMA, ACI, DtACI, AdaPI and EnbPI intervals."""
import numpy as np
import pandas as pd
KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def interval_baselines(predictions,target_coverage=.90,window=8):
    rows=[];alpha=1-target_coverage
    for _,g in predictions.groupby(KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy();y=g.MODEL_DEMAND.to_numpy(float)
        q50=g.Q50.to_numpy(float);q10=g.Q10.to_numpy(float);q90=g.Q90.to_numpy(float)
        errors=np.abs(y-q50);history=[];a_aci=a_dt=alpha
        state_ewma=0.
        methods={m:[] for m in ["Static","RollingQuantile","EWMA","ACI","DtACI","AdaPI","EnbPI"]}
        for i in range(len(g)):
            base=max((q90[i]-q10[i])/2*1.282,1e-6);prior=np.asarray(history[-window:],float)
            rq=float(np.quantile(prior,target_coverage,method="higher")) if len(prior)>=2 else base
            ew=float(pd.Series(prior).ewm(alpha=.25,adjust=False).mean().iloc[-1]*1.645) if len(prior) else base
            enb=float(np.quantile(prior,target_coverage,method="higher")) if len(prior)>=3 else base
            radii={"Static":base,"RollingQuantile":max(base,rq),"EWMA":max(base,ew),
              "ACI":base*max(.5,1+alpha-a_aci),"DtACI":base*max(.5,1+alpha-a_dt),
              "AdaPI":base*np.clip(rq/max(base,1e-6),.5,3),"EnbPI":max(base,enb)}
            for name,r in radii.items():methods[name].append((max(0,q50[i]-r),q50[i]+r))
            miss=float(y[i]<q50[i]-base or y[i]>q50[i]+base)
            a_aci=float(np.clip(a_aci+.05*(alpha-miss),.01,.5))
            a_dt=float(np.clip(a_dt+(.02/(1+i/50))*(alpha-miss),.01,.5))
            history.append(float(errors[i]))
        for name,pairs in methods.items():
            lo=np.array([p[0] for p in pairs]);hi=np.array([p[1] for p in pairs])
            rows.append(g[KEYS+["PERIOD"]].assign(SPLIT=g.SPLIT.to_numpy() if "SPLIT" in g else "test",METHOD=name,LOWER=lo,UPPER=hi,
              COVERED=((y>=lo)&(y<=hi)).astype(int),WIDTH=hi-lo))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
