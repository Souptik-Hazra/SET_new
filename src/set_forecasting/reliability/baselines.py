"""Prequential reliability interval competitors.

DtACI dispatches among multiple adaptive-conformal instances. AdaPI adapts interval
width from recent empirical coverage. EnbPI uses past residual quantiles.
"""
import numpy as np
import pandas as pd
KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def interval_baselines(predictions,target_coverage=.90,window=8):
    rows=[];alpha_target=1-target_coverage
    for _,g in predictions.groupby(KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy();y=g.MODEL_DEMAND.to_numpy(float)
        q50=g.Q50.to_numpy(float);q10=g.Q10.to_numpy(float);q90=g.Q90.to_numpy(float)
        errors=np.abs(y-q50);history=[];alpha_aci=alpha_target;gamma=.05
        dt_alphas=np.array([.05,.10,.15],dtype=float)
        dt_misses=[[],[],[]];coverage_history=[]
        methods={m:[] for m in ["Static","RollingQuantile","EWMA","ACI","DtACI","AdaPI","EnbPI"]}
        for i in range(len(g)):
            base=max((q90[i]-q10[i])/2*1.282,1e-6)
            prior=np.asarray(history[-window:],float)
            rq=float(np.quantile(prior,target_coverage,method="higher")) if len(prior)>=2 else base
            ew=float(pd.Series(prior).ewm(alpha=.25,adjust=False).mean().iloc[-1]*1.645) if len(prior) else base
            a_radius=float(np.quantile(prior,1-alpha_aci,method="higher")) if len(prior)>=3 else base
            # Dispatch among three independent ACI controllers using their past coverage.
            dt_candidates=[]
            for j,a in enumerate(dt_alphas):
                radius=float(np.quantile(prior,1-a,method="higher")) if len(prior)>=3 else base
                recent=np.asarray(dt_misses[j][-window:],float)
                cov=1-float(np.mean(recent)) if len(recent) else target_coverage
                dt_candidates.append((abs(cov-target_coverage),radius,j))
            _,dt_radius,selected=min(dt_candidates,key=lambda z:(z[0],z[1]))
            # AdaPI scales its nominal width according to prior empirical coverage error.
            prior_cov=float(np.mean(coverage_history[-window:])) if coverage_history else target_coverage
            ada_radius=base*float(np.clip(1+(target_coverage-prior_cov)*2,.5,3.))
            enb=float(np.quantile(prior,target_coverage,method="higher")) if len(prior)>=3 else base
            radii={"Static":base,"RollingQuantile":max(base,rq),"EWMA":max(base,ew),
                "ACI":max(1e-6,a_radius),"DtACI":max(1e-6,dt_radius),
                "AdaPI":max(1e-6,ada_radius),"EnbPI":max(1e-6,enb)}
            for name,radius in radii.items():
                methods[name].append((max(0,q50[i]-radius),q50[i]+radius))
            # Only after intervals are fixed do we reveal this period's observation.
            misses=[]
            for j,a in enumerate(dt_alphas):
                r=dt_candidates[j][1]
                miss=float(y[i]<max(0,q50[i]-r) or y[i]>q50[i]+r)
                dt_misses[j].append(miss);misses.append(miss)
                dt_alphas[j]=np.clip(a+.03*(alpha_target-miss),.01,.50)
            a_miss=float(y[i]<max(0,q50[i]-a_radius) or y[i]>q50[i]+a_radius)
            alpha_aci=float(np.clip(alpha_aci+gamma*(alpha_target-a_miss),.01,.50))
            coverage_history.append(float(y[i]>=max(0,q50[i]-ada_radius) and y[i]<=q50[i]+ada_radius))
            history.append(float(errors[i]))
        for name,pairs in methods.items():
            lo=np.array([p[0] for p in pairs]);hi=np.array([p[1] for p in pairs])
            rows.append(g[KEYS+["PERIOD"]].assign(
                SPLIT=g.SPLIT.to_numpy() if "SPLIT" in g else "test",
                METHOD=name,LOWER=lo,UPPER=hi,
                COVERED=((y>=lo)&(y<=hi)).astype(int),WIDTH=hi-lo))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
