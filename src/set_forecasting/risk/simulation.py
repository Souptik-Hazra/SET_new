"""Seven synthetic scenarios, periodic-review inventory, and risk baselines."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .radcsr import KEYS, infer_lead_time_distribution, DISCLAIMER

SCENARIOS=["S1_Stable","S2_Surge","S3_Drift","S4_Volatility","S5_Bias","S6_LeadTimeShock","S7_Combined"]

def make_scenarios(predictions,seed=42):
    rng=np.random.default_rng(seed);out=[]
    for _,g in predictions.groupby(KEYS,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy();base=np.maximum(g.MODEL_DEMAND.to_numpy(float),0)
        for name in SCENARIOS:
            demand=base.copy();q50=g.Q50.to_numpy(float).copy();lead=np.full(len(g),2,dtype=int)
            if name in ("S2_Surge","S7_Combined"):
                start=len(demand)//3;demand[start:start+max(1,len(demand)//3)]*=1.5
            if name in ("S3_Drift","S7_Combined"):demand*=1+.02*np.arange(len(demand))
            if name in ("S4_Volatility","S7_Combined"):demand=np.maximum(0,demand*rng.lognormal(0,.35,len(demand)))
            if name=="S5_Bias":\n                q50=np.maximum(0,q50*1.2);g["Q10"]=np.maximum(0,g.Q10.to_numpy(float)*1.2);g["Q90"]=np.maximum(0,g.Q90.to_numpy(float)*1.2)
            if name in ("S6_LeadTimeShock","S7_Combined"):lead+=2
            x=g.copy();x["SCENARIO_ID"]=name;x["SCENARIO_DEMAND"]=demand
            x["SCENARIO_Q50"]=q50;x["LEAD_TIME_MONTHS"]=lead;x["STOCKOUTS_ARE_OBSERVED"]=0
            x["OUTCOME_TYPE"]="SIMULATED";out.append(x)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def simulate_inventory(scenarios,service_level=.95,initial_cover=2.,target_cover=3.):
    """Receive orders → serve demand → compute shortage → reorder."""
    rows=[];z=float(norm.ppf(service_level))
    for keys,g in scenarios.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");stock=max(0,float(g.Q90.iloc[0])*initial_cover);pipeline=[]
        for t,row in enumerate(g.itertuples()):
            receipts=sum(q for due,q in pipeline if due<=t);pipeline=[(due,q) for due,q in pipeline if due>t]
            available=stock+receipts;demand=max(0,float(row.SCENARIO_DEMAND))
            shortage=max(0,demand-available);stock=max(0,available-demand)
            lead=max(1,int(row.LEAD_TIME_MONTHS))
            order_up_to=max(0,float(row.Q90)*target_cover+z*max(0,float(row.Q90-row.Q10))*np.sqrt(lead))
            position=stock+sum(q for _,q in pipeline);order=max(0,order_up_to-position)
            if order>0:pipeline.append((t+lead,order))
            rec=dict(zip(KEYS+["SCENARIO_ID"],keys));rec.update(PERIOD=row.PERIOD,DEMAND=demand,
              AVAILABLE_STOCK=available,ENDING_STOCK=stock,RECEIPTS=receipts,ORDER_PLACED=order,
              LEAD_TIME_MONTHS=lead,SHORTAGE_QUANTITY=shortage,STOCKOUT_LABEL=int(shortage>0),
              STOCKOUTS_ARE_OBSERVED=0,OUTCOME_TYPE="SIMULATED")
            rows.append(rec)
    return pd.DataFrame(rows)

def _w1_tail(samples,threshold,epsilon):
    """Exact empirical 1-Wasserstein tail upper bound via fractional mass transport."""
    x=np.asarray(samples,float);n=len(x)
    if not n:return np.nan
    base=float(np.mean(x>threshold));cost=np.sort(np.maximum(threshold-x[x<=threshold],0))
    budget=max(0,float(epsilon))*n;moved=0.
    for distance in cost:
        if distance<=1e-12:moved+=1.;continue
        take=min(1.,budget/distance);moved+=take;budget-=take*distance
        if budget<=1e-12:break
    return float(np.clip(base+moved/n,0,1))

def risk_baseline_probabilities(frame):
    """Quantiles + simulated availability in; shortage probability out."""
    rows=[]
    for _,g in frame.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");q10=g.Q10.to_numpy(float);q50=g.Q50.to_numpy(float);q90=g.Q90.to_numpy(float)
        avail=g.AVAILABLE_STOCK.to_numpy(float);lead=g.LEAD_TIME_MONTHS.to_numpy(float)
        mu,sigma=infer_lead_time_distribution(q10,q50,q90,lead)
        actual=g.DEMAND.to_numpy(float);residual=actual-q50
        rng=np.random.default_rng(42)
        for i in range(len(g)):
            hist=residual[max(0,i-8):i]
            # No current-period outcome is used to estimate current-period risk.
            scale=max(float(np.std(hist,ddof=1)) if len(hist)>1 else float(sigma[i]/np.sqrt(lead[i])),1e-6)
            static=float(1-norm.cdf((avail[i]-mu[i])/max(sigma[i],1e-6)))
            empirical=mu[i]+(hist*np.sqrt(lead[i]) if len(hist) else np.array([0.]))
            rolling=float(np.mean(empirical>avail[i]))
            ewma_scale=float(pd.Series(np.abs(hist)).ewm(alpha=.25,adjust=False).mean().iloc[-1]*1.645) if len(hist) else sigma[i]
            ewma=float(1-norm.cdf((avail[i]-mu[i])/max(ewma_scale*np.sqrt(lead[i]),1e-6)))
            conformal=float(np.quantile(np.abs(hist),.90,method="higher")) if len(hist)>=3 else sigma[i]/np.sqrt(lead[i])
            aci=float(1-norm.cdf((avail[i]-mu[i])/max(np.sqrt(lead[i])*max(conformal,1e-6)/1.645,1e-6)))
            samples=rng.normal(mu[i],sigma[i],2000);mc=float(np.mean(samples>avail[i]))
            dro=_w1_tail(empirical,avail[i],.05*max(float(np.std(empirical)),1e-6))
            methods={"Static":static,"RollingQuantile":rolling,"EWMA":ewma,"ACI":aci,"MonteCarlo":mc,"WassersteinDRO":dro}
            for name,p in methods.items():
                rec=dict(zip(KEYS+["PERIOD","SCENARIO_ID"],[g.iloc[i][k] for k in KEYS+["PERIOD","SCENARIO_ID"]]))
                rec.update(METHOD=name,SHORTAGE_PROBABILITY=float(np.clip(p,0,1)))
                rows.append(rec)
    return pd.DataFrame(rows)

def brier_by_method(probabilities,inventory):
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner",validate="many_to_one")
    x["BRIER"]=(x.SHORTAGE_PROBABILITY-x.STOCKOUT_LABEL)**2
    return x.groupby(["SCENARIO_ID","METHOD"],as_index=False).agg(BRIER_SCORE=("BRIER","mean"),N=("BRIER","size"))

def series_bootstrap_holm(probabilities,inventory,n_boot=2000,seed=42):
    keys=KEYS+["PERIOD","SCENARIO_ID"];x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner")
    x["SERIES_ID"]=x[KEYS].astype(str).agg("|".join,axis=1);x["LOSS"]=(x.SHORTAGE_PROBABILITY-x.STOCKOUT_LABEL)**2
    methods=list(x.METHOD.unique());rng=np.random.default_rng(seed);res=[]
    for scenario,g in x.groupby("SCENARIO_ID"):
        pivot=g.pivot_table(index="SERIES_ID",columns="METHOD",values="LOSS",aggfunc="mean").dropna()
        if "RADCSR" not in pivot:continue
        for method in methods:
            if method=="RADCSR" or method not in pivot:continue
            d=(pivot.RADCSR-pivot[method]).to_numpy()
            if not len(d):continue
            boot=rng.choice(d,(n_boot,len(d)),replace=True).mean(axis=1)
            p=float(min(1,2*min(np.mean(boot<=0),np.mean(boot>=0))))
            res.append({"SCENARIO_ID":scenario,"COMPARISON":method,"N_SERIES":len(d),
              "MEAN_BRIER_DIFFERENCE":float(d.mean()),"CI_LOWER":float(np.quantile(boot,.025)),
              "CI_UPPER":float(np.quantile(boot,.975)),"P_VALUE":p})
    if not res:return pd.DataFrame()
    order=np.argsort([r["P_VALUE"] for r in res]);running=0.;m=len(res)
    for rank,idx in enumerate(order):
        running=max(running,min(1.,res[idx]["P_VALUE"]*(m-rank)))
        res[idx]["HOLM_ADJUSTED_P"]=running
        res[idx]["SIGNIFICANT_WIN"]=bool(running<.05 and res[idx]["CI_UPPER"]<0)
        res[idx]["SIGNIFICANT_LOSS"]=bool(running<.05 and res[idx]["CI_LOWER"]>0)
    return pd.DataFrame(res)
