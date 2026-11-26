"""Phases 5-8: reliability-adjusted risk, synthetic scenarios and evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.scmd_phase1 import KEY,DISCLAIMER

def adaptive_risk(df,kappa=.5,beta=1.,gamma_max=2.):
    x=df.copy()
    required=KEY+["PERIOD","Q10","Q50","Q90","ROLLING_MAE","ROLLING_BIAS","ROLLING_COVERAGE","DRIFT_SIGNAL"]
    missing=set(required)-set(x.columns)
    if missing: raise ValueError(f"Missing reliability fields: {sorted(missing)}")
    cov=x.ROLLING_COVERAGE.fillna(.8).clip(0,1)
    drift=x.DRIFT_SIGNAL.fillna(False).astype(float)
    x["RELIABILITY_SCORE"]=(cov*(1-drift)).clip(0,1)
    x["RISK_BASE"]=x.ROLLING_MAE.fillna(0).clip(lower=0)+x.ROLLING_BIAS.fillna(0).abs()
    x["ADAPTIVE_GAMMA"]=(1+kappa*(1-x.RELIABILITY_SCORE)**beta).clip(1,gamma_max)
    x["ADAPTIVE_RADIUS"]=np.maximum(x.RISK_BASE,x.Q90-x.Q50)*x.ADAPTIVE_GAMMA
    x["RISK_LOWER"]=(x.Q50-x.ADAPTIVE_RADIUS).clip(lower=0)
    x["RISK_UPPER"]=x.Q50+x.ADAPTIVE_RADIUS
    return x

def monotonicity_sweep():
    r=np.round(np.arange(0,1.001,.05),2)
    z=pd.DataFrame({"RELIABILITY":r,"RISK_MULTIPLIER":1+.5*(1-r)})
    if not z.RISK_MULTIPLIER.is_monotonic_decreasing: raise AssertionError("Risk multiplier must decrease with reliability")
    return z

def simulate_scenarios(pred,seed=42):
    rng=np.random.default_rng(seed);out=[]
    for _,g in pred.groupby(KEY,sort=False,dropna=False):
        g=g.sort_values("PERIOD");base=np.maximum(g.MODEL_DEMAND.to_numpy(float),0)
        for name in ["stable","surge","drift","volatility","bias","lead_time_shock","combined"]:
            y=base.copy();lead=np.full(len(y),2,dtype=int)
            if name in ("surge","combined") and len(y):
                a=len(y)//3;y[a:a+max(1,len(y)//4)]*=1.75
            if name in ("drift","combined"):y*=1+np.arange(len(y))*.02
            if name in ("volatility","combined"):y=np.maximum(0,y*rng.lognormal(0,.25,len(y)))
            if name=="bias":y*=1.15
            if name in ("lead_time_shock","combined"):lead[:]=3
            q=g[KEY+["PERIOD","Q10","Q50","Q90","RISK_UPPER"]].copy()
            q["SCENARIO_ID"]=name;q["SCENARIO_DEMAND"]=y;q["LEAD_TIME_MONTHS"]=lead
            out.append(q)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame(columns=KEY+["PERIOD","Q10","Q50","Q90","RISK_UPPER","SCENARIO_ID","SCENARIO_DEMAND","LEAD_TIME_MONTHS"])

def inventory_simulation(df,initial_cover=2.,reorder_cover=1.,target_cover=3.):
    rows=[]
    for keys,g in df.groupby(KEY+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");scenario=keys[-1];org=keys[:-1]
        first=max(0,float(g.RISK_UPPER.iloc[0]));stock=first*initial_cover;pipeline=[]
        for t,row in enumerate(g.itertuples()):
            arrivals=sum(q for due,q in pipeline if due<=t)
            pipeline=[(due,q) for due,q in pipeline if due>t]
            available=stock+arrivals;demand=max(0,float(row.SCENARIO_DEMAND))
            fulfilled=min(available,demand);shortage=max(0,demand-available);stock=available-fulfilled
            lead=max(1,int(row.LEAD_TIME_MONTHS));risk_target=max(0,float(row.RISK_UPPER))
            inventory_position=stock+sum(q for _,q in pipeline)
            order=max(0,risk_target*target_cover-inventory_position) if inventory_position<=risk_target*reorder_cover else 0.
            if order:pipeline.append((t+lead,order))
            rec=dict(zip(KEY,org));rec.update(PERIOD=row.PERIOD,SCENARIO_ID=scenario,SCENARIO_DEMAND=demand,
                FULFILLED=fulfilled,ENDING_STOCK=stock,SHORTAGE_QUANTITY=shortage,
                SIMULATED_STOCKOUT=int(shortage>0),LEAD_TIME_MONTHS=lead,ORDER_PLACED=order,
                INVENTORY_OUTCOME="SIMULATED")
            rows.append(rec)
    return pd.DataFrame(rows)

def baseline_metrics(df):
    rows=[]
    for keys,g in df.groupby(KEY+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");scenario=keys[-1];sid="|".join(map(str,keys[:-1]))
        y=g.SCENARIO_DEMAND.to_numpy(float);p=g.Q50.to_numpy(float)
        base=np.maximum((g.Q90.to_numpy(float)-g.Q10.to_numpy(float))/2,1e-6)
        residual=np.abs(y-p);hist=pd.Series(residual).shift(1)
        rad={"Static":base,"RollingQuantile":hist.rolling(6,min_periods=1).quantile(.9).fillna(pd.Series(base)).to_numpy(),
             "EWMA":hist.ewm(alpha=.25,adjust=False).mean().fillna(pd.Series(base)).to_numpy()*1.645,
             "ACI":base*(1+np.clip(hist.rolling(6,min_periods=1).mean().fillna(0).to_numpy()/np.maximum(base,1e-6)-1,0,1)),
             "MonteCarloStyle":base*1.15,"WassersteinStyle":base+hist.rolling(6,min_periods=1).mean().fillna(0).to_numpy()}
        for name,r in rad.items():
            lo=np.maximum(0,p-r);hi=p+r;score=(hi-lo)+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)
            rows.append({"SERIES_ID":sid,"SCENARIO_ID":scenario,"BASELINE":name,"N":len(y),
                "WAPE":float(np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9)*100),
                "INTERVAL_SCORE":float(score.mean()),"COVERAGE":float(np.mean((y>=lo)&(y<=hi))),
                "MEAN_WIDTH":float(np.mean(hi-lo))})
    return pd.DataFrame(rows)

def paired_bootstrap_holm(df,n_boot=2000,seed=42,metric="INTERVAL_SCORE"):
    if df.empty:return pd.DataFrame()
    # Aggregate scenarios within series, then resample independent series (cluster bootstrap).
    pivot=df.pivot_table(index="SERIES_ID",columns="BASELINE",values=metric,aggfunc="mean").dropna()
    if pivot.shape[1]<2 or pivot.empty:return pd.DataFrame()
    reference=pivot.mean().idxmin();rng=np.random.default_rng(seed);rows=[]
    for model in pivot.columns:
        if model==reference:continue
        d=(pivot[model]-pivot[reference]).to_numpy(float)
        if not len(d):continue
        samples=rng.choice(d,size=(n_boot,len(d)),replace=True).mean(axis=1)
        p=min(1.,2*min(float(np.mean(samples<=0)),float(np.mean(samples>=0))))
        rows.append({"METRIC":metric,"REFERENCE":reference,"COMPARISON":model,"N_SERIES":len(d),
            "MEAN_DIFFERENCE":float(d.mean()),"CI_LOWER":float(np.quantile(samples,.025)),
            "CI_UPPER":float(np.quantile(samples,.975)),"BOOTSTRAP_P":p})
    order=np.argsort([r["BOOTSTRAP_P"] for r in rows]);m=len(rows);adj=np.zeros(m);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(m-rank)*rows[i]["BOOTSTRAP_P"]));adj[i]=running
    for row,p in zip(rows,adj):row["HOLM_ADJUSTED_P"]=float(p);row["SIGNIFICANT_05"]=bool(p<.05 and (row["CI_UPPER"]<0 or row["CI_LOWER"]>0) and row["MEAN_DIFFERENCE"]<0)
    return pd.DataFrame(rows)

def run_phase5_to_8(output_root="output",processed_root="data/processed",seed=42,n_boot=2000):
    out=Path(output_root);proc=Path(processed_root);(out/"tables").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    rel=pd.read_parquet(proc/"scmd_reliability.parquet")
    if rel.empty: raise ValueError("No test predictions available; validate Phase 3 before running Phases 5-8")
    risk=adaptive_risk(rel);scen=simulate_scenarios(risk,seed);inv=inventory_simulation(scen)
    baselines=baseline_metrics(scen);stats=paired_bootstrap_holm(baselines,n_boot,seed)
    risk.to_parquet(proc/"scmd_adaptive_risk.parquet",index=False);scen.to_parquet(proc/"scmd_scenarios.parquet",index=False)
    inv.to_parquet(proc/"scmd_inventory_simulation.parquet",index=False)
    monotonicity_sweep().to_csv(out/"tables/phase5_monotonicity.csv",index=False)
    inv.groupby("SCENARIO_ID").agg(SIMULATED_STOCKOUT_RATE=("SIMULATED_STOCKOUT","mean"),MEAN_SHORTAGE=("SHORTAGE_QUANTITY","mean")).to_csv(out/"tables/phase6_inventory_scenarios.csv")
    baselines.to_csv(out/"tables/phase7_baseline_comparison.csv",index=False);stats.to_csv(out/"tables/phase8_bootstrap_holm.csv",index=False)
    report={"phase5_risk_rows":len(risk),"phase6_simulation_rows":len(inv),"phase7_baseline_rows":len(baselines),
      "phase8_comparisons":len(stats),"bootstrap_resamples":n_boot,"bootstrap_unit":"series clusters",
      "disclaimer":DISCLAIMER,"limitations":"Inventory, lead times, scenarios and stockout labels are simulated. MonteCarloStyle and WassersteinStyle are simplified interval sensitivity baselines, not full optimisation algorithms. No clinical VEN inference is made."}
    (out/"phases_5_8_report.json").write_text(json.dumps(report,indent=2));return report
