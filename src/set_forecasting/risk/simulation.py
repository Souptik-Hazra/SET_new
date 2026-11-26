"""Seven synthetic scenarios, periodic-review inventory and comparable risk baselines."""
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from .radcsr import KEYS,infer_lead_time_distribution
SCENARIOS=["S1_Stable","S2_Surge","S3_Drift","S4_Volatility","S5_Bias","S6_LeadTimeShock","S7_Combined","S8_Intermittent","S9_CorrelatedSurge"]

def _stable_seed(seed, key):
    token = "|".join(map(str, key)).encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return (int(seed) + int.from_bytes(digest[:4], "big")) % (2**32)


def make_scenarios(predictions, seed=42):
    """Create reproducible, series-isolated stress streams with explicit shift metadata."""
    frames = []
    for key, group in predictions.groupby(KEYS, sort=False, dropna=False):
        group = group.sort_values("PERIOD").copy()
        base = np.maximum(group.MODEL_DEMAND.to_numpy(float), 0)
        start = len(base) // 3
        duration = max(1, len(base) // 3)
        for name in SCENARIOS:
            rng = np.random.default_rng(_stable_seed(seed, (*key, name)))
            x = group.copy()
            demand = base.copy()
            lead = np.full(len(x), 2, dtype=int)
            magnitude = 0.0
            if name in ("S2_Surge", "S7_Combined"):
                demand[start:start + duration] *= 1.5
                magnitude = 0.5
            if name in ("S3_Drift", "S7_Combined"):
                demand *= 1 + .02 * np.arange(len(demand))
                magnitude = .02
            if name in ("S4_Volatility", "S7_Combined"):
                demand = np.maximum(0, demand * rng.lognormal(0, .35, len(demand)))
                magnitude = .35
            if name == "S5_Bias":
                for col in ["Q10", "Q50", "Q90"]:
                    x[col] = np.maximum(0, x[col].to_numpy(float) * 1.2)
                magnitude = .2
            if name in ("S6_LeadTimeShock", "S7_Combined"):
                lead += 2
                magnitude = 2
            if name == "S8_Intermittent":
                keep = rng.random(len(demand)) < .5
                demand = np.where(keep, demand * 2.0, 0.0)
                magnitude = .5
            if name == "S9_CorrelatedSurge":
                innovations = rng.normal(0, .12, len(demand))
                ar = np.zeros(len(demand))
                for i in range(1, len(ar)):
                    ar[i] = .8 * ar[i - 1] + innovations[i]
                demand = demand * np.exp(np.clip(ar, -1.0, 1.0))
                demand[start:start + duration] *= 1.5
                magnitude = .5
            x["SCENARIO_ID"] = name
            x["SCENARIO_DEMAND"] = demand
            x["LEAD_TIME_MONTHS"] = lead
            x["SHIFT_START_INDEX"] = start if name != "S1_Stable" else -1
            x["SHIFT_MAGNITUDE"] = magnitude
            x["STOCKOUTS_ARE_OBSERVED"] = 0
            x["OUTCOME_TYPE"] = "SIMULATED"
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def simulate_inventory(scenarios,service_level=.95,initial_cover=2.,target_cover=3.):
    """Periodic review: receive → serve → measure lead-time shortage → reorder.

    AVAILABLE_STOCK includes on-hand stock plus pipeline receipts due within the
    simulated lead-time horizon. STOCKOUT_LABEL compares realised future lead-time
    demand with that availability; it is a retrospective synthetic label.
    """
    rows=[];z=float(norm.ppf(service_level))
    for keys,g in scenarios.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");demand_values=np.maximum(g.SCENARIO_DEMAND.to_numpy(float),0)
        leads=np.maximum(g.LEAD_TIME_MONTHS.to_numpy(int),1)
        stock=max(0,float(g.Q90.iloc[0])*initial_cover);pipeline=[]
        for t,row in enumerate(g.itertuples()):
            receipts=sum(q for due,q in pipeline if due<=t)
            pipeline=[(due,q) for due,q in pipeline if due>t]
            on_hand=stock+receipts
            lead=int(leads[t])
            pipeline_available=sum(q for due,q in pipeline if due<=t+lead)
            available=on_hand+pipeline_available
            lead_end=min(len(g),t+lead)
            realised_lead_demand=float(demand_values[t:lead_end].sum())
            current_demand=float(demand_values[t])
            shortage=max(0,current_demand-on_hand)
            stock=max(0,on_hand-current_demand)
            order_up_to=max(0,float(row.Q90)*target_cover+z*max(0,float(row.Q90-row.Q10))*np.sqrt(lead))
            position=stock+sum(q for _,q in pipeline)
            order=max(0,order_up_to-position)
            if order>0:pipeline.append((t+lead,order))
            rec=dict(zip(KEYS+["SCENARIO_ID"],keys))
            rec.update(PERIOD=row.PERIOD,DEMAND=current_demand,
                LEAD_TIME_DEMAND=realised_lead_demand,AVAILABLE_STOCK=available,
                ON_HAND_AVAILABLE=on_hand,ENDING_STOCK=stock,RECEIPTS=receipts,
                PIPELINE_WITHIN_LEAD_TIME=pipeline_available,ORDER_PLACED=order,
                LEAD_TIME_MONTHS=lead,SHORTAGE_QUANTITY=shortage,
                MONTHLY_STOCKOUT_LABEL=int(shortage>0),
                STOCKOUT_LABEL=int(realised_lead_demand>available),
                STOCKOUTS_ARE_OBSERVED=0,OUTCOME_TYPE="SIMULATED")
            rows.append(rec)
    return pd.DataFrame(rows)

def _wasserstein_tail(samples,threshold,epsilon):
    """Empirical W1 worst-case tail probability via optimal fractional mass transport."""
    x=np.asarray(samples,float);n=len(x)
    if not n:return np.nan
    base=float(np.mean(x>threshold));distances=np.sort(np.maximum(threshold-x[x<=threshold],0))
    budget=max(0.,float(epsilon))*n;moved=0.
    for d in distances:
        if d<=1e-12:moved+=1.;continue
        take=min(1.,budget/d);moved+=take;budget-=take*d
        if budget<=1e-12:break
    return float(np.clip(base+moved/n,0,1))

def risk_baseline_probabilities(frame, seed=42):
    """Static, Rolling Quantile, EWMA, ACI, Monte Carlo and W1-DRO risk methods."""
    rows=[]
    for _,g in frame.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD");q10=g.Q10.to_numpy(float);q50=g.Q50.to_numpy(float);q90=g.Q90.to_numpy(float)
        avail=g.AVAILABLE_STOCK.to_numpy(float);lead=g.LEAD_TIME_MONTHS.to_numpy(float)
        mu,sigma=infer_lead_time_distribution(q10,q50,q90,lead)
        residual=g.DEMAND.to_numpy(float)-q50;rng=np.random.default_rng(_stable_seed(seed, (*_, "risk_baselines")))
        for i in range(len(g)):
            hist=residual[max(0,i-8):i];base_sigma=max(float(sigma[i]),1e-6)
            samples=mu[i]+hist*np.sqrt(lead[i]) if len(hist) else np.array([mu[i]])
            p_static=float(1-norm.cdf((avail[i]-mu[i])/base_sigma))
            p_roll=float(np.mean(samples>avail[i]))
            ewma_scale=float(pd.Series(np.abs(hist)).ewm(alpha=.25,adjust=False).mean().iloc[-1]*1.645) if len(hist) else base_sigma
            p_ewma=float(1-norm.cdf((avail[i]-mu[i])/max(ewma_scale*np.sqrt(lead[i]),1e-6)))
            conformal=float(np.quantile(np.abs(hist),.90,method="higher")) if len(hist)>=3 else base_sigma/np.sqrt(lead[i])
            p_aci=float(1-norm.cdf((avail[i]-mu[i])/max(conformal*np.sqrt(lead[i])/1.645,1e-6)))
            p_mc=float(np.mean(rng.normal(mu[i],base_sigma,2000)>avail[i]))
            p_dro=_wasserstein_tail(samples,avail[i],.05*max(float(np.std(samples)),1e-6))
            ids=[g.iloc[i][k] for k in KEYS+["PERIOD","SCENARIO_ID"]]
            for name,p in {"Static":p_static,"RollingQuantile":p_roll,"EWMA":p_ewma,
                           "ACI":p_aci,"MonteCarlo":p_mc,"WassersteinDRO":p_dro}.items():
                rec=dict(zip(KEYS+["PERIOD","SCENARIO_ID"],ids));rec.update(METHOD=name,SHORTAGE_PROBABILITY=float(np.clip(p,0,1)),SPLIT=g.iloc[i].get("SPLIT","test"));rows.append(rec)
    return pd.DataFrame(rows)

def brier_by_method(probabilities,inventory):
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner",validate="many_to_one")
    x["BRIER"]=(x.SHORTAGE_PROBABILITY-x.STOCKOUT_LABEL)**2
    return x.groupby(["SCENARIO_ID","METHOD"],as_index=False).agg(BRIER_SCORE=("BRIER","mean"),N=("BRIER","size"))

def series_bootstrap_holm(probabilities,inventory,n_boot=2000,seed=42):
    keys=KEYS+["PERIOD","SCENARIO_ID"];x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner")
    x["SERIES_ID"]=x[KEYS].astype(str).agg("|".join,axis=1);x["LOSS"]=(x.SHORTAGE_PROBABILITY-x.STOCKOUT_LABEL)**2
    rng=np.random.default_rng(seed);results=[]
    for scenario,g in x.groupby("SCENARIO_ID"):
        p=g.pivot_table(index="SERIES_ID",columns="METHOD",values="LOSS",aggfunc="mean").dropna()
        if "RADCSR" not in p:continue
        for method in [m for m in p.columns if m!="RADCSR"]:
            d=(p.RADCSR-p[method]).to_numpy()
            if not len(d):continue
            boot=rng.choice(d,(n_boot,len(d)),replace=True).mean(axis=1)
            pv=float(min(1,2*min(np.mean(boot<=0),np.mean(boot>=0))))
            results.append({"SCENARIO_ID":scenario,"COMPARISON":method,"N_SERIES":len(d),
                "MEAN_BRIER_DIFFERENCE":float(d.mean()),"CI_LOWER":float(np.quantile(boot,.025)),
                "CI_UPPER":float(np.quantile(boot,.975)),"P_VALUE":pv})
    if not results:return pd.DataFrame()
    order=np.argsort([r["P_VALUE"] for r in results]);running=0.;m=len(results)
    for rank,idx in enumerate(order):
        running=max(running,min(1.,results[idx]["P_VALUE"]*(m-rank)))
        results[idx]["HOLM_ADJUSTED_P"]=running
        results[idx]["SIGNIFICANT_WIN"]=bool(running<.05 and results[idx]["CI_UPPER"]<0)
        results[idx]["SIGNIFICANT_LOSS"]=bool(running<.05 and results[idx]["CI_LOWER"]>0)
    return pd.DataFrame(results)

def calibrate_risk_probabilities(probabilities,inventory,min_validation_rows=30):
    """Fit per-scenario/method isotonic calibrators on validation labels only."""
    from sklearn.isotonic import IsotonicRegression
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner",validate="many_to_one")
    x["RAW_SHORTAGE_PROBABILITY"]=x.SHORTAGE_PROBABILITY
    calibrated=[]
    for (scenario,method),g in x.groupby(["SCENARIO_ID","METHOD"],sort=False):
        train=g[g.SPLIT=="validation"] if "SPLIT" in g else g.iloc[0:0]
        test=g[g.SPLIT=="test"] if "SPLIT" in g else g
        if len(train)>=min_validation_rows and train.STOCKOUT_LABEL.nunique()==2:
            model=IsotonicRegression(y_min=0.,y_max=1.,out_of_bounds="clip")
            model.fit(train.SHORTAGE_PROBABILITY,train.STOCKOUT_LABEL)
            test=test.copy();test["SHORTAGE_PROBABILITY"]=model.predict(test.SHORTAGE_PROBABILITY)
            test["CALIBRATOR_STATUS"]="isotonic_validation"
        else:
            test=test.copy();test["CALIBRATOR_STATUS"]="identity_insufficient_validation"
        calibrated.append(test)
    return pd.concat(calibrated,ignore_index=True) if calibrated else x.iloc[0:0]
