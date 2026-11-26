"""Nine synthetic scenarios, periodic-review inventory and comparable risk baselines."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .radcsr import KEYS,infer_lead_time_distribution
SCENARIOS=["S1_Stable","S2_Surge","S3_Drift","S4_Volatility","S5_Bias","S6_LeadTimeShock","S7_Combined","S8_Intermittent","S9_CorrelatedSurge"]

def _stable_seed(seed, key):
    token = "|".join(map(str, key)).encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return (int(seed) + int.from_bytes(digest[:4], "big")) % (2**32)


def make_scenarios(predictions, seed=42, lead_times=(1, 2, 3)):
    """Create reproducible, series-isolated synthetic stress streams."""
    required = KEYS + ["PERIOD", "MODEL_DEMAND", "Q10", "Q50", "Q90"]
    missing = set(required) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing scenario inputs: {sorted(missing)}")
    if predictions.duplicated(KEYS + ["PERIOD"]).any():
        raise ValueError("Scenario input must have one row per series and period")
    if predictions[required].isna().any().any():
        raise ValueError("Scenario inputs must not contain null keys, periods, demand or quantiles")
    raw_lead_times = tuple(lead_times)
    if not raw_lead_times or any(not np.isfinite(float(value)) or float(value) < 1 or not float(value).is_integer() for value in raw_lead_times):
        raise ValueError("lead_times must contain positive whole-month values")
    lead_times = tuple(sorted({int(value) for value in raw_lead_times}))
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
            base_lead = lead_times[_stable_seed(seed, (*key, "base_lead_time")) % len(lead_times)]
            lead_shock = 0
            perturbations = []
            if name in ("S2_Surge", "S7_Combined"):
                demand[start:start + duration] *= 1.5
                perturbations.append({"name": "surge", "start_index": start, "duration": duration, "multiplier": 1.5, "magnitude": 0.5})
            if name in ("S3_Drift", "S7_Combined"):
                demand *= 1 + .02 * np.arange(len(demand))
                perturbations.append({"name": "gradual_drift", "start_index": 0, "monthly_rate": .02, "magnitude": .02})
            if name in ("S4_Volatility", "S7_Combined"):
                demand = np.maximum(0, demand * rng.lognormal(0, .35, len(demand)))
                perturbations.append({"name": "lognormal_volatility", "start_index": 0, "sigma": .35, "magnitude": .35})
            if name == "S5_Bias":
                for col in ["Q10", "Q50", "Q90"]:
                    x[col] = np.maximum(0, x[col].to_numpy(float) * 1.2)
                perturbations.append({"name": "forecast_bias", "start_index": 0, "forecast_multiplier": 1.2, "magnitude": .2})
            if name in ("S6_LeadTimeShock", "S7_Combined"):
                lead_shock = 2
                perturbations.append({"name": "lead_time_shock", "start_index": 0, "additional_months": 2, "magnitude": 2.0})
            if name == "S8_Intermittent":
                keep = rng.random(len(demand)) < .5
                demand = np.where(keep, demand * 2.0, 0.0)
                perturbations.append({"name": "intermittency", "start_index": 0, "zero_probability": .5, "positive_multiplier": 2.0, "magnitude": .5})
            if name == "S9_CorrelatedSurge":
                innovations = rng.normal(0, .12, len(demand))
                ar = np.zeros(len(demand))
                for i in range(1, len(ar)):
                    ar[i] = .8 * ar[i - 1] + innovations[i]
                demand = demand * np.exp(np.clip(ar, -1.0, 1.0))
                demand[start:start + duration] *= 1.5
                perturbations.append({"name": "correlated_surge", "start_index": start, "duration": duration, "ar1": .8, "innovation_sigma": .12, "surge_multiplier": 1.5, "magnitude": .5})
            lead = np.full(len(x), base_lead + lead_shock, dtype=int)
            x["SCENARIO_ID"] = name
            x["SCENARIO_DEMAND"] = demand
            x["LEAD_TIME_BASE_MONTHS"] = base_lead
            x["LEAD_TIME_SHOCK_MONTHS"] = lead_shock
            x["LEAD_TIME_MONTHS"] = lead
            x["PERTURBATIONS"] = json.dumps(perturbations, separators=(",", ":"))
            x["SHIFT_START_INDEX"] = next((p["start_index"] for p in perturbations if p["name"] in ("surge", "correlated_surge")), 0 if perturbations else -1)
            x["SHIFT_MAGNITUDE"] = max([float(p.get("magnitude", 0.0)) for p in perturbations] + [0.0])
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
    required = KEYS + ["SCENARIO_ID", "PERIOD", "SCENARIO_DEMAND", "Q10", "Q90", "LEAD_TIME_MONTHS"]
    missing = set(required) - set(scenarios.columns)
    if missing:
        raise ValueError(f"Missing inventory inputs: {sorted(missing)}")
    if not 0 < float(service_level) < 1:
        raise ValueError("service_level must be strictly between 0 and 1")
    if not np.isfinite([initial_cover, target_cover]).all() or initial_cover < 0 or target_cover < 0:
        raise ValueError("Inventory cover values must be finite and non-negative")
    if scenarios.duplicated(KEYS + ["SCENARIO_ID", "PERIOD"]).any():
        raise ValueError("Inventory input must have one row per series, scenario and period")
    rows=[];z=float(norm.ppf(service_level))
    for keys,g in scenarios.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        g=g.sort_values("PERIOD")
        raw_leads=pd.to_numeric(g.LEAD_TIME_MONTHS,errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(raw_leads)) or np.any(raw_leads < 1) or np.any(raw_leads != np.floor(raw_leads)):
            raise ValueError("LEAD_TIME_MONTHS must contain finite positive whole-month values")
        demand_raw=pd.to_numeric(g.SCENARIO_DEMAND,errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(demand_raw)) or np.any(demand_raw < 0):
            raise ValueError("SCENARIO_DEMAND must contain finite non-negative values")
        quantiles=g[["Q10","Q90"]].to_numpy(float)
        if np.any(~np.isfinite(quantiles)) or np.any(quantiles < 0) or np.any(quantiles[:,0] > quantiles[:,1]):
            raise ValueError("Inventory quantiles must be finite and satisfy 0 <= Q10 <= Q90")
        leads=raw_leads.astype(int)
        demand_values=demand_raw
        stock=max(0,float(g.Q90.iloc[0])*initial_cover);pipeline=[]
        for t,row in enumerate(g.itertuples()):
            receipts=sum(q for due,q in pipeline if due<=t)
            pipeline=[(due,q) for due,q in pipeline if due>t]
            on_hand=stock+receipts
            lead=int(leads[t])
            # The lead-time window is t ... t+lead-1. A receipt due at t+lead
            # arrives after this demand horizon and cannot reduce its shortage.
            pipeline_available=sum(q for due,q in pipeline if due<=t+lead-1)
            available=on_hand+pipeline_available
            lead_end=min(len(g),t+lead)
            horizon_complete = (t + lead) <= len(g)
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
                LEAD_TIME_DEMAND=realised_lead_demand,LEAD_TIME_HORIZON_COMPLETE=int(horizon_complete),AVAILABLE_STOCK=available,
                ON_HAND_AVAILABLE=on_hand,ENDING_STOCK=stock,RECEIPTS=receipts,
                PIPELINE_WITHIN_LEAD_TIME=pipeline_available,ORDER_PLACED=order,
                LEAD_TIME_MONTHS=lead,SHORTAGE_QUANTITY=shortage,
                MONTHLY_STOCKOUT_LABEL=int(shortage>0),
                STOCKOUT_LABEL=int(realised_lead_demand>available) if horizon_complete else np.nan,
                STOCKOUTS_ARE_OBSERVED=0,OUTCOME_TYPE="SIMULATED")
            rows.append(rec)
    return pd.DataFrame(rows)

def inventory_sensitivity(scenarios, service_levels=(0.90, 0.95, 0.99),
                            initial_covers=(1.0, 2.0), target_covers=(2.0, 3.0),
                            lead_times=(1, 2, 3)):
    """Evaluate a predeclared inventory-assumption grid on identical scenarios."""
    raw_lead_times = tuple(lead_times)
    if not raw_lead_times or any(not np.isfinite(float(value)) or float(value) < 1 or not float(value).is_integer() for value in raw_lead_times):
        raise ValueError("lead_times must contain positive whole-month values")
    lead_times = tuple(sorted({int(value) for value in raw_lead_times}))
    rows = []
    for lead_time in lead_times:
      scenario_copy = scenarios.copy()
      shock = scenario_copy.get("LEAD_TIME_SHOCK_MONTHS", pd.Series(0, index=scenario_copy.index)).to_numpy(int)
      scenario_copy["LEAD_TIME_BASE_MONTHS"] = int(lead_time)
      scenario_copy["LEAD_TIME_MONTHS"] = int(lead_time) + shock
      for service in service_levels:
        for initial in initial_covers:
            for target in target_covers:
                inventory = simulate_inventory(
                    scenario_copy, service_level=float(service),
                    initial_cover=float(initial), target_cover=float(target))
                summary = inventory.groupby("SCENARIO_ID", as_index=False).agg(
                    SIMULATED_STOCKOUT_RATE=("STOCKOUT_LABEL", "mean"),
                    MONTHLY_SHORTAGE_RATE=("MONTHLY_STOCKOUT_LABEL", "mean"),
                    MEAN_SHORTAGE_QUANTITY=("SHORTAGE_QUANTITY", "mean"),
                    MEAN_AVAILABLE_STOCK=("AVAILABLE_STOCK", "mean"),
                    COMPLETE_HORIZON_ROWS=("LEAD_TIME_HORIZON_COMPLETE", "sum"))
                summary["BASE_LEAD_TIME_MONTHS"] = int(lead_time)
                summary["SERVICE_LEVEL"] = float(service)
                summary["INITIAL_COVER_MONTHS"] = float(initial)
                summary["TARGET_COVER_MONTHS"] = float(target)
                rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
    x=x.dropna(subset=["STOCKOUT_LABEL"])
    x["BRIER"]=(x.SHORTAGE_PROBABILITY-x.STOCKOUT_LABEL)**2
    return x.groupby(["SCENARIO_ID","METHOD"],as_index=False).agg(BRIER_SCORE=("BRIER","mean"),N=("BRIER","size"))

def calibration_metrics(probabilities, inventory, n_bins=10):
    """Return held-out calibration diagnostics against the supplied outcome labels."""
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2")
    keys = KEYS + ["PERIOD", "SCENARIO_ID"]
    x = probabilities.merge(inventory[keys + ["STOCKOUT_LABEL"]],
                            on=keys, how="inner", validate="many_to_one")
    x = x.dropna(subset=["STOCKOUT_LABEL", "SHORTAGE_PROBABILITY"]).copy()
    x["SHORTAGE_PROBABILITY"] = x.SHORTAGE_PROBABILITY.clip(1e-6, 1 - 1e-6)
    x["STOCKOUT_LABEL"] = x.STOCKOUT_LABEL.astype(float)
    x["BRIER"] = (x.SHORTAGE_PROBABILITY - x.STOCKOUT_LABEL) ** 2
    x["LOG_LOSS"] = -(x.STOCKOUT_LABEL * np.log(x.SHORTAGE_PROBABILITY) +
                      (1 - x.STOCKOUT_LABEL) * np.log(1 - x.SHORTAGE_PROBABILITY))
    x["PROBABILITY_BIN"] = pd.cut(x.SHORTAGE_PROBABILITY,
                                  bins=np.linspace(0, 1, int(n_bins) + 1),
                                  include_lowest=True, duplicates="drop")
    rows = []
    for (scenario, method), group in x.groupby(["SCENARIO_ID", "METHOD"], sort=True):
        ece = 0.0
        for _, bucket in group.groupby("PROBABILITY_BIN", observed=True):
            ece += len(bucket) / len(group) * abs(
                float(bucket.SHORTAGE_PROBABILITY.mean()) -
                float(bucket.STOCKOUT_LABEL.mean()))
        rows.append({
            "SCENARIO_ID": scenario, "METHOD": method, "N": len(group),
            "BRIER_SCORE": float(group.BRIER.mean()),
            "LOG_LOSS": float(group.LOG_LOSS.mean()),
            "EXPECTED_CALIBRATION_ERROR": float(ece),
            "MEAN_PREDICTED_PROBABILITY": float(group.SHORTAGE_PROBABILITY.mean()),
            "OBSERVED_EVENT_RATE": float(group.STOCKOUT_LABEL.mean()),
            "OUTCOME_TYPE": str(inventory.OUTCOME_TYPE.iloc[0]) if "OUTCOME_TYPE" in inventory and len(inventory) else "UNKNOWN",
        })
    return pd.DataFrame(rows)


def series_bootstrap_holm(probabilities, inventory, n_boot=2000, seed=42):
    """Paired cluster bootstrap CI and paired sign-flip test, with global Holm correction.

    The independent resampling unit is the organisation/VMP/unit series, not months.
    All scenario-method hypotheses are corrected together as one predeclared family.
    """
    keys = KEYS + ["PERIOD", "SCENARIO_ID"]
    x = probabilities.merge(inventory[keys + ["STOCKOUT_LABEL"]],
                            on=keys, how="inner", validate="many_to_one")
    x=x.dropna(subset=["STOCKOUT_LABEL"])
    x["SERIES_ID"] = x[KEYS].astype(str).agg("|".join, axis=1)
    x["LOSS"] = (x.SHORTAGE_PROBABILITY - x.STOCKOUT_LABEL) ** 2
    if int(n_boot) < 100:
        raise ValueError("n_boot must be at least 100 for stable resampling")
    rng = np.random.default_rng(seed)
    results = []
    for scenario, group in x.groupby("SCENARIO_ID", sort=True):
        pivot = group.pivot_table(index="SERIES_ID", columns="METHOD",
                                  values="LOSS", aggfunc="mean").dropna()
        if "RADCSR" not in pivot.columns:
            continue
        for method in [m for m in pivot.columns if m != "RADCSR"]:
            differences = (pivot["RADCSR"] - pivot[method]).to_numpy(float)
            n = len(differences)
            if n < 2:
                continue
            # Chunk resampling to bound peak memory when the full SCMD cohort has
            # tens or hundreds of thousands of independent series.
            boot = np.empty(int(n_boot), dtype=float)
            null_means = np.empty(int(n_boot), dtype=float)
            batch_size = max(1, min(64, int(2_000_000 / max(n, 1))))
            for start in range(0, int(n_boot), batch_size):
                stop = min(int(n_boot), start + batch_size)
                size = stop - start
                boot[start:stop] = rng.choice(differences, size=(size, n), replace=True).mean(axis=1)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
                null_means[start:stop] = (signs * differences).mean(axis=1)
            observed = float(differences.mean())
            p_value = float((1 + np.count_nonzero(np.abs(null_means) >= abs(observed))) /
                            (int(n_boot) + 1))
            results.append({
                "SCENARIO_ID": scenario, "COMPARISON": method, "N_SERIES": n,
                "MEAN_BRIER_DIFFERENCE": observed,
                "CI_LOWER": float(np.quantile(boot, .025)),
                "CI_UPPER": float(np.quantile(boot, .975)),
                "P_VALUE": p_value,
                "BOOTSTRAP_RESAMPLES": int(n_boot),
                "TEST": "paired_series_sign_flip",
                "INFERENCE_STATUS": "LOW_CLUSTER_COUNT_DESCRIPTIVE_ONLY" if n < 10 else "INFERENTIAL",
                "MINIMUM_SERIES_FOR_INFERENCE": 10
            })
    if not results:
        return pd.DataFrame()
    # Holm step-down adjusted p-values are monotone in sorted raw p-value order.
    order = np.argsort([row["P_VALUE"] for row in results])
    running = 0.0
    m = len(results)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * results[idx]["P_VALUE"]))
        results[idx]["HOLM_ADJUSTED_P"] = running
        enough_clusters = results[idx]["N_SERIES"] >= 10
        results[idx]["SIGNIFICANT_WIN"] = bool(
            enough_clusters and running < .05 and results[idx]["CI_UPPER"] < 0 and
            results[idx]["MEAN_BRIER_DIFFERENCE"] < 0)
        results[idx]["SIGNIFICANT_LOSS"] = bool(
            enough_clusters and running < .05 and results[idx]["CI_LOWER"] > 0 and
            results[idx]["MEAN_BRIER_DIFFERENCE"] > 0)
    return pd.DataFrame(results)

def calibrate_risk_probabilities(probabilities,inventory,min_validation_rows=30):
    """Fit per-scenario/method isotonic calibrators on validation labels only."""
    from sklearn.isotonic import IsotonicRegression
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    x=probabilities.merge(inventory[keys+["STOCKOUT_LABEL"]],on=keys,how="inner",validate="many_to_one")
    x=x.dropna(subset=["STOCKOUT_LABEL"])
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
