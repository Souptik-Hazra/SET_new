"""Single orchestrator for the SCMD → Quantile LightGBM → RADCSR workflow."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .forecasting.model import train_and_predict
from .reliability.index import build_reliability_index
from .reliability.baselines import interval_baselines
from .risk.radcsr import radcsr_probability,radcsr_ablations,monotonicity_audit,KEYS,DISCLAIMER
from .risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities,brier_by_method,series_bootstrap_holm

def run_model_pipeline(series_path="data/processed/scmd_monthly_series.parquet",output_dir="output",processed_dir="data/processed",seed=42,n_boot=2000):
    out=Path(output_dir);proc=Path(processed_dir)
    (out/"tables").mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    series=pd.read_parquet(series_path)
    features,pred,forecast_metrics,forecast_report=train_and_predict(series)
    features.to_parquet(proc/"features.parquet",index=False);pred.to_parquet(proc/"forecasts.parquet",index=False)
    forecast_metrics.to_csv(out/"tables/forecast_metrics.csv",index=False)
    if pred.empty:
        raise RuntimeError("No test predictions were generated; check cohort size and chronological split.")
    reliability=build_reliability_index(pred)
    reliability.to_parquet(proc/"reliability.parquet",index=False)
    intervals=interval_baselines(pred)
    intervals.to_parquet(proc/"reliability_baselines.parquet",index=False)
    signal_corr=reliability[["R_E","R_B","R_C","R_D"]].corr()
    signal_corr.to_csv(out/"tables/reliability_signal_correlations.csv")
    high_corr=[]
    for i,a in enumerate(signal_corr.columns):
        for b in signal_corr.columns[i+1:]:
            v=float(signal_corr.loc[a,b])
            if np.isfinite(v) and abs(v)>.70:high_corr.append({"signal_a":a,"signal_b":b,"correlation":v})
    scenarios=make_scenarios(reliability,seed)
    # Recompute reliability under each stress scenario; this is essential for S5_Bias.
    scenario_rel=[]
    for _,g in scenarios.groupby("SCENARIO_ID",sort=False):
        scenario_rel.append(build_reliability_index(g.assign(MODEL_DEMAND=g.SCENARIO_DEMAND)))
    scenarios=pd.concat(scenario_rel,ignore_index=True)
    inventory=simulate_inventory(scenarios)
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    joined=scenarios.merge(inventory[keys+["AVAILABLE_STOCK","STOCKOUT_LABEL","DEMAND"]],on=keys,how="inner",validate="one_to_one")
    rad=[]
    for _,g in joined.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        r=radcsr_probability(g.Q10.to_numpy(),g.Q50.to_numpy(),g.Q90.to_numpy(),
          g.AVAILABLE_STOCK.to_numpy(),g.LEAD_TIME_MONTHS.to_numpy(),g.R_REL.to_numpy())
        z=g[keys+["STOCKOUT_LABEL","AVAILABLE_STOCK","LEAD_TIME_MONTHS"]].reset_index(drop=True)
        rad.append(pd.concat([z,r],axis=1))\n        ab=radcsr_ablations(g).reset_index(drop=True)\n        ablation_rows.append(pd.concat([g[keys].reset_index(drop=True),ab],axis=1))
    rad_df=pd.concat(rad,ignore_index=True) if rad else pd.DataFrame()
    rad_df["METHOD"]="RADCSR"
    base=risk_baseline_probabilities(joined)
    rad_prob=rad_df[keys+["METHOD","RADCSR_PROBABILITY"]].rename(columns={"RADCSR_PROBABILITY":"SHORTAGE_PROBABILITY"})
    probabilities=pd.concat([base,rad_prob,ablation_df],ignore_index=True)
    probabilities.to_parquet(proc/"risk_probabilities.parquet",index=False)
    inventory.to_parquet(proc/"simulated_inventory.parquet",index=False)
    rad_df.to_parquet(proc/"radcsr_risk_details.parquet",index=False)
    scores=brier_by_method(probabilities,inventory);scores.to_csv(out/"tables/risk_brier_by_scenario.csv",index=False)
    stats=series_bootstrap_holm(probabilities,inventory,n_boot,seed);stats.to_csv(out/"tables/radcsr_vs_baselines_bootstrap_holm.csv",index=False)
    # Monotonicity audit on a fixed representative test case (risk probability, not an alert).
    sample=joined.iloc[0]
    mono=monotonicity_audit(float(sample.Q10),float(sample.Q50),float(sample.Q90),
      float(sample.AVAILABLE_STOCK),float(sample.LEAD_TIME_MONTHS))
    (out/"radcsr_monotonicity_audit.json").write_text(json.dumps(mono,indent=2))
    report={"forecast":forecast_report,"prediction_rows":len(pred),"reliability_rows":len(reliability),
      "reliability_baseline_rows":len(intervals),"scenario_rows":len(scenarios),"ablation_rows":len(ablation_df),"inventory_rows":len(inventory),
      "risk_probability_rows":len(probabilities),"signal_correlations_above_0_70":high_corr,
      "monotonicity_audit":{k:v for k,v in mono.items() if k!="probability_curve"},
      "bootstrap_resamples":n_boot,"disclaimer":DISCLAIMER,
      "clinical_priority":"Not computed: no verified clinical VEN mapping supplied.",\n      "mean_reliability_by_scenario":scenarios.groupby("SCENARIO_ID").R_REL.mean().to_dict(),
      "interpretation":"Brier scores compare probabilities with synthetic stockout labels only; they are not evidence of observed hospital stockout prediction.",
      "limitations":"SCMD has no observed stock, orders, lead times or stockout events. Simulated inventory and scenario assumptions must be sensitivity-tested."}
    (out/"pipeline_report.json").write_text(json.dumps(report,indent=2,default=str))
    return report
