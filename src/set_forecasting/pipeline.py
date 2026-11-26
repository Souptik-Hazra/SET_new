"""Single orchestrator for the SCMD → Quantile LightGBM → RADCSR workflow."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .forecasting.model import train_and_predict
from .reliability.index import build_reliability_index
from .reliability.baselines import interval_baselines
from .risk.radcsr import radcsr_probability,radcsr_ablations,monotonicity_audit,KEYS,DISCLAIMER
from .risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities,brier_by_method,series_bootstrap_holm\nfrom .risk.clinical import load_verified_ven_mapping,compare_clinical_priority

def run_model_pipeline(series_path="data/processed/scmd_monthly_series.parquet",output_dir="output",processed_dir="data/processed",seed=42,n_boot=2000):
    out=Path(output_dir);proc=Path(processed_dir)
    (out/"tables").mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    series=pd.read_parquet(series_path)
    features,pred,forecast_metrics,forecast_report=train_and_predict(series)
    features.to_parquet(proc/"features.parquet",index=False);pred.to_parquet(proc/"forecasts.parquet",index=False)
    forecast_metrics.to_csv(out/"tables/forecast_metrics.csv",index=False)\n    if len(features):\n        per_series=[]\n        for key,g in pred.groupby(KEYS,sort=False,dropna=False):\n            y=g.MODEL_DEMAND.to_numpy(float);q=g.Q50.to_numpy(float)\n            per_series.append(dict(zip(KEYS,key),N_TEST=len(g),WAPE=float(np.abs(y-q).sum()/max(np.abs(y).sum(),1e-9)*100),MAE=float(np.mean(np.abs(y-q)))))\n        per_series=pd.DataFrame(per_series)\n        per_series.to_csv(out/"tables/per_series_forecast_metrics.csv",index=False)\n        outlier_count=int(features.get("OUTLIER_995_FLAG",pd.Series(dtype=bool)).fillna(False).sum())\n    else:\n        per_series=pd.DataFrame();outlier_count=0
    if pred.empty:raise RuntimeError("No test predictions were generated; check cohort size and chronological split.")
    reliability=build_reliability_index(pred)
    reliability.to_parquet(proc/"reliability.parquet",index=False)
    intervals=interval_baselines(pred);intervals.to_parquet(proc/"reliability_baselines.parquet",index=False)\n    interval_summary=intervals.groupby("METHOD",as_index=False).agg(PICP=("COVERED","mean"),MPIW=("WIDTH","mean"))\n    interval_summary.to_csv(out/"tables/reliability_baseline_metrics.csv",index=False)
    corr=reliability[["R_E","R_B","R_C","R_D"]].corr();corr.to_csv(out/"tables/reliability_signal_correlations.csv")
    high_corr=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            value=float(corr.loc[a,b])
            if np.isfinite(value) and abs(value)>.70:high_corr.append({"signal_a":a,"signal_b":b,"correlation":value})
    test_reliability=reliability[reliability.SPLIT=="test"].copy() if "SPLIT" in reliability else reliability.copy()\n    scenarios=make_scenarios(test_reliability,seed)
    scenario_rel=[]
    for _,g in scenarios.groupby("SCENARIO_ID",sort=False):
        scenario_rel.append(build_reliability_index(g.assign(MODEL_DEMAND=g.SCENARIO_DEMAND)))
    scenarios=pd.concat(scenario_rel,ignore_index=True)
    inventory=simulate_inventory(scenarios)
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    joined=scenarios.merge(inventory[keys+["AVAILABLE_STOCK","STOCKOUT_LABEL","DEMAND"]],
        on=keys,how="inner",validate="one_to_one")
    rad=[];ablation_rows=[]
    for _,g in joined.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        result=radcsr_probability(g.Q10.to_numpy(),g.Q50.to_numpy(),g.Q90.to_numpy(),
            g.AVAILABLE_STOCK.to_numpy(),g.LEAD_TIME_MONTHS.to_numpy(),g.R_REL.to_numpy())
        base=g[keys+["STOCKOUT_LABEL","AVAILABLE_STOCK","LEAD_TIME_MONTHS"]].reset_index(drop=True)
        rad.append(pd.concat([base,result],axis=1))
        ab=radcsr_ablations(g).reset_index(drop=True)
        ablation_rows.append(pd.concat([g[keys].reset_index(drop=True),ab],axis=1))
    rad_df=pd.concat(rad,ignore_index=True);rad_df["METHOD"]="RADCSR"
    ablation_df=pd.concat(ablation_rows,ignore_index=True)
    baseline=risk_baseline_probabilities(joined)
    rad_prob=rad_df[keys+["METHOD","RADCSR_PROBABILITY"]].rename(columns={"RADCSR_PROBABILITY":"SHORTAGE_PROBABILITY"})
    probabilities=pd.concat([baseline,rad_prob,ablation_df],ignore_index=True)
    probabilities.to_parquet(proc/"risk_probabilities.parquet",index=False)
    inventory.to_parquet(proc/"simulated_inventory.parquet",index=False)
    rad_df.to_parquet(proc/"radcsr_risk_details.parquet",index=False)
    scores=brier_by_method(probabilities,inventory);scores.to_csv(out/"tables/risk_brier_by_scenario.csv",index=False)
    stats=series_bootstrap_holm(probabilities,inventory,n_boot,seed)
    stats.to_csv(out/"tables/radcsr_vs_baselines_bootstrap_holm.csv",index=False)\n    if len(stats):\n        stats[(stats.SCENARIO_ID=="S5_Bias") & (stats.COMPARISON.isin(["Static","WassersteinDRO"]))].to_csv(out/"tables/s5_bias_radcsr_vs_static_wasserstein.csv",index=False)
    clinical_status="Not computed: no verified clinical VEN mapping supplied."\n    ven_path=Path("data/reference/clinical_ven_mapping.csv")\n    if ven_path.exists():\n        mapping=load_verified_ven_mapping(ven_path)\n        clinical_detail,clinical_summary=compare_clinical_priority(rad_df,mapping)\n        clinical_detail.to_parquet(proc/"clinical_priority.parquet",index=False)\n        clinical_summary.to_csv(out/"tables/clinical_priority_ranking_comparison.csv",index=False)\n        clinical_status={"mapping_file":str(ven_path),"mapped_medicines":int(mapping.VMP_SNOMED_CODE.nunique()),"ranking_changed":bool(clinical_summary.RANKING_CHANGED.any())}\n    sample=joined.iloc[0]
    mono=monotonicity_audit(float(sample.Q10),float(sample.Q50),float(sample.Q90),
        float(sample.AVAILABLE_STOCK),float(sample.LEAD_TIME_MONTHS))
    (out/"radcsr_monotonicity_audit.json").write_text(json.dumps(mono,indent=2))
    mean_rel=scenarios.groupby("SCENARIO_ID").R_REL.mean().to_dict()
    report={"forecast":forecast_report,"median_per_series_wape":float(per_series.WAPE.median()) if len(per_series) else None,"outlier_995_count":outlier_count,"prediction_rows":len(pred),"reliability_rows":len(reliability),
      "reliability_baseline_rows":len(intervals),"scenario_rows":len(scenarios),"ablation_rows":len(ablation_df),
      "inventory_rows":len(inventory),"risk_probability_rows":len(probabilities),
      "signal_correlations_above_0_70":high_corr,"mean_reliability_by_scenario":mean_rel,"stress_reliability_lift_vs_stable":{k:float(v-mean_rel.get("S1_Stable",0)) for k,v in mean_rel.items()},
      "monotonicity_audit":{k:v for k,v in mono.items() if k!="probability_curve"},
      "bootstrap_resamples":n_boot,"disclaimer":DISCLAIMER,
      "clinical_priority":clinical_status,
      "interpretation":"Brier scores compare probabilities with synthetic stockout labels only; they are not evidence of observed hospital stockout prediction.",
      "limitations":"SCMD has no observed stock, orders, lead times or stockout events. Simulated inventory and scenario assumptions must be sensitivity-tested."}
    (out/"pipeline_report.json").write_text(json.dumps(report,indent=2,default=str))
    return report
