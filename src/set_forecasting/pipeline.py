"""Single orchestrator for SCMD → quantile forecasting → RADCSR → calibrated risk."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .data.source import get_project_root
from .forecasting.model import train_and_predict
from .reliability.index import build_reliability_index
from .reliability.baselines import interval_baselines
from .risk.radcsr import radcsr_probability,radcsr_ablations,monotonicity_audit,KEYS,DISCLAIMER
from .risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities,brier_by_method,calibration_metrics,series_bootstrap_holm,calibrate_risk_probabilities
from .risk.clinical import load_verified_ven_mapping,compare_clinical_priority

def run_model_pipeline(series_path="data/processed/scmd_monthly_series.parquet",output_dir="output",processed_dir="data/processed",seed=42,n_boot=2000):
    root = get_project_root()
    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    series_path = project_path(series_path)
    out = project_path(output_dir)
    proc = project_path(processed_dir)
    config_path = root / "config" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    resolved_config = dict(cfg)
    resolved_config.setdefault("seed", seed)
    resolved_config.setdefault("evaluation", {})
    resolved_config["evaluation"]["bootstrap_resamples"] = n_boot
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, default=str), encoding="utf-8")
    rel_cfg=cfg.get("reliability",{});risk_cfg=cfg.get("radcsr",{});inv_cfg=cfg.get("inventory",{});data_cfg=cfg.get("data",{})
    (out/"tables").mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    series=pd.read_parquet(series_path)
    features,pred,forecast_metrics,forecast_report=train_and_predict(series,random_state=int(seed),winsor_quantile=float(data_cfg.get("winsor_quantile",.995)))
    features.to_parquet(proc/"features.parquet",index=False);pred.to_parquet(proc/"forecasts.parquet",index=False)
    forecast_metrics.to_csv(out/"tables/forecast_metrics.csv",index=False)
    test_pred=pred[pred.SPLIT=="test"].copy() if "SPLIT" in pred else pred
    per_series=[]
    for key,g in test_pred.groupby(KEYS,sort=False,dropna=False):
        y=g.MODEL_DEMAND.to_numpy(float);q=g.Q50.to_numpy(float)
        per_series.append(dict(zip(KEYS,key),N_TEST=len(g),
            WAPE=float(np.abs(y-q).sum()/max(np.abs(y).sum(),1e-9)*100),
            MAE=float(np.mean(np.abs(y-q)))))
    per_series=pd.DataFrame(per_series)
    per_series.to_csv(out/"tables/per_series_forecast_metrics.csv",index=False)
    outlier_count=int(features.get("OUTLIER_995_FLAG",pd.Series(dtype=bool)).fillna(False).sum())
    outlier_summary=features.groupby(KEYS,as_index=False).agg(WINSOR_LIMIT_995=("WINSOR_LIMIT_995","first"),OUTLIER_COUNT=("OUTLIER_995_FLAG","sum"),N_PERIODS=("PERIOD","size"))
    outlier_summary.to_csv(out/"tables/per_series_outlier_summary.csv",index=False)
    if pred.empty:raise RuntimeError("No validation/test predictions; check cohort size and chronological split.")
    reliability=build_reliability_index(pred,target_coverage=float(rel_cfg.get("target_coverage",.90)),window=int(rel_cfg.get("window",8)),weights=rel_cfg.get("weights"))
    reliability.to_parquet(proc/"reliability.parquet",index=False)
    intervals=interval_baselines(pred,target_coverage=float(rel_cfg.get("target_coverage",.90)),window=int(rel_cfg.get("window",8)));intervals.to_parquet(proc/"reliability_baselines.parquet",index=False)
    interval_test=intervals[intervals.SPLIT=="test"] if "SPLIT" in intervals else intervals
    interval_summary=interval_test.groupby("METHOD",as_index=False).agg(PICP=("COVERED","mean"),MPIW=("WIDTH","mean"))
    interval_summary.to_csv(out/"tables/reliability_baseline_metrics.csv",index=False)
    corr=reliability[["R_E","R_B","R_C","R_D"]].corr();corr.to_csv(out/"tables/reliability_signal_correlations.csv")
    high_corr=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            value=float(corr.loc[a,b])
            if np.isfinite(value) and abs(value)>.70:high_corr.append({"signal_a":a,"signal_b":b,"correlation":value})
    # Generate validation and test scenarios together so test reliability has a validation warm-up.
    scenarios=make_scenarios(reliability,seed)
    scenario_rel=[]
    for _,g in scenarios.groupby("SCENARIO_ID",sort=False):
        scenario_rel.append(build_reliability_index(g.assign(MODEL_DEMAND=g.SCENARIO_DEMAND),target_coverage=float(rel_cfg.get("target_coverage",.90)),window=int(rel_cfg.get("window",8)),weights=rel_cfg.get("weights")))
    scenarios=pd.concat(scenario_rel,ignore_index=True)
    scenarios.to_parquet(proc/"scenario_streams.parquet",index=False)
    inventory=simulate_inventory(scenarios,service_level=float(inv_cfg.get("service_level",.95)),
        initial_cover=float(inv_cfg.get("initial_cover_months",2)),target_cover=float(inv_cfg.get("target_cover_months",3)))
    keys=KEYS+["PERIOD","SCENARIO_ID"]
    joined=scenarios.merge(inventory[keys+["AVAILABLE_STOCK","STOCKOUT_LABEL","DEMAND"]],
        on=keys,how="inner",validate="one_to_one")
    rad=[];ablation_rows=[]
    for _,g in joined.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False):
        result=radcsr_probability(g.Q10.to_numpy(),g.Q50.to_numpy(),g.Q90.to_numpy(),
            g.AVAILABLE_STOCK.to_numpy(),g.LEAD_TIME_MONTHS.to_numpy(),g.R_REL.to_numpy(),
            kappa_sigma=float(risk_cfg.get("kappa_sigma",.75)),kappa_mu=float(risk_cfg.get("kappa_mu",.50)),
            kappa_l=float(risk_cfg.get("kappa_l",1.5)),
            horizon_cap=risk_cfg.get("horizon_cap",2.0))
        base=g[keys+["SPLIT","STOCKOUT_LABEL","AVAILABLE_STOCK","LEAD_TIME_MONTHS"]].reset_index(drop=True)
        rad.append(pd.concat([base,result],axis=1))
        ab=radcsr_ablations(g,
            kappa_sigma=float(risk_cfg.get("kappa_sigma",.75)),
            kappa_mu=float(risk_cfg.get("kappa_mu",.50)),
            kappa_l=float(risk_cfg.get("kappa_l",1.5)),
            horizon_cap=risk_cfg.get("horizon_cap",2.0),
            weights=rel_cfg.get("weights")).reset_index(drop=True)
        ablation_keys = pd.concat([g[keys+["SPLIT"]].reset_index(drop=True)] * ab.METHOD.nunique(), ignore_index=True)
        if len(ablation_keys) != len(ab):
            raise RuntimeError("Ablation key expansion does not match method output rows")
        ablation_rows.append(pd.concat([ablation_keys,ab.reset_index(drop=True)],axis=1))
    rad_df=pd.concat(rad,ignore_index=True);rad_df["METHOD"]="RADCSR"
    ablation_df=pd.concat(ablation_rows,ignore_index=True)
    ablation_df.to_parquet(proc/"radcsr_ablations.parquet",index=False)
    baseline=risk_baseline_probabilities(joined, seed=seed)
    rad_prob=rad_df[keys+["SPLIT","METHOD","RADCSR_PROBABILITY"]].rename(columns={"RADCSR_PROBABILITY":"SHORTAGE_PROBABILITY"})
    raw_probabilities=pd.concat([baseline,rad_prob,ablation_df],ignore_index=True)
    # Calibrate every risk method using validation-only synthetic labels; score only held-out test.
    calibrated=calibrate_risk_probabilities(raw_probabilities,inventory)
    calibrated.to_parquet(proc/"calibrated_risk_probabilities.parquet",index=False)
    raw_probabilities.to_parquet(proc/"raw_risk_probabilities.parquet",index=False)
    inventory.to_parquet(proc/"simulated_inventory.parquet",index=False)
    rad_df.to_parquet(proc/"radcsr_risk_details.parquet",index=False)
    eval_prob=calibrated.drop(columns=["STOCKOUT_LABEL"],errors="ignore")
    scores=brier_by_method(eval_prob,inventory);scores.to_csv(out/"tables/risk_brier_by_scenario.csv",index=False)
    calibration=calibration_metrics(eval_prob,inventory,n_bins=int(cfg.get("evaluation",{}).get("calibration_bins",10)))
    calibration.to_csv(out/"tables/risk_calibration_metrics.csv",index=False)
    ablation_test=calibrated[calibrated.METHOD.str.startswith("RADCSR")].copy()
    if len(ablation_test):
        ablation_test["BRIER"]=(ablation_test.SHORTAGE_PROBABILITY-ablation_test.STOCKOUT_LABEL)**2
        ablation_summary=ablation_test.groupby(["SCENARIO_ID","METHOD"],as_index=False).agg(
            BRIER_SCORE=("BRIER","mean"),N=("BRIER","size"))
    else:
        ablation_summary=pd.DataFrame(columns=["SCENARIO_ID","METHOD","BRIER_SCORE","N"])
    ablation_summary.to_csv(out/"tables/phase7_ablation_metrics.csv",index=False)
    inventory.groupby("SCENARIO_ID",as_index=False).agg(
        SIMULATED_STOCKOUT_RATE=("STOCKOUT_LABEL","mean"),
        MEAN_SHORTAGE_QUANTITY=("SHORTAGE_QUANTITY","mean"),
        MEAN_AVAILABLE_STOCK=("AVAILABLE_STOCK","mean")
    ).to_csv(out/"tables/phase6_inventory_scenarios.csv",index=False)
    stats=series_bootstrap_holm(eval_prob,inventory,n_boot,seed)
    stats.to_csv(out/"tables/radcsr_vs_baselines_bootstrap_holm.csv",index=False)
    if len(stats):
        stats[(stats.SCENARIO_ID=="S5_Bias") & (stats.COMPARISON.isin(["Static","WassersteinDRO"]))].to_csv(
            out/"tables/s5_bias_radcsr_vs_static_wasserstein.csv",index=False)
    clinical_status="Not computed: no verified clinical VEN mapping supplied."
    ven_path=root / "data" / "reference" / "clinical_ven_mapping.csv"
    if ven_path.exists():
        mapping=load_verified_ven_mapping(ven_path)
        evaluated_codes=set(rad_df.loc[rad_df.SPLIT=="test","VMP_SNOMED_CODE"].astype(str))
        mapped_codes=set(mapping.VMP_SNOMED_CODE.astype(str))
        missing_codes=sorted(evaluated_codes-mapped_codes)
        if missing_codes:
            raise ValueError(f"VEN mapping is incomplete for held-out cohort: {len(missing_codes):,} VMP codes missing")
        clinical_detail,clinical_summary=compare_clinical_priority(rad_df[rad_df.SPLIT=="test"],mapping)
        clinical_detail.to_parquet(proc/"clinical_priority.parquet",index=False)
        clinical_summary.to_csv(out/"tables/clinical_priority_ranking_comparison.csv",index=False)
        clinical_status={"mapping_file":str(ven_path),"mapped_medicines":int(mapping.VMP_SNOMED_CODE.nunique()),
            "evaluated_medicines":len(evaluated_codes),"mapping_coverage":1.0,
            "ranking_changed":bool(clinical_summary.RANKING_CHANGED.any())}
    test_joined=joined[joined.SPLIT=="test"]
    if test_joined.empty:
        raise RuntimeError("No held-out test rows available for RADCSR monotonicity acceptance gate.")
    mono_cfg = dict(ks=float(risk_cfg.get("kappa_sigma", .75)),
                    km=float(risk_cfg.get("kappa_mu", .50)),
                    kl=float(risk_cfg.get("kappa_l", 1.5)),
                    horizon_cap=risk_cfg.get("horizon_cap", 2.0))
    audit_rows=[]
    audit_sample=test_joined.groupby(KEYS+["SCENARIO_ID"],sort=False,dropna=False).head(1)
    for row in audit_sample.itertuples():
        audit_rows.append(monotonicity_audit(float(row.Q10),float(row.Q50),float(row.Q90),
            float(row.AVAILABLE_STOCK),float(row.LEAD_TIME_MONTHS),**mono_cfg))
    mono_pass=all(item["raw_pass"] for item in audit_rows)
    mono={"gate_status":"PASS" if mono_pass else "FAIL",
          "cases_checked":len(audit_rows),"all_raw_curves_monotone":mono_pass,
          "method":"analytic_monotone_risk_uplift_no_posthoc_envelope",
          "cases":[{k:v for k,v in item.items() if k!="probability_curve"} for item in audit_rows]}
    (out/"radcsr_monotonicity_audit.json").write_text(json.dumps(mono,indent=2))
    if not mono_pass:
        raise RuntimeError("RADCSR monotonicity acceptance gate failed; refusing to publish research outputs.")
    test_scenarios=scenarios[scenarios.SPLIT=="test"]
    mean_rel=test_scenarios.groupby("SCENARIO_ID").R_REL.mean().to_dict()
    mean_risk=rad_df[rad_df.SPLIT=="test"].groupby("SCENARIO_ID").RADCSR_PROBABILITY.mean().to_dict()
    stable_rel=mean_rel.get("S1_Stable")
    stress_checks={}
    for scenario,value in mean_rel.items():
        if scenario=="S1_Stable" or stable_rel is None:
            continue
        stress_checks[scenario]={"reliability_lift_vs_stable":float(value-stable_rel),
            "higher_reliability_penalty":bool(value>=stable_rel)}
    report={"forecast":forecast_report,"median_per_series_wape":float(per_series.WAPE.median()) if len(per_series) else None,
      "outlier_995_count":outlier_count,"series_with_outliers":int((outlier_summary.OUTLIER_COUNT>0).sum()),"prediction_rows":len(pred),"reliability_rows":len(reliability),
      "reliability_baseline_rows":len(intervals),"scenario_rows":len(scenarios),"ablation_rows":len(ablation_df),
      "inventory_rows":len(inventory),"risk_probability_rows":len(raw_probabilities),
      "calibrated_test_probability_rows":len(calibrated),
      "calibrator_status_counts":calibrated.CALIBRATOR_STATUS.value_counts().to_dict(),
      "signal_correlations_above_0_70":high_corr,"mean_reliability_by_test_scenario":mean_rel,
      "stress_reliability_lift_vs_stable":{k:float(v-mean_rel.get("S1_Stable",0)) for k,v in mean_rel.items()},
      "monotonicity_audit":{k:v for k,v in mono.items() if k!="cases"},
      "scenario_reliability_acceptance":stress_checks,
      "mean_radcsr_probability_by_test_scenario":mean_risk,
      "ablation_status":{"implemented_methods":sorted(ablation_df.METHOD.unique().tolist()),
          "clinical_ablation":"NOT_RUN: requires verified VEN mapping; no clinical category is inferred"},
      "statistical_inference":{"unit":"organisation/VMP/unit series cluster",
          "confidence_interval":"paired cluster bootstrap percentile 95%",
          "hypothesis_test":"paired series sign-flip randomisation",
          "multiple_testing":"global Holm step-down across scenario-method comparisons",
          "resamples":int(n_boot)},
      "bootstrap_resamples":n_boot,"disclaimer":DISCLAIMER,
      "clinical_priority":clinical_status,
      "interpretation":"Risk probabilities are isotonic-calibrated on validation-period simulated labels and evaluated on test-period simulated labels. Brier scores are not evidence of observed hospital stockout prediction.",
      "limitations":"SCMD has no observed stock, orders, lead times or stockout events. Inventory, lead-time demand, scenarios and stockout labels are simulated."}
    (out/"pipeline_report.json").write_text(json.dumps(report,indent=2,default=str))
    wins=int(stats.SIGNIFICANT_WIN.sum()) if len(stats) else 0
    losses=int(stats.SIGNIFICANT_LOSS.sum()) if len(stats) else 0
    verdict = [
        "# Final Research Verdict (C1–C6)",
        "",
        "This verdict is generated from the current pipeline run. All inventory and stockout outcomes are simulated.",
        "",
        f"## C1 — Forecasting performance",
        f"Forecast model: {forecast_report.get('model', 'not reported')}. Test metrics are in `output/tables/forecast_metrics.csv`; no universal pass threshold is asserted.",
        "",
        "## C2 — Reliability monitoring",
        f"Reliability rows: {len(reliability)}. Signal correlations above 0.70: {len(high_corr)}. Review `output/tables/reliability_signal_correlations.csv` and reliability baseline metrics.",
        "",
        "## C3 — RADCSR monotonicity",
        f"Acceptance gate: **{mono['gate_status']}** across {mono['cases_checked']} held-out series/scenario cases.",
        "",
        "## C4 — Scenario stress behaviour",
        f"Test scenarios evaluated: {len(mean_rel)}. Reliability and RADCSR risk summaries are recorded in `pipeline_report.json`; stress lift is descriptive, not a real-world validation.",
        "",
        "## C5 — Comparative risk performance",
        f"Significant wins after global Holm correction: {wins}. Significant losses: {losses}. Full paired series bootstrap and sign-flip results are in `output/tables/radcsr_vs_baselines_bootstrap_holm.csv`.",
        "",
        "## C6 — Clinical priority",
        f"{clinical_status if isinstance(clinical_status, str) else 'Verified VEN mapping supplied; see clinical priority ranking comparison.'}",
        "",
        "## Interpretation boundary",
        "SCMD does not contain observed stock-on-hand, procurement receipts, supplier lead times or stockout events. Therefore simulated Brier scores, simulated stockout labels and synthetic stress tests cannot establish observed hospital stockout prediction performance. Clinical priority conclusions require verified VEN provenance and complete cohort coverage.",
        ""
    ]
    (out/"final_verdict.md").write_text("\n".join(verdict),encoding="utf-8")
    return report
