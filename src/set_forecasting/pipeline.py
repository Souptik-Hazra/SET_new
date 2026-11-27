"""Single orchestrator for SCMD → quantile forecasting → RADCSR → calibrated risk."""
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .config import get_config, validate_config
from .data.source import get_project_root
from .forecasting.model import train_and_predict
from .reliability.index import build_reliability_index
from .reliability.baselines import interval_baselines
from .risk.radcsr import radcsr_probability,radcsr_ablations,monotonicity_audit,KEYS,DISCLAIMER
from .risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities,brier_by_method,calibration_metrics,series_bootstrap_holm,calibrate_risk_probabilities,align_primary_method_observations,PRIMARY_RISK_METHODS
from .risk.clinical import load_verified_ven_mapping,compare_clinical_priority


def _stamp_output_csvs(output_dir):
    """Stamp every exported research table with its simulated-outcome boundary."""
    tables_dir = Path(output_dir) / "tables"
    for csv_path in tables_dir.glob("*.csv"):
        table = pd.read_csv(csv_path)
        table["OUTCOME_TYPE"] = "SIMULATED"
        table["STOCKOUTS_ARE_OBSERVED"] = 0
        table["DISCLAIMER"] = DISCLAIMER
        table.to_csv(csv_path, index=False)


def run_model_pipeline(series_path="data/processed/scmd_monthly_series.parquet",output_dir="output",processed_dir="data/processed",seed=None,n_boot=2000):
    if sys.version_info[:2] != (3, 11) and os.environ.get("SET_ALLOW_UNSUPPORTED_PYTHON") != "1":
        raise RuntimeError(
            f"Unsupported Python {platform.python_version()}; research pipeline requires Python 3.11.x. "
            "Set SET_ALLOW_UNSUPPORTED_PYTHON=1 only for an explicitly non-reproducible compatibility run."
        )
    root = get_project_root()
    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    series_path = project_path(series_path)
    out = project_path(output_dir)
    proc = project_path(processed_dir)
    cfg = validate_config(get_config())
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    resolved_config = dict(cfg)
    resolved_config["seed"] = int(seed)
    resolved_config["evaluation"] = dict(cfg.get("evaluation") or {})
    resolved_config["evaluation"]["bootstrap_resamples"] = int(n_boot)
    resolved_config["inventory"] = dict(cfg.get("inventory") or {})
    resolved_config["inventory"]["lead_time_months"] = list(resolved_config["inventory"].get("lead_time_months", [1, 2, 3]))
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, default=str), encoding="utf-8")
    rel_cfg=cfg.get("reliability",{});risk_cfg=cfg.get("radcsr",{});inv_cfg=cfg.get("inventory",{});data_cfg=cfg.get("data",{})
    radcsr_ks=float(risk_cfg.get("kappa_sigma",.50))
    radcsr_km=float(risk_cfg.get("kappa_mu",.75))
    if radcsr_km < radcsr_ks:
        raise ValueError("RADCSR configuration violates monotonicity: kappa_mu must be >= kappa_sigma")
    if radcsr_ks <= 0 or radcsr_km <= 0:
        raise ValueError("RADCSR strict discrimination requires positive kappa_sigma and kappa_mu")
    (out/"tables").mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    series=pd.read_parquet(series_path)
    required_series = KEYS + ["PERIOD", "MODEL_DEMAND", "OBSERVATION_MISSING", "SPLIT"]
    missing_series = sorted(set(required_series) - set(series.columns))
    if missing_series:
        raise ValueError(f"Monthly series file is missing required columns: {missing_series}")
    if series[KEYS + ["PERIOD", "SPLIT"]].isna().any().any():
        raise ValueError("Monthly series contains null business keys, periods or split labels")
    if series.duplicated(KEYS + ["PERIOD"]).any():
        raise ValueError("Monthly series must be unique by organisation/VMP/unit/month")
    series["PERIOD"] = pd.to_datetime(series["PERIOD"], errors="coerce")
    if series.PERIOD.isna().any():
        raise ValueError("Monthly series contains invalid PERIOD values")
    series["MODEL_DEMAND"] = pd.to_numeric(series["MODEL_DEMAND"], errors="coerce")
    observed = ~series["OBSERVATION_MISSING"].fillna(True).astype(bool)
    if series.loc[observed, "MODEL_DEMAND"].isna().any() or not np.isfinite(series.loc[observed, "MODEL_DEMAND"]).all():
        raise ValueError("Observed monthly demand must be finite and non-null")
    if (series.loc[observed, "MODEL_DEMAND"] < 0).any():
        raise ValueError("MODEL_DEMAND must be non-negative")
    allowed_splits = {"train", "validation", "test"}
    if not set(series.SPLIT.astype(str)).issubset(allowed_splits):
        raise ValueError("SPLIT must contain only train, validation and test")
    for key, group in series.groupby(KEYS, sort=False, dropna=False):
        bounds = {name: group.loc[group.SPLIT == name, "PERIOD"] for name in allowed_splits}
        if any(bounds[name].empty for name in allowed_splits):
            raise ValueError(f"Series {key} does not contain all chronological splits")
        if not (bounds["train"].max() < bounds["validation"].min() and
                bounds["validation"].max() < bounds["test"].min()):
            raise ValueError(f"Chronological split overlap/order violation for series {key}")
    features,pred,forecast_metrics,forecast_report=train_and_predict(series,random_state=int(seed),winsor_quantile=float(data_cfg.get("winsor_quantile",.995)))
    if not pred.empty and "FORECAST_MODEL" not in pred:
        raise RuntimeError("Forecast predictions are missing FORECAST_MODEL provenance.")
    provenance = pred.groupby(KEYS + ["FORECAST_MODEL"], as_index=False, dropna=False).agg(
        FORECAST_OBSERVATIONS=("PERIOD", "size"),
        VALIDATION_OBSERVATIONS=("SPLIT", lambda values: int((values == "validation").sum())),
        TEST_OBSERVATIONS=("SPLIT", lambda values: int((values == "test").sum())),
    )
    total_forecasts = int(provenance.FORECAST_OBSERVATIONS.sum()) if len(provenance) else 0
    provenance["FORECAST_SHARE"] = provenance.FORECAST_OBSERVATIONS / max(total_forecasts, 1)
    provenance["IS_FALLBACK"] = provenance.FORECAST_MODEL.eq("naive_fallback")
    provenance.to_csv(out/"tables/forecast_provenance.csv", index=False)
    source_summary = pred.groupby("FORECAST_MODEL", as_index=False).size().rename(
        columns={"size": "FORECAST_OBSERVATIONS"}
    ) if not pred.empty else pd.DataFrame(columns=["FORECAST_MODEL", "FORECAST_OBSERVATIONS"])
    source_summary["FORECAST_OBSERVATION_SHARE"] = (
        source_summary.FORECAST_OBSERVATIONS / max(total_forecasts, 1)
    )
    source_summary.to_csv(out/"tables/forecast_source_summary.csv", index=False)
    fallback_observations = int(provenance.loc[provenance.IS_FALLBACK, "FORECAST_OBSERVATIONS"].sum()) if len(provenance) else 0
    forecast_report.update({
        "forecast_observations": total_forecasts,
        "fallback_observations": fallback_observations,
        "fallback_observation_share": float(fallback_observations / total_forecasts) if total_forecasts else None,
        "forecast_sources": provenance.FORECAST_MODEL.value_counts().to_dict() if len(provenance) else {},
        "forecast_source_observation_share": dict(zip(
            source_summary.FORECAST_MODEL,
            source_summary.FORECAST_OBSERVATION_SHARE
        )),
    })
    features.to_parquet(proc/"features.parquet",index=False);pred.to_parquet(proc/"forecasts.parquet",index=False)
    forecast_metrics.to_csv(out/"tables/forecast_metrics.csv",index=False)
    test_pred=pred[pred.SPLIT=="test"].copy() if "SPLIT" in pred else pred
    per_series=[]
    for key,g in test_pred.groupby(KEYS,sort=False,dropna=False):
        y=g.MODEL_DEMAND.to_numpy(float);q=g.Q50.to_numpy(float)
        valid=np.isfinite(y)&np.isfinite(q)
        y_valid=y[valid];q_valid=q[valid]
        per_series.append(dict(zip(KEYS,key),N_TEST=int(valid.sum()),
            WAPE=float(np.abs(y_valid-q_valid).sum()/max(np.abs(y_valid).sum(),1e-9)*100) if valid.any() else np.nan,
            MAE=float(np.mean(np.abs(y_valid-q_valid))) if valid.any() else np.nan))
    per_series=pd.DataFrame(per_series)
    per_series.to_csv(out/"tables/per_series_forecast_metrics.csv",index=False)
    outlier_count=int(features.get("OUTLIER_995_FLAG",pd.Series(dtype=bool)).fillna(False).sum())
    outlier_summary=features.groupby(KEYS,as_index=False).agg(WINSOR_LIMIT_995=("WINSOR_LIMIT_995","first"),OUTLIER_COUNT=("OUTLIER_995_FLAG","sum"),N_PERIODS=("PERIOD","size"))
    outlier_summary.to_csv(out/"tables/per_series_outlier_summary.csv",index=False)
    if pred.empty:raise RuntimeError("No validation/test predictions; check cohort size and chronological split.")
    observed_pred = pred[~pred.get("OBSERVATION_MISSING", pd.Series(False, index=pred.index)).fillna(True).astype(bool)].copy()
    reliability_observed = build_reliability_index(
        observed_pred, target_coverage=float(rel_cfg.get("target_coverage",.90)),
        window=int(rel_cfg.get("window",8)), weights=rel_cfg.get("weights")
    )
    reliability_columns = [
        column for column in reliability_observed.columns
        if column not in pred.columns and column not in KEYS + ["PERIOD"]
    ]
    required_reliability_signals = [
        "R_E", "R_B", "R_B_SYMMETRIC", "R_B_OVER", "R_B_UNDER",
        "R_C", "R_D", "R_UNRELIABILITY", "R_REL", "SHIFT_DETECTOR_FLAG",
    ]
    for column in required_reliability_signals:
        if column not in reliability_columns:
            reliability_observed[column] = 0
            reliability_columns.append(column)
    reliability = pred.merge(
        reliability_observed[KEYS + ["PERIOD"] + reliability_columns],
        on=KEYS + ["PERIOD"], how="left", validate="one_to_one"
    )
    if reliability_columns:
        reliability = reliability.sort_values(KEYS + ["PERIOD"]).copy()
        for column in reliability_columns:
            reliability[column] = reliability.groupby(KEYS, sort=False, dropna=False)[column].ffill()
            if column in {"R_E", "R_B", "R_B_SYMMETRIC", "R_B_OVER", "R_B_UNDER", "R_C", "R_D", "R_UNRELIABILITY", "R_REL", "SHIFT_DETECTOR_FLAG"}:
                reliability[column] = reliability[column].fillna(0)
    reliability.to_parquet(proc/"reliability.parquet",index=False)
    intervals=interval_baselines(observed_pred,target_coverage=float(rel_cfg.get("target_coverage",.90)),window=int(rel_cfg.get("window",8)));intervals.to_parquet(proc/"reliability_baselines.parquet",index=False)
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
    configured_lead_times = tuple(inv_cfg.get("lead_time_months", [1, 2, 3]))
    scenario_input = reliability.copy()
    missing_for_simulation = scenario_input.get(
        "OBSERVATION_MISSING", pd.Series(False, index=scenario_input.index)
    ).fillna(True).astype(bool)
    # Missing actual demand is never rewritten in MODEL_DEMAND. For the synthetic
    # inventory stream only, use the forecast median and explicitly flag imputation.
    scenario_input["SCENARIO_BASE_DEMAND"] = scenario_input.MODEL_DEMAND.where(
        ~missing_for_simulation, scenario_input.Q50
    )
    scenario_input["DEMAND_IMPUTED_FOR_SIMULATION"] = missing_for_simulation
    scenarios=make_scenarios(scenario_input,seed,lead_times=configured_lead_times)
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
            kappa_sigma=radcsr_ks,kappa_mu=radcsr_km,
            kappa_l=float(risk_cfg.get("kappa_l",1.5)),
            horizon_cap=risk_cfg.get("horizon_cap",2.0))
        base=g[keys+["SPLIT","STOCKOUT_LABEL","AVAILABLE_STOCK","LEAD_TIME_MONTHS"] + (["FORECAST_MODEL"] if "FORECAST_MODEL" in g else [])].reset_index(drop=True)
        rad.append(pd.concat([base,result],axis=1))
        ab=radcsr_ablations(g,
            kappa_sigma=radcsr_ks,
            kappa_mu=radcsr_km,
            kappa_l=float(risk_cfg.get("kappa_l",1.5)),
            horizon_cap=risk_cfg.get("horizon_cap",2.0),
            weights=rel_cfg.get("weights")).reset_index(drop=True)
        ablation_identity = keys + ["SPLIT"] + (["FORECAST_MODEL"] if "FORECAST_MODEL" in g else [])
        ablation_keys = pd.concat([g[ablation_identity].reset_index(drop=True)] * ab.METHOD.nunique(), ignore_index=True)
        if len(ablation_keys) != len(ab):
            raise RuntimeError("Ablation key expansion does not match method output rows")
        ablation_rows.append(pd.concat([ablation_keys,ab.reset_index(drop=True)],axis=1))
    rad_df=pd.concat(rad,ignore_index=True);rad_df["METHOD"]="RADCSR"
    ablation_df=pd.concat(ablation_rows,ignore_index=True)
    ablation_df.to_parquet(proc/"radcsr_ablations.parquet",index=False)
    baseline=risk_baseline_probabilities(joined, seed=seed)
    rad_identity = keys + ["SPLIT", "METHOD"] + (["FORECAST_MODEL"] if "FORECAST_MODEL" in rad_df else [])
    rad_prob=rad_df[rad_identity+["RADCSR_PROBABILITY"]].rename(columns={"RADCSR_PROBABILITY":"SHORTAGE_PROBABILITY"})
    primary_raw = pd.concat([baseline, rad_prob], ignore_index=True)
    primary_raw, observation_counts = align_primary_method_observations(primary_raw)
    observation_counts.to_csv(out/"tables/observation_counts.csv", index=False)
    ablation_only = ablation_df[ablation_df.METHOD != "RADCSR"].copy()
    raw_probabilities=pd.concat([primary_raw,ablation_only],ignore_index=True)
    # Calibrate primary methods and ablations separately. The primary RADCSR
    # method appears exactly once; ablation RADCSR is represented by primary_raw.
    calibrated_primary=calibrate_risk_probabilities(primary_raw,inventory)
    calibrated_ablations=(
        calibrate_risk_probabilities(ablation_only,inventory)
        if not ablation_only.empty else ablation_only.copy()
    )
    calibrated=pd.concat([calibrated_primary,calibrated_ablations],ignore_index=True)
    primary_test = calibrated_primary[calibrated_primary.SPLIT == "test"].copy()
    primary_test, bootstrap_observation_counts = align_primary_method_observations(primary_test)
    if not bootstrap_observation_counts.empty:
        bootstrap_observation_counts.to_csv(out/"tables/bootstrap_observation_counts.csv", index=False)
    eval_prob = pd.concat([
        primary_test,
        calibrated_ablations[calibrated_ablations.SPLIT == "test"]
        if not calibrated_ablations.empty else calibrated_ablations
    ], ignore_index=True)
    calibrated.to_parquet(proc/"calibrated_risk_probabilities.parquet",index=False)
    raw_probabilities.to_parquet(proc/"raw_risk_probabilities.parquet",index=False)
    inventory.to_parquet(proc/"simulated_inventory.parquet",index=False)
    rad_df.to_parquet(proc/"radcsr_risk_details.parquet",index=False)
    eval_prob=eval_prob.drop(columns=["STOCKOUT_LABEL"],errors="ignore")
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
    primary_eval = primary_test.drop(columns=["STOCKOUT_LABEL"], errors="ignore")
    stats=series_bootstrap_holm(primary_eval,inventory,n_boot,seed)
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
    mono_cfg = dict(ks=radcsr_ks,
                    km=radcsr_km,
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
    mean_unreliability=test_scenarios.groupby("SCENARIO_ID").R_UNRELIABILITY.mean().to_dict()
    mean_risk=rad_df[rad_df.SPLIT=="test"].groupby("SCENARIO_ID").RADCSR_PROBABILITY.mean().to_dict()
    stable_unreliability=mean_unreliability.get("S1_Stable")
    stress_checks={}
    for scenario,value in mean_unreliability.items():
        if scenario=="S1_Stable" or stable_unreliability is None:
            continue
        stress_checks[scenario]={"unreliability_increase_vs_stable":float(value-stable_unreliability),
            "higher_unreliability_severity":bool(value>=stable_unreliability)}
    _stamp_output_csvs(out)
    report={"runtime":{"python_version":platform.python_version(),
        "python_supported":platform.python_version().startswith("3.11."),
        "python_override":os.environ.get("SET_ALLOW_UNSUPPORTED_PYTHON")=="1"},
      "forecast":forecast_report,"median_per_series_wape":float(per_series.WAPE.median()) if len(per_series) else None,
      "outlier_995_count":outlier_count,"series_with_outliers":int((outlier_summary.OUTLIER_COUNT>0).sum()),"prediction_rows":len(pred),"reliability_rows":len(reliability),
      "reliability_baseline_rows":len(intervals),
      "reliability_score_semantics":{"canonical_column":"R_UNRELIABILITY","legacy_alias":"R_REL","direction":"higher_values_mean_worse_forecast_reliability"},
      "configured_lead_time_months":list(configured_lead_times),"scenario_rows":len(scenarios),"ablation_rows":len(ablation_df),
      "inventory_rows":len(inventory),"risk_probability_rows":len(raw_probabilities),
      "calibrated_test_probability_rows":len(calibrated),
      "calibrator_status_counts":calibrated.CALIBRATOR_STATUS.value_counts().to_dict(),
      "probability_semantics":{
          "RADCSR_DISTRIBUTION_PROBABILITY":"p_adj=1-Phi((A-mu_eff)/sigma_eff). L_eff=min(L*(1+kappa_l*R_rel),L*horizon_cap); sigma_eff=sigma_L*(1+kappa_sigma*R_rel)*sqrt(L_eff/L). If A<mu_L: mu_eff=A+(mu_L-A)*(1+kappa_mu*R_rel)*(L_eff/L). If A>=mu_L: mu_eff=A-(A-mu_L)*exp(-kappa_mu*R_rel*L_eff/L), strictly below A for finite R. Require kappa_mu>=kappa_sigma.",
          "RADCSR_PROBABILITY":"1-(1-p_adj)*exp(-S), where p_adj=RADCSR_DISTRIBUTION_PROBABILITY and S=log(1+kappa_mu*R_rel)+log(L_eff/L). Strictly increasing in both regimes for positive coefficients; separate reliability uplift, not the adjusted-distribution tail. Validation calibration and held-out evaluation are required."
      },
      "signal_correlations_above_0_70":high_corr,"mean_unreliability_by_test_scenario":mean_unreliability,
      "stress_unreliability_increase_vs_stable":{k:float(v-mean_unreliability.get("S1_Stable",0)) for k,v in mean_unreliability.items()},
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
    wins=int(stats.get("SIGNIFICANT_WIN",pd.Series(dtype=bool)).sum())
    losses=int(stats.get("SIGNIFICANT_LOSS",pd.Series(dtype=bool)).sum())
    verdict = [
        "# Final Research Verdict (C1–C6)",
        "",
        DISCLAIMER,
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
        f"Test scenarios evaluated: {len(mean_unreliability)}. Forecast-unreliability and RADCSR risk summaries are recorded in `pipeline_report.json`; stress lift is descriptive, not a real-world validation.",
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
