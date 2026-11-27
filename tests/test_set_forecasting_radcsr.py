import numpy as np
import pandas as pd
from src.set_forecasting.features.engineering import create_features,leakage_check
from src.set_forecasting.reliability.index import build_reliability_index
from src.set_forecasting.reliability.baselines import interval_baselines
from src.set_forecasting.risk.radcsr import radcsr_probability,monotonicity_audit
from src.set_forecasting.risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities

KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def sample(n=36):
    """Build a deterministic fixture with exactly n rows and valid split labels."""
    if n < 3:
        raise ValueError("sample fixture requires at least 3 rows for train/validation/test")
    x={k:[v]*n for k,v in zip(KEYS,["ORG01","000123","U1"])}
    pattern=np.array([10,12,0,11,14,9],dtype=float)
    y=np.resize(pattern,n)
    train_end=max(1,int(n*.60))
    validation_end=max(train_end+1,int(n*.80))
    validation_end=min(validation_end,n-1)
    splits=(["train"]*train_end +
            ["validation"]*(validation_end-train_end) +
            ["test"]*(n-validation_end))
    x.update(PERIOD=pd.date_range("2023-01-01",periods=n,freq="MS"),
      MODEL_DEMAND=y,OBSERVATION_MISSING=[False]*n,
      SPLIT=splits,
      POSITIVE_QUANTITY=y,TOTAL_INDICATIVE_COST=y*2)
    return pd.DataFrame(x)

def test_feature_engineering_is_past_only():
    x=sample();f=create_features(x)
    assert f.loc[1,"LAG_1"]==10
    assert f.loc[0,"LAG_1"]!=f.loc[0,"MODEL_DEMAND"]
    assert leakage_check(f)
    changed=x.copy();changed.loc[20,"MODEL_DEMAND"]=999999
    f2=create_features(changed)
    # Changing current target cannot alter its own predictor features.
    assert f.loc[20,"LAG_1"]==f2.loc[20,"LAG_1"]
    assert f.loc[20,"ROLL_MEAN_6"]==f2.loc[20,"ROLL_MEAN_6"]

def test_reliability_score_is_prequential():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND-2;x["Q50"]=x.MODEL_DEMAND-1;x["Q90"]=x.MODEL_DEMAND+2
    a=build_reliability_index(x)
    x.loc[x.index[5],"MODEL_DEMAND"]+=100
    b=build_reliability_index(x)
    assert a.iloc[5]["R_REL"] == b.iloc[5]["R_REL"]
    assert set(["R_E","R_B","R_C","R_D","R_REL"]).issubset(a.columns)

def test_reliability_baselines_are_all_exported():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND-2;x["Q50"]=x.MODEL_DEMAND-1;x["Q90"]=x.MODEL_DEMAND+2
    out=interval_baselines(x)
    assert set(out.METHOD.unique())=={"Static","RollingQuantile","EWMA","ACI_simplified","DtACI_simplified","AdaPI_simplified","EnbPI_simplified"}

def test_radcsr_monotone_probability_for_fixed_case():
    r=np.linspace(0,1,21)
    out=radcsr_probability(np.repeat(8.,len(r)),np.repeat(10.,len(r)),np.repeat(13.,len(r)),
      np.repeat(15.,len(r)),np.repeat(2.,len(r)),r)
    assert out.RADCSR_PROBABILITY.is_monotonic_increasing
    audit=monotonicity_audit(8,10,13,15,2)
    assert audit["raw_pass"]
    assert audit["gate_status"] == "PASS"

def test_nine_scenarios_are_isolated_and_inventory_is_simulated():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND*.8;x["Q50"]=x.MODEL_DEMAND;x["Q90"]=x.MODEL_DEMAND*1.2
    x["R_REL"]=.2;x["R_E"]=.1;x["R_B"]=.1;x["R_C"]=.1;x["R_D"]=0
    s=make_scenarios(x)
    assert len(s.SCENARIO_ID.unique())==9
    stable=s[s.SCENARIO_ID=="S1_Stable"]
    bias=s[s.SCENARIO_ID=="S5_Bias"]
    assert np.allclose(stable.Q50,x.Q50)
    assert np.allclose(bias.Q50,x.Q50*1.2)
    inv=simulate_inventory(s)
    assert (inv.STOCKOUTS_ARE_OBSERVED==0).all()
    assert set(inv.OUTCOME_TYPE)=={"SIMULATED"}

def test_risk_baseline_interface_returns_six_methods():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND*.8;x["Q50"]=x.MODEL_DEMAND;x["Q90"]=x.MODEL_DEMAND*1.2
    x["R_REL"]=.2
    s=make_scenarios(x);inv=simulate_inventory(s)
    frame=s.merge(inv[KEYS+["PERIOD","SCENARIO_ID","AVAILABLE_STOCK","DEMAND"]],on=KEYS+["PERIOD","SCENARIO_ID"])
    p=risk_baseline_probabilities(frame)
    assert set(p.METHOD.unique())=={"Static","RollingQuantile","EWMA","ACI_simplified","MonteCarlo","WassersteinDRO"}
    assert p.SHORTAGE_PROBABILITY.between(0,1).all()

def test_end_to_end_small_cohort(tmp_path):
    from src.set_forecasting.pipeline import run_model_pipeline
    frames=[]
    for j in range(4):
        x=sample(36)
        x["ODS_CODE"]=f"ORG{j:02d}"
        x["MODEL_DEMAND"]=x.MODEL_DEMAND*(j+1)
        frames.append(x)
    series=pd.concat(frames,ignore_index=True)
    source=tmp_path/"series.parquet"
    series.to_parquet(source,index=False)
    report=run_model_pipeline(series_path=str(source),output_dir=str(tmp_path/"output"),
        processed_dir=str(tmp_path/"processed"),seed=123,n_boot=100)
    assert report["prediction_rows"]>0
    assert report["inventory_rows"]>0
    assert report["ablation_rows"]>0
    import json
    resolved=json.loads((tmp_path/"output/resolved_config.json").read_text())
    assert resolved["seed"] == 123
    assert report["reliability_score_semantics"]["direction"] == "higher_values_mean_worse_forecast_reliability"
    ablations = pd.read_parquet(tmp_path/"processed/radcsr_ablations.parquet")
    assert ablations[KEYS + ["PERIOD", "SCENARIO_ID", "SPLIT"]].notna().all().all()
    assert ablations.METHOD.nunique() >= 7
    assert (tmp_path/"output/tables/s5_bias_radcsr_vs_static_wasserstein.csv").exists()

def test_probability_calibration_uses_validation_and_returns_test_only():
    from src.set_forecasting.risk.simulation import calibrate_risk_probabilities
    rows=[];labels=[]
    for i in range(40):
        split="validation" if i<32 else "test"
        p=i/40
        rows.append({**dict(zip(KEYS,["ORG","0001","U"])),"PERIOD":pd.Timestamp("2024-01-01")+pd.offsets.MonthBegin(i),
            "SCENARIO_ID":"S1_Stable","SPLIT":split,"METHOD":"RADCSR","SHORTAGE_PROBABILITY":p})
        labels.append({**dict(zip(KEYS,["ORG","0001","U"])),"PERIOD":pd.Timestamp("2024-01-01")+pd.offsets.MonthBegin(i),
            "SCENARIO_ID":"S1_Stable","STOCKOUT_LABEL":int(i%2==0)})
    out=calibrate_risk_probabilities(pd.DataFrame(rows),pd.DataFrame(labels),min_validation_rows=20)
    assert set(out.SPLIT)=={"test"}
    assert set(out.CALIBRATOR_STATUS)=={"isotonic_validation"}
    assert out.SHORTAGE_PROBABILITY.between(0,1).all()

def test_inventory_label_uses_realised_lead_time_demand():
    x=sample(36).head(4).copy()
    x["Q10"]=1.;x["Q50"]=1.;x["Q90"]=1.
    x["SCENARIO_ID"]="S1_Stable";x["SCENARIO_DEMAND"]=[10.,10.,0.,0.]
    x["LEAD_TIME_MONTHS"]=2
    inv=simulate_inventory(x,initial_cover=1.,target_cover=1.)
    assert inv.loc[0,"LEAD_TIME_DEMAND"]==20.
    assert inv.loc[0,"STOCKOUTS_ARE_OBSERVED"]==0
    assert inv.loc[0,"STOCKOUT_LABEL"]==int(inv.loc[0,"LEAD_TIME_DEMAND"]>inv.loc[0,"AVAILABLE_STOCK"])


def test_project_root_and_relative_source_resolution(tmp_path, monkeypatch):
    from src.set_forecasting.data.source import get_project_root, get_scmd_path
    root = get_project_root()
    assert root.name == "SET_new"
    relative = ".scmd_path_resolution_test.csv"
    target = root / relative
    try:
        target.write_text("temporary", encoding="utf-8")
        monkeypatch.chdir(root)
        assert get_scmd_path(relative) == target
    finally:
        target.unlink(missing_ok=True)


def test_radcsr_ablation_uses_configured_coefficients():
    from src.set_forecasting.risk.radcsr import radcsr_ablations
    x = sample(36).tail(8).copy()
    x["Q10"] = 5.0
    x["Q50"] = 10.0
    x["Q90"] = 15.0
    x["AVAILABLE_STOCK"] = 30.0
    x["LEAD_TIME_MONTHS"] = 2
    x["R_E"] = .2
    x["R_B"] = .4
    x["R_C"] = .6
    x["R_D"] = .8
    x["R_REL"] = .45
    default = radcsr_ablations(x)
    configured = radcsr_ablations(
        x, kappa_sigma=.6, kappa_mu=.8, kappa_l=2.0,
        horizon_cap=3.0,
        weights={"error": .1, "bias": .2, "coverage": .3, "drift": .4},
    )
    assert len(default) == len(configured)
    assert not np.allclose(
        default.loc[default.METHOD == "RADCSR_noBias", "SHORTAGE_PROBABILITY"],
        configured.loc[configured.METHOD == "RADCSR_noBias", "SHORTAGE_PROBABILITY"],
    )


def test_scenario_generation_is_reproducible_and_includes_sparse_and_correlated_cases():
    from src.set_forecasting.risk.simulation import SCENARIOS
    x = sample().tail(12).copy()
    x["Q10"] = x.MODEL_DEMAND * .8
    x["Q50"] = x.MODEL_DEMAND
    x["Q90"] = x.MODEL_DEMAND * 1.2
    first = make_scenarios(x, seed=123)
    second = make_scenarios(x, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert set(SCENARIOS).issubset(set(first.SCENARIO_ID))
    assert {"SHIFT_START_INDEX", "SHIFT_MAGNITUDE"}.issubset(first.columns)
    sparse = first[first.SCENARIO_ID == "S8_Intermittent"]
    assert (sparse.SCENARIO_DEMAND == 0).any()
    correlated = first[first.SCENARIO_ID == "S9_CorrelatedSurge"]
    assert np.isfinite(correlated.SCENARIO_DEMAND).all()


def test_radcsr_raw_probability_is_monotone_without_posthoc_envelope():
    from src.set_forecasting.risk.radcsr import radcsr_probability
    reliability = np.linspace(0.0, 1.0, 101)
    out = radcsr_probability(8.0, 10.0, 13.0, 15.0, 2.0, reliability)
    assert out.RADCSR_RAW_PROBABILITY.is_monotonic_increasing
    assert np.allclose(out.RADCSR_RAW_PROBABILITY, out.RADCSR_PROBABILITY)
    assert "RADCSR_DISTRIBUTION_PROBABILITY" in out


def test_radcsr_rejects_invalid_distribution_and_coefficients():
    import pytest
    with pytest.raises(ValueError, match="Q10"):
        radcsr_probability(12.0, 10.0, 13.0, 15.0, 2.0, .5)
    with pytest.raises(ValueError, match="non-negative"):
        radcsr_probability(8.0, 10.0, 13.0, 15.0, 2.0, .5, kappa_sigma=-.1)


def test_clinical_ablation_is_not_fabricated_without_verified_mapping():
    from src.set_forecasting.risk.radcsr import radcsr_ablations
    x = sample().tail(4).copy()
    x["Q10"] = 5.0; x["Q50"] = 10.0; x["Q90"] = 15.0
    x["AVAILABLE_STOCK"] = 8.0; x["LEAD_TIME_MONTHS"] = 2
    x["R_E"] = .2; x["R_B"] = .4; x["R_C"] = .6; x["R_D"] = .8; x["R_REL"] = .45
    methods = set(radcsr_ablations(x).METHOD)
    assert "RADCSR_noClinical" not in methods
    assert "RADCSR" in methods


def test_series_bootstrap_holm_uses_series_unit_and_reports_global_adjustment():
    from src.set_forecasting.risk.simulation import series_bootstrap_holm
    x = pd.concat([sample().head(12), sample().head(12).assign(ODS_CODE="ORG02")], ignore_index=True)
    x["SCENARIO_ID"] = "S1_Stable"
    x["STOCKOUT_LABEL"] = (x.MODEL_DEMAND > 10).astype(int)
    inventory = x[KEYS + ["PERIOD", "SCENARIO_ID", "STOCKOUT_LABEL"]].copy()
    rows = []
    for method, offset in [("RADCSR", 0.0), ("Static", .1), ("EWMA", .2)]:
        p = np.clip(.2 + .04 * x.MODEL_DEMAND + offset, 0, 1)
        z = x[KEYS + ["PERIOD", "SCENARIO_ID"]].copy()
        z["METHOD"] = method
        z["SHORTAGE_PROBABILITY"] = p
        rows.append(z)
    result = series_bootstrap_holm(pd.concat(rows), inventory, n_boot=100, seed=7)
    assert set(result.TEST) == {"paired_series_sign_flip"}
    assert (result.N_SERIES >= 2).all()
    assert result.HOLM_ADJUSTED_P.between(0, 1).all()
    assert (result.INFERENCE_STATUS == "LOW_CLUSTER_COUNT_DESCRIPTIVE_ONLY").all()
    assert not result.SIGNIFICANT_WIN.any()
    assert not result.SIGNIFICANT_LOSS.any()


def test_low_availability_regime_is_strictly_discriminating():
    from src.set_forecasting.risk.radcsr import radcsr_probability
    r = np.linspace(0.0, 1.0, 21)
    out = radcsr_probability(5.0, 10.0, 15.0, 8.0, 2.0, r)
    assert (out.MU_L_EFFECTIVE > 8.0).all()
    assert np.all(np.diff(out.RADCSR_DISTRIBUTION_PROBABILITY) > 0)
    assert np.all(np.diff(out.RADCSR_PROBABILITY) > 0)
    assert (out.L_EFF >= 2.0).all()


def test_high_availability_mean_approaches_availability_without_crossing():
    from src.set_forecasting.risk.radcsr import radcsr_probability
    r = np.linspace(0.0, 1.0, 101)
    availability = 30.0
    out = radcsr_probability(5.0, 10.0, 15.0, availability, 2.0, r)
    assert (out.MU_L_EFFECTIVE < availability).all()
    assert np.all(np.diff(out.MU_L_EFFECTIVE) > 0)
    assert np.all(np.diff(out.RADCSR_DISTRIBUTION_PROBABILITY) > 0)
    assert np.all(np.diff(out.RADCSR_PROBABILITY) > 0)
    assert out.SIGMA_ADJ.is_monotonic_increasing


def test_radcsr_rejects_nonpositive_lead_and_nonfinite_coefficients():
    import pytest
    with pytest.raises(ValueError, match="positive"):
        radcsr_probability(8.0, 10.0, 13.0, 15.0, 0.0, .5)
    with pytest.raises(ValueError, match="finite"):
        radcsr_probability(8.0, 10.0, 13.0, 15.0, 2.0, .5, kappa_sigma=np.nan)


def test_inventory_rejects_invalid_service_level():
    import pytest
    x = sample(4).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = x.MODEL_DEMAND
    x["Q10"] = 5.0; x["Q90"] = 15.0
    x["LEAD_TIME_MONTHS"] = 2
    with pytest.raises(ValueError, match="service_level"):
        simulate_inventory(x, service_level=1.0)


def test_inventory_right_censors_incomplete_lead_time_horizon():
    x = sample(4).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = [1.0, 1.0, 1.0, 1.0]
    x["Q10"] = 1.0; x["Q50"] = 1.0; x["Q90"] = 1.0
    x["LEAD_TIME_MONTHS"] = 2
    inv = simulate_inventory(x, initial_cover=1.0, target_cover=1.0)
    assert inv.loc[0, "LEAD_TIME_HORIZON_COMPLETE"] == 1
    assert inv.loc[0, "STOCKOUT_LABEL"] in (0, 1)
    assert inv.loc[len(inv) - 1, "LEAD_TIME_HORIZON_COMPLETE"] == 0
    assert pd.isna(inv.loc[len(inv) - 1, "STOCKOUT_LABEL"])


def test_observed_inventory_loader_requires_provenance_and_valid_labels(tmp_path):
    from src.set_forecasting.data.observed_inventory import load_observed_inventory
    source = tmp_path / "observed_inventory.csv"
    pd.DataFrame({
        "ODS_CODE": ["ORG01"], "VMP_SNOMED_CODE": ["000123"],
        "UNIT_OF_MEASURE_IDENTIFIER": ["U1"], "PERIOD": ["2025-01-15"],
        "OPENING_STOCK": [100], "RECEIPTS": [20], "ISSUES": [30],
        "LEAD_TIME_DAYS": [14], "STOCKOUT_FLAG": [0],
        "SOURCE_NAME": ["Hospital inventory system"], "SOURCE_VERSION": ["2025-01"],
        "SOURCE_URL": ["https://example.org/data-dictionary"],
    }).to_csv(source, index=False)
    result = load_observed_inventory(source)
    assert result.loc[0, "VMP_SNOMED_CODE"] == "000123"
    assert result.loc[0, "OUTCOME_TYPE"] == "OBSERVED"
    assert result.loc[0, "STOCKOUTS_ARE_OBSERVED"] == 1


def test_observed_inventory_loader_rejects_missing_provenance(tmp_path):
    import pytest
    from src.set_forecasting.data.observed_inventory import load_observed_inventory
    source = tmp_path / "bad_inventory.csv"
    pd.DataFrame({"ODS_CODE": ["ORG01"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_observed_inventory(source)


def test_calibration_metrics_exclude_censored_rows_and_report_metrics():
    from src.set_forecasting.risk.simulation import calibration_metrics
    x = sample(6).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    inventory = x[KEYS + ["PERIOD", "SCENARIO_ID"]].copy()
    inventory["STOCKOUT_LABEL"] = [0, 1, 0, 1, np.nan, np.nan]
    inventory["OUTCOME_TYPE"] = "SIMULATED"
    probabilities = inventory[KEYS + ["PERIOD", "SCENARIO_ID"]].copy()
    probabilities["METHOD"] = "RADCSR"
    probabilities["SHORTAGE_PROBABILITY"] = [.1, .8, .2, .7, .5, .5]
    result = calibration_metrics(probabilities, inventory, n_bins=5)
    assert result.loc[0, "N"] == 4
    assert 0 <= result.loc[0, "EXPECTED_CALIBRATION_ERROR"] <= 1
    assert result.loc[0, "OUTCOME_TYPE"] == "SIMULATED"


def test_inventory_sensitivity_grid_runs_on_shared_scenario_streams():
    from src.set_forecasting.risk.simulation import inventory_sensitivity
    x = sample(5).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = x.MODEL_DEMAND
    x["Q10"] = 5.0; x["Q50"] = 10.0; x["Q90"] = 15.0
    x["LEAD_TIME_MONTHS"] = 2
    result = inventory_sensitivity(x, service_levels=(.95,),
                                   initial_covers=(1., 2.), target_covers=(2.,))
    assert len(result) == 6
    assert set(result.BASE_LEAD_TIME_MONTHS) == {1, 2, 3}
    assert set(result.INITIAL_COVER_MONTHS) == {1.0, 2.0}
    assert result.SIMULATED_STOCKOUT_RATE.between(0, 1).all()


def test_ingestion_persists_filtered_series_with_chronological_splits(tmp_path):
    from src.set_forecasting.data.ingestion import build_monthly_series

    rows = []
    for org in ("ORG01", "ORG02"):
        for i, period in enumerate(pd.date_range("2023-01-01", periods=24, freq="MS")):
            rows.append({
                "YEAR_MONTH": period.strftime("%Y%m"),
                "ODS_CODE": org,
                "VMP_SNOMED_CODE": "000123",
                "UNIT_OF_MEASURE_IDENTIFIER": "U1",
                "VMP_PRODUCT_NAME": "Test medicine",
                "UNIT_OF_MEASURE_NAME": "tablet",
                "SOURCE_RECORDS": 1,
                "POSITIVE_QUANTITY": float(10 + i),
                "NEGATIVE_QUANTITY": 0.0,
                "NET_QUANTITY": float(10 + i),
                "TOTAL_INDICATIVE_COST": float((10 + i) * 2),
                "VALID_QUANTITY_RECORDS": 1,
                "INVALID_QUANTITY_RECORDS": 0,
                "MISSING_OR_INVALID_COST_RECORDS": 0,
                "NEGATIVE_COST_RECORDS": 0,
            })
    aggregated = tmp_path / "aggregated.parquet"
    series_path = tmp_path / "monthly.parquet"
    pd.DataFrame(rows).to_parquet(aggregated, index=False)

    returned, cohort, report = build_monthly_series(
        aggregated, series_path, tmp_path / "cohort.csv",
        tmp_path / "diagnostics.json", min_periods=18, min_nonzero=6,
        load_series=False,
    )

    assert returned is None
    assert cohort.COHORT_INCLUDED.all()
    saved = pd.read_parquet(series_path)
    assert {"SPLIT", "TRAIN_END", "VALIDATION_END", "TEST_START"}.issubset(saved.columns)
    assert set(saved.SPLIT) == {"train", "validation", "test"}
    assert saved.groupby(KEYS).size().eq(24).all()
    assert report["series_retained"] == 2
    assert "bounded batches" in report["memory_strategy"]


def test_inventory_excludes_receipt_due_after_current_lead_time_window():
    x = sample(3).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = 0.0
    x["Q10"] = 2.0
    x["Q90"] = 2.0
    x["LEAD_TIME_MONTHS"] = [2, 1, 1]

    inventory = simulate_inventory(x, service_level=.95, initial_cover=0.0, target_cover=1.0)

    # The first order is due at t=2. At t=1, a one-month horizon covers
    # only t=1; the receipt due at t=2 must not be included yet.
    assert inventory.loc[0, "ORDER_PLACED"] > 0
    assert inventory.loc[1, "PIPELINE_WITHIN_LEAD_TIME"] == 0
    assert inventory.loc[1, "AVAILABLE_STOCK"] == inventory.loc[1, "ON_HAND_AVAILABLE"]


def test_configured_lead_times_and_scenario_perturbations_are_recorded():
    import json
    x = sample(12).copy()
    x["Q10"] = x.MODEL_DEMAND * .8
    x["Q50"] = x.MODEL_DEMAND
    x["Q90"] = x.MODEL_DEMAND * 1.2

    scenarios = make_scenarios(x, seed=17, lead_times=(3,))
    stable = scenarios[scenarios.SCENARIO_ID == "S1_Stable"]
    shock = scenarios[scenarios.SCENARIO_ID == "S6_LeadTimeShock"]
    combined = scenarios[scenarios.SCENARIO_ID == "S7_Combined"]

    assert set(stable.LEAD_TIME_MONTHS) == {3}
    assert set(shock.LEAD_TIME_MONTHS) == {5}
    metadata = json.loads(combined.PERTURBATIONS.iloc[0])
    assert {item["name"] for item in metadata} == {
        "surge", "gradual_drift", "lognormal_volatility", "lead_time_shock"
    }
    assert combined.SHIFT_MAGNITUDE.iloc[0] == .5


def test_inventory_sensitivity_includes_lead_time_assumption_grid():
    from src.set_forecasting.risk.simulation import inventory_sensitivity
    x = sample(5).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = x.MODEL_DEMAND
    x["Q10"] = 5.0
    x["Q50"] = 10.0
    x["Q90"] = 15.0
    x["LEAD_TIME_MONTHS"] = 2
    result = inventory_sensitivity(
        x, service_levels=(.95,), initial_covers=(1.,),
        target_covers=(2.,), lead_times=(1, 3)
    )
    assert set(result.BASE_LEAD_TIME_MONTHS) == {1, 3}
    assert len(result) == 2


def test_unreliability_score_has_explicit_direction_and_legacy_alias():
    x = sample().tail(10).copy()
    x["Q10"] = x.MODEL_DEMAND - 2
    x["Q50"] = x.MODEL_DEMAND - 1
    x["Q90"] = x.MODEL_DEMAND + 2
    result = build_reliability_index(x)
    assert "R_UNRELIABILITY" in result
    np.testing.assert_allclose(result.R_UNRELIABILITY, result.R_REL)



def test_inventory_lead_time_label_respects_receipt_arrival_timing():
    x = sample(4).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = [0.0, 10.0, 0.0, 0.0]
    x["Q10"] = 20.0
    x["Q50"] = 20.0
    x["Q90"] = 20.0
    x["LEAD_TIME_MONTHS"] = 2

    inventory = simulate_inventory(
        x, service_level=.95, initial_cover=0.0, target_cover=1.0
    )

    # An order placed at t=0 arrives at t=2. It cannot prevent the shortage
    # at t=1 even though total horizon receipts exceed total horizon demand.
    assert inventory.loc[0, "ORDER_PLACED"] == 20.0
    assert inventory.loc[1, "PIPELINE_WITHIN_LEAD_TIME"] == 20.0
    assert inventory.loc[1, "AVAILABLE_STOCK"] == 20.0
    assert inventory.loc[1, "LEAD_TIME_DEMAND"] == 10.0
    assert inventory.loc[1, "MONTHLY_STOCKOUT_LABEL"] == 1
    assert inventory.loc[1, "STOCKOUT_LABEL"] == 1
    assert inventory.loc[1, "LEAD_TIME_STOCKOUT_MONTHS"] == 1


def test_inventory_rejects_missing_calendar_months():
    import pytest

    x = sample(4).copy()
    x["SCENARIO_ID"] = "S1_Stable"
    x["SCENARIO_DEMAND"] = x.MODEL_DEMAND
    x["Q10"] = 5.0
    x["Q90"] = 15.0
    x["LEAD_TIME_MONTHS"] = 2
    x = x.drop(index=1).reset_index(drop=True)

    with pytest.raises(ValueError, match="contiguous calendar months"):
        simulate_inventory(x)

def test_all_four_radcsr_probability_sweeps_have_strictly_positive_increments():
    from src.set_forecasting.risk.radcsr import radcsr_probability, infer_lead_time_distribution
    reliability = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    q10, q50, q90, lead = 80.0, 100.0, 130.0, 2.0
    mu, sigma = infer_lead_time_distribution(q10, q50, q90, lead)
    for availability in (mu + sigma, max(0.0, mu - sigma)):
        out = radcsr_probability(
            np.full(len(reliability), q10), np.full(len(reliability), q50),
            np.full(len(reliability), q90), np.full(len(reliability), availability),
            np.full(len(reliability), lead), reliability,
            kappa_sigma=.50, kappa_mu=.75, kappa_l=1.5, horizon_cap=2.0,
        )
        for column in ("RADCSR_DISTRIBUTION_PROBABILITY", "RADCSR_PROBABILITY"):
            values = out[column].to_numpy(float)
            assert np.all(np.diff(values) > 0.0), (column, availability, values.tolist())
    high = radcsr_probability(q10, q50, q90, mu + sigma, lead, reliability,
                              kappa_sigma=.50, kappa_mu=.75)
    assert (high.MU_L_EFFECTIVE < mu + sigma).all()


def test_radcsr_enforces_mean_coefficient_at_least_sigma_coefficient():
    import pytest
    with pytest.raises(ValueError, match="kappa_mu >= kappa_sigma"):
        radcsr_probability(8.0, 10.0, 13.0, 15.0, 2.0, .5,
                           kappa_sigma=.75, kappa_mu=.50)



def test_primary_risk_methods_are_aligned_to_explicit_common_keys():
    import pytest
    from src.set_forecasting.risk.simulation import (
        PRIMARY_RISK_METHODS, align_primary_method_observations,
    )
    rows = []
    periods = pd.date_range("2025-01-01", periods=2, freq="MS")
    for method in PRIMARY_RISK_METHODS:
        for index, period in enumerate(periods):
            if method == "Static" and index == 1:
                continue
            row = dict(zip(KEYS, ["ORG01", "000123", "U1"]))
            row.update(PERIOD=period, SCENARIO_ID="S1_Stable",
                       METHOD=method, SHORTAGE_PROBABILITY=.25)
            rows.append(row)
    aligned, counts = align_primary_method_observations(pd.DataFrame(rows))
    assert aligned.groupby("METHOD").size().nunique() == 1
    assert aligned.groupby("METHOD").size().iloc[0] == 1
    static = counts[counts.METHOD == "Static"].iloc[0]
    assert static.INPUT_KEYS == 1
    assert static.ALIGNED_KEYS == 1
    assert static.DROPPED_KEYS == 0
    assert (counts.ALIGNED_KEYS == 1).all()
    assert (counts.loc[counts.METHOD != "Static", "DROPPED_KEYS"] == 1).all()

    duplicate = pd.concat([pd.DataFrame(rows), pd.DataFrame(rows).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate primary method/evaluation keys"):
        align_primary_method_observations(duplicate)


def test_calibration_rejects_duplicate_method_evaluation_keys():
    import pytest
    from src.set_forecasting.risk.simulation import calibrate_risk_probabilities
    x = sample(4)
    x["SCENARIO_ID"] = "S1_Stable"
    x["METHOD"] = "RADCSR"
    x["SHORTAGE_PROBABILITY"] = .25
    duplicated = pd.concat([x, x.iloc[[0]]], ignore_index=True)
    inventory = x[KEYS + ["PERIOD", "SCENARIO_ID"]].copy()
    inventory["STOCKOUT_LABEL"] = 0
    with pytest.raises(ValueError, match="duplicate method/evaluation keys"):
        calibrate_risk_probabilities(duplicated, inventory)


def test_ingestion_persists_missing_month_model_demand_as_nan(tmp_path):
    from src.set_forecasting.data.ingestion import build_monthly_series, compute_adi_cv2

    rows = []
    periods = pd.date_range("2023-01-01", periods=24, freq="MS")
    for period in periods:
        if period == periods[8]:
            continue
        rows.append({
            "YEAR_MONTH": period.strftime("%Y%m"),
            "ODS_CODE": "ORG01",
            "VMP_SNOMED_CODE": "000123",
            "UNIT_OF_MEASURE_IDENTIFIER": "U1",
            "VMP_PRODUCT_NAME": "Test medicine",
            "UNIT_OF_MEASURE_NAME": "tablet",
            "SOURCE_RECORDS": 1,
            "POSITIVE_QUANTITY": 10.0,
            "NEGATIVE_QUANTITY": 0.0,
            "NET_QUANTITY": 10.0,
            "TOTAL_INDICATIVE_COST": 20.0,
            "VALID_QUANTITY_RECORDS": 1,
            "INVALID_QUANTITY_RECORDS": 0,
            "MISSING_OR_INVALID_COST_RECORDS": 0,
            "NEGATIVE_COST_RECORDS": 0,
        })
    aggregated = tmp_path / "aggregated.parquet"
    series_path = tmp_path / "monthly.parquet"
    pd.DataFrame(rows).to_parquet(aggregated, index=False)
    _, cohort, _ = build_monthly_series(
        aggregated, series_path, tmp_path / "cohort.csv",
        tmp_path / "diagnostics.json", min_periods=18, min_nonzero=6,
        load_series=False,
    )
    saved = pd.read_parquet(series_path)
    missing = saved[saved.OBSERVATION_MISSING]
    assert len(missing) == 1
    assert missing.MODEL_DEMAND.isna().all()
    observed = saved.loc[~saved.OBSERVATION_MISSING, "MODEL_DEMAND"].to_numpy(float)
    adi, _ = compute_adi_cv2(observed)
    assert adi == 1.0
    assert cohort.iloc[0].N_OBSERVED == 23
    cohort_export = pd.read_csv(tmp_path / "cohort.csv")
    assert cohort_export.loc[0, "DATA_TYPE"] == "OBSERVED_DEMAND"
    assert "OUTCOME_TYPE" not in cohort_export.columns
    assert cohort_export.loc[0, "DISCLAIMER"] == (
        "Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."
    )



def test_missing_month_lags_remain_missing_without_becoming_zero():
    x = sample(12).copy()
    x.loc[8, "OBSERVATION_MISSING"] = True
    x.loc[8, "MODEL_DEMAND"] = np.nan
    features = create_features(x)
    assert features.loc[8, "LAG_1"] == x.loc[7, "MODEL_DEMAND"]
    assert np.isnan(features.loc[9, "LAG_1"])
    assert features.loc[9, "LAG_2"] == x.loc[7, "MODEL_DEMAND"]
    assert not bool(features.loc[8, "TARGET_OBSERVED"])


def test_missing_month_uses_explicit_forecast_demand_only_for_simulation_and_censors_labels():
    x = sample(12).copy()
    x["Q10"] = 8.0
    x["Q50"] = 10.0
    x["Q90"] = 12.0
    x.loc[5, "OBSERVATION_MISSING"] = True
    x.loc[5, "MODEL_DEMAND"] = np.nan
    x["SCENARIO_BASE_DEMAND"] = x.MODEL_DEMAND.where(
        ~x.OBSERVATION_MISSING, x.Q50
    )
    x["DEMAND_IMPUTED_FOR_SIMULATION"] = x.OBSERVATION_MISSING
    scenarios = make_scenarios(x, seed=7, lead_times=(2,))
    stable = scenarios[scenarios.SCENARIO_ID == "S1_Stable"]
    assert np.isfinite(stable.SCENARIO_DEMAND).all()
    assert stable.loc[stable.PERIOD == x.loc[5, "PERIOD"], "DEMAND_IMPUTED_FOR_SIMULATION"].iloc[0]
    inventory = simulate_inventory(stable)
    missing_period = x.loc[5, "PERIOD"]
    assert inventory.loc[inventory.PERIOD == missing_period, "MONTHLY_STOCKOUT_LABEL"].isna().all()
    previous_period = x.loc[4, "PERIOD"]
    assert inventory.loc[inventory.PERIOD == previous_period, "STOCKOUT_LABEL"].isna().all()


def test_pipeline_stamps_csvs_by_observed_or_simulated_semantics(tmp_path):
    from src.set_forecasting.pipeline import _stamp_output_csvs, _stamp_simulated_frame
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"METHOD": ["RADCSR"], "BRIER_SCORE": [.1]}).to_csv(
        tables / "risk_brier_by_scenario.csv", index=False
    )
    pd.DataFrame({"MAE": [1.2]}).to_csv(tables / "forecast_metrics.csv", index=False)
    _stamp_output_csvs(tmp_path)
    result = pd.read_csv(tables / "risk_brier_by_scenario.csv")
    assert result.loc[0, "OUTCOME_TYPE"] == "SIMULATED"
    assert result.loc[0, "STOCKOUTS_ARE_OBSERVED"] == 0
    assert result.loc[0, "DISCLAIMER"] == (
        "Inventory outcomes are SIMULATED. Stockout labels are not observed events."
    )
    observed = pd.read_csv(tables / "forecast_metrics.csv")
    assert observed.loc[0, "DATA_TYPE"] == "OBSERVED_DEMAND"
    assert "OUTCOME_TYPE" not in observed.columns
    assert observed.loc[0, "DISCLAIMER"] == (
        "Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."
    )
    parquet_frame = _stamp_simulated_frame(pd.DataFrame({"METHOD": ["RADCSR"]}))
    assert parquet_frame.loc[0, "OUTCOME_TYPE"] == "SIMULATED"
    assert parquet_frame.loc[0, "STOCKOUTS_ARE_OBSERVED"] == 0
    assert parquet_frame.loc[0, "DISCLAIMER"] == result.loc[0, "DISCLAIMER"]



def test_mathematical_validation_reports_four_strict_monotonicity_sweeps():
    from src.set_forecasting.validation.mathematical import monotonicity
    sweeps = monotonicity()
    assert len(sweeps) == 4
    assert {(item["quantity"], item["regime"]) for item in sweeps} == {
        ("RADCSR_DISTRIBUTION_PROBABILITY", "A < mu_L"),
        ("RADCSR_PROBABILITY", "A < mu_L"),
        ("RADCSR_DISTRIBUTION_PROBABILITY", "A >= mu_L"),
        ("RADCSR_PROBABILITY", "A >= mu_L"),
    }
    for item in sweeps:
        assert len(item["sequence"]) == 21
        assert item["pass"] is True
        assert item["minimum_increment"] > 0.0
