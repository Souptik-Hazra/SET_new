import inspect

import numpy as np
import pandas as pd
import pytest

from src.set_forecasting.reliability.index import build_reliability_index
from src.set_forecasting.reliability import baselines as interval_module
from src.set_forecasting.risk import simulation as risk_module
from src.set_forecasting.risk.radcsr import radcsr_probability
from src.set_forecasting.data.ingestion import build_monthly_series
from src.set_forecasting.validation.empirical_monotonicity import audit_frame

KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]


def _reliability_frame(periods):
    n = len(periods)
    return pd.DataFrame({
        "ODS_CODE": ["ORG"] * n,
        "VMP_SNOMED_CODE": ["000123"] * n,
        "UNIT_OF_MEASURE_IDENTIFIER": ["U1"] * n,
        "PERIOD": periods,
        "MODEL_DEMAND": np.full(n, 30.0),
        "Q10": np.full(n, 5.0),
        "Q50": np.full(n, 10.0),
        "Q90": np.full(n, 15.0),
    })


def test_reliability_resets_all_history_after_calendar_gap():
    continuous_periods = pd.date_range("2024-01-01", periods=16, freq="MS")
    gapped_periods = continuous_periods.delete(slice(8, 12))
    continuous = build_reliability_index(_reliability_frame(continuous_periods), window=8)
    gapped = build_reliability_index(_reliability_frame(gapped_periods), window=8)
    first_after_gap = gapped.index[gapped.PERIOD.eq(continuous_periods[12])][0]
    continuous_same_month = continuous.loc[continuous.PERIOD.eq(continuous_periods[12])].iloc[0]
    assert gapped.loc[first_after_gap, "R_REL"] == 0.0
    assert gapped.loc[first_after_gap, "R_D"] == 0.0
    assert not gapped.loc[gapped.PERIOD.ge(continuous_periods[12]), "SHIFT_DETECTOR_FLAG"].any()
    assert continuous_same_month.R_REL > 0.0


def test_simplified_baselines_have_qualified_names_and_deviation_docstrings():
    periods = pd.date_range("2025-01-01", periods=8, freq="MS")
    n = len(periods)
    frame = pd.DataFrame({
        "ODS_CODE": ["ORG"] * n, "VMP_SNOMED_CODE": ["000123"] * n,
        "UNIT_OF_MEASURE_IDENTIFIER": ["U1"] * n, "PERIOD": periods,
        "MODEL_DEMAND": np.linspace(10, 17, n), "DEMAND": np.linspace(10, 17, n),
        "Q10": np.full(n, 8.0), "Q50": np.full(n, 10.0), "Q90": np.full(n, 13.0),
        "AVAILABLE_STOCK": np.full(n, 9.0), "LEAD_TIME_MONTHS": np.full(n, 2),
        "SCENARIO_ID": ["S1_Stable"] * n, "SPLIT": ["test"] * n,
    })
    interval_names = set(interval_module.interval_baselines(frame).METHOD)
    risk_names = set(risk_module.risk_baseline_probabilities(frame).METHOD)
    assert {"ACI_simplified", "DtACI_simplified", "AdaPI_simplified", "EnbPI_simplified"}.issubset(interval_names)
    assert not {"ACI", "DtACI", "AdaPI", "EnbPI"} & interval_names
    assert "ACI_simplified" in risk_names and "ACI" not in risk_names
    assert "ACI_simplified" in risk_module.PRIMARY_RISK_METHODS
    for helper in (
        interval_module._aci_simplified_radius,
        interval_module._dtaci_simplified_radius,
        interval_module._adapi_simplified_radius,
        interval_module._enbpi_simplified_radius,
        risk_module._aci_simplified_probability,
    ):
        doc = inspect.getdoc(helper)
        assert doc and "published" in doc.lower() and ("not" in doc.lower() or "omits" in doc.lower())


def test_mu_adj_matches_effective_mean_in_both_availability_regimes():
    reliability = np.linspace(0.0, 1.0, 11)
    for availability in (8.0, 30.0):
        result = radcsr_probability(5.0, 10.0, 15.0, availability, 2.0, reliability)
        np.testing.assert_array_equal(result.MU_ADJ.to_numpy(), result.MU_L_EFFECTIVE.to_numpy())


def _aggregated_fixture(path):
    rows = []
    for i, period in enumerate(pd.date_range("2023-01-01", periods=24, freq="MS")):
        rows.append({
            "YEAR_MONTH": period.strftime("%Y%m"), "ODS_CODE": "ORG01",
            "VMP_SNOMED_CODE": "000123", "UNIT_OF_MEASURE_IDENTIFIER": "U1",
            "VMP_PRODUCT_NAME": "Test medicine", "UNIT_OF_MEASURE_NAME": "tablet",
            "SOURCE_RECORDS": 1, "POSITIVE_QUANTITY": float(10+i),
            "NEGATIVE_QUANTITY": 0.0, "NET_QUANTITY": float(10+i),
            "TOTAL_INDICATIVE_COST": float((10+i)*2), "VALID_QUANTITY_RECORDS": 1,
            "INVALID_QUANTITY_RECORDS": 0, "MISSING_OR_INVALID_COST_RECORDS": 0,
            "NEGATIVE_COST_RECORDS": 0,
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_monthly_split_consumes_all_three_fractions(tmp_path):
    aggregated = tmp_path / "aggregated.parquet"
    _aggregated_fixture(aggregated)
    series_path = tmp_path / "series.parquet"
    build_monthly_series(
        aggregated, series_path, tmp_path / "cohort.csv", tmp_path / "diagnostics.json",
        min_periods=18, min_nonzero=6, train_frac=.50, val_frac=.25, test_frac=.25,
        load_series=False,
    )
    saved = pd.read_parquet(series_path)
    assert saved.groupby("SPLIT").size().to_dict() == {"train": 12, "validation": 6, "test": 6}
    assert set(saved.DATA_TYPE) == {"OBSERVED_DEMAND"}
    assert "OUTCOME_TYPE" not in saved.columns
    cohort = pd.read_csv(tmp_path / "cohort.csv")
    monthly = pd.read_csv(tmp_path / "monthly_data_summary.csv")
    assert set(cohort.DATA_TYPE) == {"OBSERVED_DEMAND"}
    assert set(monthly.DATA_TYPE) == {"OBSERVED_DEMAND"}


def test_inconsistent_split_fractions_raise_value_error(tmp_path):
    aggregated = tmp_path / "aggregated.parquet"
    _aggregated_fixture(aggregated)
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_monthly_series(
            aggregated, tmp_path / "series.parquet", tmp_path / "cohort.csv",
            tmp_path / "diagnostics.json", train_frac=.50, val_frac=.20, test_frac=.20,
        )


def test_empirical_monotonicity_script_passes_synthetic_fixture():
    rows = []
    for scenario, availability in (("S1_Stable", 8.0), ("S2_Surge", 30.0)):
        for month in range(2):
            rows.append({
                "ODS_CODE": "ORG", "VMP_SNOMED_CODE": "000123",
                "UNIT_OF_MEASURE_IDENTIFIER": "U1",
                "PERIOD": pd.Timestamp("2025-01-01") + pd.offsets.MonthBegin(month),
                "SCENARIO_ID": scenario, "SPLIT": "test",
                "Q10": 5.0, "Q50": 10.0, "Q90": 15.0,
                "AVAILABLE_STOCK": availability, "LEAD_TIME_MONTHS": 2,
                "R_REL": .4,
            })
    report, violations = audit_frame(pd.DataFrame(rows), grid_size=101, batch_size=2)
    assert report["status"] == "PASS"
    assert report["test_observations"] == 4
    assert report["distribution_pass"] == 4
    assert report["radcsr_pass"] == 4
    assert report["both_fail"] == 0
    assert report["OUTCOME_TYPE"] == "SIMULATED"
    assert violations.empty
    assert "OUTCOME_TYPE" in violations.columns


def test_observed_and_simulated_output_labels_are_not_conflated(tmp_path):
    from src.set_forecasting.pipeline import _stamp_simulated_frame, _stamp_output_csvs

    tables = tmp_path / "tables"
    tables.mkdir()
    observed_names = (
        "forecast_metrics.csv", "per_series_forecast_metrics.csv",
        "forecast_provenance.csv", "forecast_source_summary.csv",
        "per_series_outlier_summary.csv", "reliability_baseline_metrics.csv",
        "reliability_signal_correlations.csv",
    )
    for name in observed_names:
        pd.DataFrame({"METRIC": [1.0], "OUTCOME_TYPE": ["SIMULATED"]}).to_csv(tables/name, index=False)
    pd.DataFrame({"BRIER_SCORE": [.1]}).to_csv(tables/"risk_brier_by_scenario.csv", index=False)
    _stamp_output_csvs(tmp_path)
    for name in observed_names:
        frame = pd.read_csv(tables/name)
        assert set(frame.DATA_TYPE) == {"OBSERVED_DEMAND"}
        assert "OUTCOME_TYPE" not in frame.columns
        assert set(frame.DISCLAIMER) == {"Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."}
    risk = pd.read_csv(tables/"risk_brier_by_scenario.csv")
    assert set(risk.OUTCOME_TYPE) == {"SIMULATED"}
    simulated_parquet = _stamp_simulated_frame(pd.DataFrame({"DATA_TYPE": ["OBSERVED_DEMAND"]}))
    simulated_path = tmp_path / "simulated_inventory.parquet"
    simulated_parquet.to_parquet(simulated_path, index=False)
    saved_inventory = pd.read_parquet(simulated_path)
    assert set(saved_inventory.OUTCOME_TYPE) == {"SIMULATED"}
    assert "DATA_TYPE" not in saved_inventory.columns



def test_standalone_phase_adapters_use_correct_data_type_labels():
    from src.scmd_forecasting import _stamp_observed_demand_frame as stamp_forecast
    from src.scmd_reliability import _stamp_observed_demand_frame as stamp_reliability
    from scripts.run_inventory_sensitivity import _stamp_simulated_outcomes

    raw = pd.DataFrame({"VALUE": [1.0], "OUTCOME_TYPE": ["SIMULATED"]})
    for stamped in (stamp_forecast(raw), stamp_reliability(raw)):
        assert set(stamped.DATA_TYPE) == {"OBSERVED_DEMAND"}
        assert "OUTCOME_TYPE" not in stamped.columns
        assert set(stamped.DISCLAIMER) == {"Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."}

    simulated = _stamp_simulated_outcomes(pd.DataFrame({"VALUE": [1.0], "DATA_TYPE": ["OBSERVED_DEMAND"]}))
    assert set(simulated.OUTCOME_TYPE) == {"SIMULATED"}
    assert set(simulated.STOCKOUTS_ARE_OBSERVED) == {0}
    assert "DATA_TYPE" not in simulated.columns
