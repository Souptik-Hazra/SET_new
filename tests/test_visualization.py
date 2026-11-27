import json
from pathlib import Path

import pandas as pd

from src.visualization.generate_figures import DISCLAIMER, generate_figures


def _write_parquet(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_csv(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_generates_ten_figures_from_pipeline_artifacts(tmp_path):
    keys = {
        "ODS_CODE": ["ORG1"] * 3,
        "VMP_SNOMED_CODE": ["VMP1"] * 3,
        "UNIT_OF_MEASURE_IDENTIFIER": ["U"] * 3,
    }
    periods = pd.date_range("2025-01-01", periods=3, freq="MS")
    base = pd.DataFrame({
        **keys, "PERIOD": periods, "MODEL_DEMAND": [10.0, 12.0, 11.0],
        "SPLIT": ["train", "train", "train"],
    })
    _write_parquet(tmp_path, "data/processed/scmd_monthly_series.parquet", base)

    forecasts = pd.DataFrame({
        **keys, "PERIOD": periods, "MODEL_DEMAND": [10.0, 12.0, 11.0],
        "Q10": [8.0, 9.0, 9.0], "Q50": [10.0, 11.0, 10.5],
        "Q90": [12.0, 13.0, 12.0], "SPLIT": ["test"] * 3,
    })
    _write_parquet(tmp_path, "data/processed/forecasts.parquet", forecasts)

    reliability = forecasts.copy()
    for column, values in {
        "R_E": [.1, .2, .15], "R_B": [.05, .1, .08],
        "R_C": [.2, .15, .1], "R_D": [0.0, 1.0, 0.0],
    }.items():
        reliability[column] = values
    _write_parquet(tmp_path, "data/processed/reliability.parquet", reliability)

    scenarios = pd.concat([
        forecasts.assign(SCENARIO_ID="S1_Stable"),
        forecasts.assign(SCENARIO_ID="S3_Drift"),
    ], ignore_index=True)
    _write_parquet(tmp_path, "data/processed/scenario_streams.parquet", scenarios)

    inventory = scenarios[["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER", "PERIOD", "SCENARIO_ID"]].copy()
    inventory["STOCKOUT_LABEL"] = [0, 1, 0, 1, 1, 0]
    inventory["SHORTAGE_QUANTITY"] = [0, 2, 0, 1, 3, 0]
    _write_parquet(tmp_path, "data/processed/simulated_inventory.parquet", inventory)

    calibrated = inventory[["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER", "PERIOD", "SCENARIO_ID"]].copy()
    calibrated["SPLIT"] = "test"
    calibrated["METHOD"] = ["RADCSR"] * 6
    calibrated["SHORTAGE_PROBABILITY"] = [.1, .7, .2, .6, .8, .1]
    _write_parquet(tmp_path, "data/processed/calibrated_risk_probabilities.parquet", calibrated)

    risk = scenarios[["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER", "PERIOD", "SCENARIO_ID", "SPLIT"]].copy()
    risk["RADCSR_PROBABILITY"] = [.1, .2, .3, .4, .5, .6]
    _write_parquet(tmp_path, "data/processed/radcsr_risk_details.parquet", risk)

    _write_csv(tmp_path, "output/tables/risk_brier_by_scenario.csv", pd.DataFrame({
        "SCENARIO_ID": ["S1_Stable", "S3_Drift"], "METHOD": ["RADCSR", "RADCSR"],
        "BRIER_SCORE": [.1, .2],
    }))
    _write_csv(tmp_path, "output/tables/phase7_ablation_metrics.csv", pd.DataFrame({
        "SCENARIO_ID": ["S1_Stable", "S3_Drift"], "METHOD": ["RADCSR", "No_Drift"],
        "BRIER_SCORE": [.1, .25],
    }))
    corr = pd.DataFrame(
        [[1.0, .2, .3, .1], [.2, 1.0, .4, .2], [.3, .4, 1.0, .5], [.1, .2, .5, 1.0]],
        columns=["R_E", "R_B", "R_C", "R_D"],
    )
    corr.insert(0, "signal", ["R_E", "R_B", "R_C", "R_D"])
    _write_csv(tmp_path, "output/tables/reliability_signal_correlations.csv", corr)
    _write_csv(tmp_path, "output/tables/inventory_sensitivity.csv", pd.DataFrame({
        "SERVICE_LEVEL": [0.90, 0.95, 0.90, 0.95],
        "INITIAL_COVER_MONTHS": [1.0, 1.0, 2.0, 2.0],
        "TARGET_COVER_MONTHS": [2.0, 2.0, 3.0, 3.0],
        "SIMULATED_STOCKOUT_RATE": [0.2, 0.1, 0.15, 0.05],
    }))

    generated = generate_figures(tmp_path)

    assert len(generated) == 11
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
    report = json.loads((tmp_path / "output/figures/figure_generation_report.json").read_text())
    assert report["generated_count"] == 10
    assert report["disclaimer"] == DISCLAIMER
    assert report["skipped"] == {}


def test_missing_artifacts_are_reported_without_fabricated_figures(tmp_path):
    generated = generate_figures(tmp_path)
    assert generated == []
    report = json.loads((tmp_path / "output/figures/figure_generation_report.json").read_text())
    assert report["generated_count"] == 0
    assert len(report["skipped"]) == 10
    assert "illustrative" not in json.dumps(report).lower()
