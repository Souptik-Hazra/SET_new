"""Deprecated compatibility adapter for canonical reliability components.

New code must import from :mod:`src.set_forecasting.reliability`.
All calibration and reliability calculations are delegated to the package.
"""
import json
from pathlib import Path

import pandas as pd
from src.set_forecasting.config import get_config, validate_config

from src.set_forecasting.reliability.index import build_reliability_index

KEY = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events."
OBSERVED_DEMAND_DISCLAIMER = "Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."


def _stamp_observed_demand_frame(frame):
    """Mark reliability outputs as demand-derived, not simulated inventory outcomes."""
    result = frame.copy().drop(columns=["OUTCOME_TYPE", "STOCKOUTS_ARE_OBSERVED"], errors="ignore")
    result["DATA_TYPE"] = "OBSERVED_DEMAND"
    result["DISCLAIMER"] = OBSERVED_DEMAND_DISCLAIMER
    return result


def calibrate_intervals(predictions, target_coverage=0.90, window=8, min_history=3):
    """Expose canonical prequential intervals under legacy column names."""
    result = build_reliability_index(
        predictions,
        target_coverage=target_coverage,
        window=window,
        min_history=min_history,
    ).copy()
    result["CAL_Q10"] = result["CAL_LOWER_90"]
    result["CAL_Q90"] = result["CAL_UPPER_90"]
    result["ABS_ERROR"] = (result["MODEL_DEMAND"] - result["Q50"]).abs()
    result["SIGNED_ERROR"] = result["MODEL_DEMAND"] - result["Q50"]
    result["CALIBRATED_COVERED"] = result["CAL_COVERED_90"].astype("int8")
    result["ROLLING_MAE"] = result["ROLLING_MAE_PRIOR"]
    result["ROLLING_BIAS"] = result["ROLLING_BIAS_PRIOR"]
    result["ROLLING_COVERAGE"] = result["ROLLING_COVERAGE_PRIOR"]
    result["DRIFT_SIGNAL"] = (result["R_D"] > 0) | result["SHIFT_DETECTOR_FLAG"]
    return _stamp_observed_demand_frame(result)


def run_reliability(
    prediction_path="data/processed/scmd_test_predictions.parquet",
    output_dir="output",
    processed_dir="data/processed",
    target_coverage=None,
    window=None,
):
    reliability_config = validate_config(get_config())["reliability"]
    target_coverage = float(
        target_coverage if target_coverage is not None
        else reliability_config["target_coverage"]
    )
    window = int(window if window is not None else reliability_config["window"])

    calibrated = calibrate_intervals(
        pd.read_parquet(prediction_path),
        target_coverage=target_coverage,
        window=window,
    )
    output = Path(output_dir)
    processed = Path(processed_dir)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(processed / "scmd_reliability.parquet", index=False)

    if calibrated.empty:
        summary = pd.DataFrame(columns=KEY + [
            "N", "MAE", "BIAS", "RAW_COVERAGE", "CALIBRATED_COVERAGE", "DRIFT_RATE"
        ])
    else:
        calibrated["RAW_COVERED"] = (
            (calibrated.MODEL_DEMAND >= calibrated.Q10)
            & (calibrated.MODEL_DEMAND <= calibrated.Q90)
        )
        summary = calibrated.groupby(KEY, dropna=False).agg(
            N=("MODEL_DEMAND", "size"),
            MAE=("ABS_ERROR", "mean"),
            BIAS=("SIGNED_ERROR", "mean"),
            RAW_COVERAGE=("RAW_COVERED", "mean"),
            CALIBRATED_COVERAGE=("CALIBRATED_COVERED", "mean"),
            DRIFT_RATE=("DRIFT_SIGNAL", "mean"),
        ).reset_index()
    summary = _stamp_observed_demand_frame(summary)
    summary.to_csv(output / "tables" / "phase4_reliability_summary.csv", index=False)
    report = {
        "series_evaluated": len(summary),
        "prediction_rows": len(calibrated),
        "target_coverage": target_coverage,
        "window": window,
        "implementation": "src.set_forecasting.reliability.index.build_reliability_index",
        "compatibility_adapter": True,
        "DATA_TYPE": "OBSERVED_DEMAND",
        "disclaimer": OBSERVED_DEMAND_DISCLAIMER,
    }
    (output / "phase4_reliability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return calibrated, summary, report


__all__ = ["build_reliability_index", "calibrate_intervals", "run_reliability"]
