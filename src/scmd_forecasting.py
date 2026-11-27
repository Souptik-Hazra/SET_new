"""Deprecated compatibility adapter for the canonical forecasting package.

New code must import from :mod:`src.set_forecasting.forecasting.model`.
This module intentionally contains no independent forecasting implementation.
"""
from pathlib import Path

import pandas as pd

from src.set_forecasting.forecasting.model import train_and_predict


def run_forecasting(
    series_path="data/processed/scmd_monthly_series.parquet",
    output_dir="output",
    processed_dir="data/processed",
    random_state=42,
    winsor_quantile=0.995,
):
    """Run the package forecaster while preserving the historical return contract."""
    series = pd.read_parquet(series_path)
    features, predictions, metrics, report = train_and_predict(
        series,
        random_state=random_state,
        winsor_quantile=winsor_quantile,
    )

    processed = Path(processed_dir)
    output = Path(output_dir)
    processed.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    features.to_parquet(processed / "scmd_features.parquet", index=False)
    predictions.to_parquet(processed / "scmd_test_predictions.parquet", index=False)
    metrics.to_csv(output / "tables" / "phase3_forecast_metrics.csv", index=False)

    report = {
        **report,
        "implementation": "src.set_forecasting.forecasting.model.train_and_predict",
        "compatibility_adapter": True,
    }
    return metrics, predictions, report


__all__ = ["train_and_predict", "run_forecasting"]
