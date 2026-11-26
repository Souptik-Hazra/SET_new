# SCMD implementation: Phases 1–4

All four phases now use monthly SCMD series keyed by `ODS_CODE`, `VMP_SNOMED_CODE`, and `UNIT_OF_MEASURE_IDENTIFIER`. Identifiers remain strings.

## Phase 1 — ingestion and cohort
`python run_phase1_scmd.py --source data/combined_secondary_care_medicines.csv`

DuckDB aggregates the eight-column NHSBSA source to organisation × VMP × unit × month. It retains positive, negative and net quantity, cost, record counts and invalid-value counts. Monthly calendars are explicit. An absent source month is marked `OBSERVATION_MISSING` and its modelling target remains null; it is not silently converted to zero. Cohort defaults are 18 calendar months and six observed positive months. Series splits are chronological 60/20/20.

`MODEL_DEMAND` is positive reported quantity and is only a modelling proxy. It is not verified consumption. Negative and net quantities are retained for sensitivity analysis.

## Phase 2 — features
`python run_phase2_scmd.py`

Creates past-only lags (1, 2, 3, 6, 12), rolling summaries (3, 6, 12), observation ratios, and monthly/quarterly/annual calendar features. Missing observations are excluded from target lags. No daily/weekly or medicine-ID fields are used.

## Phase 3 — forecasting
`python run_phase3_scmd.py`

Uses the fixed Phase 1 chronological split and evaluates only observed test targets. Quantile LightGBM (q10/q50/q90) is used when LightGBM is installed and sufficient validation data exists. Otherwise a monthly seasonal-naive-12 baseline is used with explicitly labelled fallback intervals. Outputs include MAE, RMSE, WAPE, interval coverage/width, pinball losses and quantile-crossing rate.

Install optional model dependency with `pip install lightgbm` to enable quantile LightGBM. The fallback is not equivalent to a trained probabilistic model.

## Phase 4 — reliability and calibration
`python run_phase4_scmd.py`

Per-series chronological calibration expands using only residuals observed before the current test timestamp. It reports rolling MAE, signed bias, coverage and a simple drift flag. This is an online diagnostic, not a guarantee of nominal coverage; assess coverage on a separate untouched evaluation period before making claims.

## Output artifacts
- `data/processed/scmd_monthly_aggregated.parquet`
- `data/processed/scmd_monthly_series.parquet`
- `data/processed/scmd_features.parquet`
- `data/processed/scmd_test_predictions.parquet`
- `data/processed/scmd_reliability.parquet`
- `output/tables/cohort_inclusion_report.csv`
- `output/tables/phase3_forecast_metrics.csv`
- `output/tables/phase4_reliability_summary.csv`

Inventory outcomes are SIMULATED. Stockout labels are not observed events. SCMD does not provide stock-on-hand, receipts, procurement, supplier lead times or observed stockout events.
