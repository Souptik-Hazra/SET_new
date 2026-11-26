# SETReliable — SCMD Demand Forecasting and RADCSR

SETReliable studies whether reliability-adaptive demand-conditioned shortage risk improves simulated shortage-risk estimates when demand forecasts become unreliable.

## Research contribution

RADCSR is a three-layer pipeline:

1. Quantile forecasting: global LightGBM P10/P50/P90 forecasts, compared with previous-observation and seasonal-12 naive baselines.
2. Reliability monitoring: prequential error, bias, 90% conformal coverage and CUSUM drift signals form the reliability index. P10–P90 is an 80% interval; its PICP is reported as PICP_80.
3. Risk adjustment: a monotone reliability-uplift score is calculated from the baseline distribution tail and capped effective horizon. The adjusted-distribution tail is reported separately. Validation-only calibration is required before interpreting a score probabilistically.

## Dataset boundary

The NHSBSA Secondary Care Medicines Indicative Price dataset contains monthly organisation, VMP, unit, quantity and indicative-cost records. It does not contain on-hand inventory, purchase orders, supplier lead times or observed stockout events. Positive quantity is a modelling proxy, not confirmed consumption. Negative quantity is retained separately and is not automatically treated as demand.

**Inventory outcomes are SIMULATED. Stockout labels are not observed events.**

The project does not claim to detect observed hospital stockouts. Statistical comparisons with fewer than 10 independent organisation/VMP/unit series are descriptive and cannot support significance claims.

## Layout

See [the architecture and research protocol](docs/SETReliable_SCMD_RADCSR.md). New implementation code belongs in `src/set_forecasting/`. Root-level compatibility wrappers are retained only for migration support. The older pharmacy-schema implementation is historical; the SCMD pipeline does not import it.

## Setup

Python 3.11 is the supported environment.

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Canonical end-to-end run

Place the source CSV under `data/raw/` (do not commit it), then run from the repository root:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
```

This is the single documented end-to-end command. It performs Phase 1 ingestion and then calls `src.set_forecasting.pipeline.run_model_pipeline()`. The runner prints `PIPELINE_ENTRYPOINT=src.set_forecasting.pipeline.run_model_pipeline` as an execution trace. Generated intermediates go under `data/processed/`; reports and tables go under `output/`.

To rerun only the model pipeline after the processed monthly Parquet file already exists:

```bash
python run_phase5_8_scmd.py
```

This also calls the canonical `run_model_pipeline()`; it does not invoke the retired legacy implementation.

## Clinical priority

VEN weighting is optional and is enabled only when a medicine-level mapping with source name, source version and source URL is supplied at `data/reference/clinical_ven_mapping.csv`. No clinical category is inferred from demand, product name or model output.

## Evaluation outputs and sensitivity analysis

The active pipeline excludes right-censored lead-time outcomes from probability calibration, Brier scoring and paired statistical comparisons. See `LEAD_TIME_HORIZON_COMPLETE` in `data/processed/simulated_inventory.parquet`. Final periods with incomplete lead-time horizons are not treated as negative stockout labels.

Held-out calibration diagnostics are written to `output/tables/risk_calibration_metrics.csv`. Scenario streams are saved to `data/processed/scenario_streams.parquet`. After a successful canonical pipeline run, inventory assumption sensitivity can be run with:

```bash
python scripts/run_inventory_sensitivity.py
```

The default grid compares service levels 0.90/0.95/0.99, initial cover 1/2 months and target cover 2/3 months. Results are written to `output/tables/inventory_sensitivity.csv`. All outcomes remain simulated.

For independently observed inventory data, follow [the observed inventory data contract](docs/OBSERVED_INVENTORY_DATA_CONTRACT.md). Real-data evaluation is not enabled until an authorised observed inventory extract is supplied.
