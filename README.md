# SETReliable — SCMD Demand Forecasting and RADCSR

SETReliable studies whether reliability-adaptive demand-conditioned shortage risk improves simulated shortage-risk estimates when demand forecasts become unreliable.

## Research contribution

RADCSR is a three-layer pipeline:
1. Quantile forecasting: global LightGBM P10/P50/P90 forecasts, compared with previous-observation and seasonal-12 naive baselines.
2. Reliability monitoring: prequential error, bias, 90% conformal coverage and CUSUM drift signals form (R_{rel}). P10–P90 itself is an 80% interval; its PICP is reported as PICP_80.
3. Risk adjustment: adjusted spread, conditional mean and effective planning horizon produce a shortage probability from the forecast distribution and simulated availability.

## Dataset boundary

The NHSBSA Secondary Care Medicines Indicative Price dataset contains monthly organisation, VMP, unit, quantity and indicative-cost records. It does not contain on-hand inventory, purchase orders, supplier lead times or observed stockout events. Positive quantity is a modelling proxy, not confirmed consumption. Negative quantity is retained separately and is not automatically treated as demand.

Every inventory level, lead time, stress scenario and stockout label generated here is **SIMULATED**. The project does not claim to detect observed hospital stockouts.

## Layout

See [the architecture and research protocol](docs/SETReliable_SCMD_RADCSR.md). New implementation code belongs in `src/set_forecasting/`; legacy `src/scmd_*.py` modules remain compatibility entry points during migration.

## Setup

Python 3.11 is the supported environment.

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Place the source CSV under `data/raw/` (do not commit it), then:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
pytest -q
```

Generated intermediates go under `data/processed/`; reports and tables go under `output/`.

## Clinical priority

VEN weighting is optional and is only enabled when a medicine-level mapping with source name, source version and source URL is supplied at `data/reference/clinical_ven_mapping.csv`. No clinical category is inferred from demand, product name or model output.
