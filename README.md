# SETReliable — SCMD Demand Forecasting and RADCSR

SETReliable studies whether reliability-adaptive demand-conditioned shortage risk improves simulated shortage-risk estimates when demand forecasts become unreliable.

## Research contribution

RADCSR is a three-layer pipeline:
1. Quantile forecasting: global LightGBM P10/P50/P90 forecasts, compared with previous-observation and seasonal-12 naive baselines.
2. Reliability monitoring: prequential error, bias, 90% conformal coverage and CUSUM drift signals form (R_{rel}). P10–P90 itself is an 80% interval; its PICP is reported as PICP_80.
3. Risk adjustment: a monotone reliability-uplift score is calculated from the baseline distribution tail and capped effective horizon. The adjusted-distribution tail is reported separately; validation-only isotonic calibration maps the score to a probability against simulated labels.

## Dataset boundary

The NHSBSA Secondary Care Medicines Indicative Price dataset contains monthly organisation, VMP, unit, quantity and indicative-cost records. It does not contain on-hand inventory, purchase orders, supplier lead times or observed stockout events. Positive quantity is a modelling proxy, not confirmed consumption. Negative quantity is retained separately and is not automatically treated as demand.

Every inventory level, lead time, stress scenario and stockout label generated here is **SIMULATED**. The project does not claim to detect observed hospital stockouts. Statistical comparisons with fewer than 10 independent organisation/VMP/unit series are marked descriptive and cannot produce significance claims.

## Layout

See [the architecture and research protocol](docs/SETReliable_SCMD_RADCSR.md). New implementation code belongs in `src/set_forecasting/`. Root-level `src/scmd_phase1.py`, `src/data_loader.py`, `src/features.py` and `src/preprocessing.py` are compatibility wrappers. The older `src/forecasting/`, `src/risk/`, `src/inventory/` and `src/reliability/` modules retain the original pharmacy-schema implementation for historical comparison; the SCMD pipeline does not import them.

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


## Active dataset and research scope

The supported workflow is the monthly NHSBSA SCMD pipeline. Read [SCMD_MIGRATION.md](SCMD_MIGRATION.md) and [SCMD_PHASES_1_4.md](SCMD_PHASES_1_4.md) before running experiments. The older pharmacy-transaction specification is historical only. Positive reported quantity is a modelling proxy, not verified consumption. All inventory, lead-time and stockout outcomes are simulated; SCMD alone does not support real hospital stockout prediction claims.

Run the supported end-to-end workflow from the repository root:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
```

The pipeline records its resolved configuration in `output/resolved_config.json`.
