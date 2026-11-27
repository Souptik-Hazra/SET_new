# SETReliable — SCMD and RADCSR

## Standard structure

- `config/`: configuration
- `data/raw/`: original source data (never commit)
- `data/processed/`: reproducible Parquet intermediates (never commit)
- `docs/`: research protocol and data dictionary
- `notebooks/`: exploratory analysis and Colab execution
- `output/tables/`, `output/figures/`: generated artifacts (never commit)
- `scripts/run_pipeline.py`: unified command-line entry point for end-to-end and stage-level execution
- `src/set_forecasting/`: canonical package, organized by data, features, forecasting, reliability, risk and orchestration
- `tests/`: unit and regression tests

The former root-level `run_*_scmd.py` scripts and
`run_full_cohort_pipeline.py` have been consolidated into
`scripts/run_pipeline.py`. The legacy pharmacy-schema full-cohort workflow
remains disabled because its data contract is incompatible with SCMD.
The `src/scmd_*.py` modules are compatibility adapters only. New
implementation code belongs in `src/set_forecasting/`; forecasting and
reliability logic must not be duplicated in the adapters.

## Canonical execution

Install dependencies and run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
pytest -q
```

The default command performs Phase 1 ingestion followed by the canonical
RADCSR model pipeline. Stage-level commands are available through the same
entry point:

```bash
python scripts/run_pipeline.py phase1
python scripts/run_pipeline.py phase2
python scripts/run_pipeline.py phase3
python scripts/run_pipeline.py phase4
python scripts/run_pipeline.py model
```

Use `model` when the monthly series Parquet file already exists and only
the forecasting/reliability/RADCSR model pipeline needs to be rerun. Run
`python scripts/run_pipeline.py --help` for command-specific options.

## Research boundary

NHSBSA SCMD has monthly medicine quantity and indicative-cost records by
organisation, VMP and unit. It does not contain observed stock, orders,
supplier lead times or stockout events. Positive quantity is a modelling
proxy, not verified consumption. Negative quantities remain separate; their
meaning is not assumed.

All inventory, lead times, stress scenarios and stockout labels are simulated.
Brier scores therefore evaluate agreement with simulated labels, not real
hospital shortage prediction.

## RADCSR

RADCSR is the contribution, not a standalone forecaster:

1. Quantile LightGBM P10/P50/P90 supplies the demand distribution. P10–P90 is
   a nominal central 80% interval, so forecast evaluation reports PICP_80.
   The 90% reliability target is assessed separately using a prequential
   conformal interval around P50.
2. Prequential error, shortage-relevant bias, coverage and CUSUM drift signals
   form the reliability index.
3. RADCSR computes a monotone risk-uplift score from the baseline distribution
   tail, reliability severity and capped effective horizon. The adjusted
   distribution tail is exported separately; it is not the same quantity as
   the uplift score. Validation-only calibration and held-out evaluation are
   required before interpreting scores probabilistically.

Default reliability weights are fixed before evaluation: error .35, bias .25,
coverage .25, drift .15. The coverage signal uses past-only 90% conformal
intervals; it does not treat P10–P90 as a 90% interval. Pairwise signal
correlations above .70 are reported. The two-window shift detector is
explicitly labelled as a simple shift detector, not ADWIN.

The monotonicity audit tests the raw risk-uplift score over fixed
distribution/availability/lead-time conditions as reliability varies from 0
to 1. Monotonicity is a property of the score construction, not evidence of
calibration. The adjusted-distribution tail is separately reported and is not
guaranteed to be monotone for every availability threshold.
