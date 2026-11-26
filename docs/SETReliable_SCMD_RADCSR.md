# SETReliable — SCMD and RADCSR

## Standard structure
- `config/`: configuration
- `data/raw/`: original source data (never commit)
- `data/processed/`: reproducible parquet intermediates (never commit)
- `docs/`: research protocol and data dictionary
- `notebooks/`: exploratory analysis
- `output/tables/`, `output/figures/`: generated artefacts (never commit)
- `scripts/run_pipeline.py`: single end-to-end entry point
- `src/set_forecasting/`: package split by responsibility (data, features, forecasting, reliability, risk)
- `tests/`: unit and regression tests

Legacy `src/scmd_*.py` files remain compatibility modules during migration; new code belongs in `src/set_forecasting/`.

## Research boundary
NHSBSA SCMD has monthly medicine quantity and indicative-cost records by organisation, VMP and unit. It does not contain observed stock, orders, supplier lead times or stockout events. Positive quantity is a modelling proxy, not verified consumption. Negative quantities remain separate; their meaning is not assumed.

All inventory, lead times, stress scenarios and stockout labels are simulated. Brier scores therefore evaluate agreement with simulated labels, not real hospital shortage prediction.

## RADCSR
RADCSR is the contribution, not a standalone forecaster:
1. Quantile LightGBM P10/P50/P90 supplies the demand distribution. P10–P90 is a nominal central 80% interval, so forecast evaluation reports PICP_80. The 90% reliability target is assessed separately using a prequential conformal interval around P50.
2. Prequential error, bias, coverage and CUSUM drift signals form (R_{rel}).
3. Spread, conditional mean and effective horizon adjustments map (R_{rel}), distribution and simulated availability to a shortage probability.

Default weights are fixed before evaluation: error .35, bias .25, coverage .25, drift .15. The coverage signal uses past-only 90% conformal intervals; it does not treat P10–P90 as a 90% interval. Pairwise signal correlations above .70 are reported. A two-window shift detector is explicitly labelled as a simple shift detector, not ADWIN.

The monotonicity audit tests a fixed distribution/availability/lead-time curve over (R_{rel}in[0,1]). Both raw adjusted probability and the monotone envelope are retained. The monotone envelope is an explicit model constraint, not evidence of calibration; calibration must be assessed separately.

## Run
```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
pytest -q
```
