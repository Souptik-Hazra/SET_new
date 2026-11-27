# SCMD Dataset Migration

The project migrated from pharmacy transaction records to the NHSBSA Secondary
Care Medicines Indicative Price dataset. The migration is complete for the
supported SCMD pipeline; the historical pharmacy-schema implementation is
retained only for traceability and must not be used for SCMD evaluation.

## Source schema

`YEAR_MONTH`, `ODS_CODE`, `VMP_SNOMED_CODE`, `VMP_PRODUCT_NAME`,
`UNIT_OF_MEASURE_IDENTIFIER`, `UNIT_OF_MEASURE_NAME`,
`TOTAL_QUANITY_IN_VMP_UNIT` (source spelling retained), and `INDICATIVE_COST`.

The source is monthly and aggregates medicine quantities and indicative costs
by NHS organisation. It is not patient-level demand, dispensing transactions,
or an inventory ledger.

## Series grain

A series is identified by
`ODS_CODE + VMP_SNOMED_CODE + UNIT_OF_MEASURE_IDENTIFIER`. The monthly
aggregation key additionally includes `YEAR_MONTH`. IDs are stored as strings.

## Quantity semantics

The pipeline preserves positive quantity, negative quantity, net quantity and
cost separately. Negative movements are not silently discarded or presumed to
be returns. The current candidate modelling target is positive quantity; this
is an explicit modelling choice to be validated, not a claim that it equals
net consumption.

Absent month records receive `OBSERVATION_MISSING=true`; they are not silently
treated as observed zero demand. Costs are tracked separately and are not used
as demand.

## Cohort

Default inclusion: at least 18 monthly periods and at least 6 observed
nonzero positive-quantity periods. Demand-pattern diagnostics use ADI =
observed periods / nonzero periods and CV² of positive demand sizes. Smooth,
Erratic, Intermittent and Lumpy are distinguished.

## Canonical package and execution

All active SCMD implementation code belongs in `src/set_forecasting/`:

- Phase 1: `data/ingestion.py`
- Phase 2: `features/engineering.py`
- Phase 3: `forecasting/model.py`
- Phase 4: `reliability/index.py` and `reliability/baselines.py`
- Phases 5–8: `pipeline.py`, `risk/` and `reliability/`

Run the single supported end-to-end command from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
pytest -q
```

The old individual phase scripts and `src/scmd_*.py` modules are compatibility
adapters only. Do not add new implementation logic to them.

## Reproducibility and limitations

Seed: 42. Raw and processed data are local and ignored by Git. Do not commit
the source CSV, Parquet files or generated reports.

The dataset does not contain observed inventory, orders, supplier lead times
or stockout labels. Any such outcomes are simulated and every relevant report
must state:

> Inventory outcomes are SIMULATED. Stockout labels are not observed events.
