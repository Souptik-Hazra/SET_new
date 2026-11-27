# SCMD Phases 1–4 — archived implementation notes

> **Status: ARCHIVED.** The individual `run_phase1_scmd.py` through
> `run_phase4_scmd.py` commands are compatibility entry points only. They are
> not separate stages in the supported execution workflow. The implementation
> is consolidated into `src/set_forecasting/`.

## Canonical implementation

| Phase | Package implementation |
|---|---|
| 1 — SCMD ingestion and monthly cohort | `src/set_forecasting/data/ingestion.py` |
| 2 — past-only feature engineering | `src/set_forecasting/features/engineering.py` |
| 3 — global quantile forecasting and naive baselines | `src/set_forecasting/forecasting/model.py` |
| 4 — prequential reliability index and interval baselines | `src/set_forecasting/reliability/index.py`, `baselines.py` |

The orchestration and downstream RADCSR stages are in
`src/set_forecasting/pipeline.py`.

## Supported execution

Run the complete workflow from the repository root:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
```

If monthly SCMD data has already been prepared and only the model/RADCSR
workflow needs to be rerun:

```bash
python run_phase5_8_scmd.py
```

The second command invokes the same canonical package orchestrator; it is not
a second implementation.

## Compatibility

The root-level phase scripts and `src/scmd_*.py` modules remain temporarily
for compatibility with older commands and imports. The Phase 3 and Phase 4
modules delegate to `src.set_forecasting`; they must not contain independent
forecasting or reliability algorithms. New code and tests should import from
the package directly.

## Research boundary

The source dataset contains no observed stock-on-hand, receipts, purchase
orders, supplier lead times or stockout events. Inventory outcomes are
SIMULATED. Stockout labels are not observed events. Positive reported quantity
is a modelling proxy, not verified consumption.
