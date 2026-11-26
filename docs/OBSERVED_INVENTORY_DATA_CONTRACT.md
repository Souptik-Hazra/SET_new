# Observed inventory data contract (P2)

The SCMD source does not contain stock-on-hand, receipts, issues, supplier lead times or observed stockout events. The synthetic inventory simulator must therefore remain explicitly labelled SIMULATED unless an independently sourced inventory extract is supplied.

## Required input

Place a provenance-backed CSV at `data/reference/observed_inventory.csv`. Required columns:

| Column | Meaning |
|---|---|
| ODS_CODE | Organisation identifier, retained as text |
| VMP_SNOMED_CODE | VMP identifier, retained as text |
| UNIT_OF_MEASURE_IDENTIFIER | Unit identifier, retained as text |
| PERIOD | Month or date, normalised to month |
| OPENING_STOCK | Observed opening quantity |
| RECEIPTS | Observed receipts during the period |
| ISSUES | Observed issues/dispensing during the period |
| LEAD_TIME_DAYS | Observed supplier lead time, if available |
| STOCKOUT_FLAG | Observed binary stockout outcome from the source system |
| SOURCE_NAME | Source system / dataset |
| SOURCE_VERSION | Extract version or snapshot date |
| SOURCE_URL | Public source URL or governed internal source reference |

Optional: `CLOSING_STOCK`. When supplied, the loader reports the stock-balance residual against opening stock + receipts − issues.

## Validation

`src/set_forecasting/data/observed_inventory.py` validates required columns, unique organisation × VMP × unit × month grain, non-negative finite quantities, binary outcome flags, and source provenance. It labels loaded records as `OUTCOME_TYPE=OBSERVED`; it never converts the synthetic simulator's labels into observed outcomes.

## Important integration boundary

This is a validated ingestion contract, not evidence that an observed dataset is present and not yet a replacement for the simulated inventory engine in the research pipeline. Integration requires agreement on:
- the source system's formal stockout definition and observation window;
- unit-of-measure conversions and product identifier mapping;
- whether receipts/issues are complete and timestamped;
- censoring and missing-period rules;
- organisation-level data permissions and privacy controls.

Do not claim real-world stockout prediction performance until this data is supplied, mapped, integrated and evaluated on a held-out period that was not used for model or threshold selection.

## VEN mapping

If clinical priority is evaluated, `data/reference/clinical_ven_mapping.csv` must cover every VMP in the held-out cohort. Each mapping row must include VEN class, substitutability, source name, source version and HTTP(S) source URL. Partial coverage is rejected rather than silently inner-joined.
