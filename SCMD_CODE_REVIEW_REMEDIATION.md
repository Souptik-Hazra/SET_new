# SCMD full repository review remediation

## Canonical execution path

There is one supported end-to-end command:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
```

It performs Phase 1 ingestion and calls `src.set_forecasting.pipeline.run_model_pipeline()`. The runner prints the imported pipeline module path. If the processed monthly Parquet file already exists, `python run_phase5_8_scmd.py` reruns that same canonical model pipeline without repeating ingestion.

The former `src.scmd_phases_5_8` implementation is retired and retained only under `legacy/` with a deprecation notice. It is not an executable or CI path.

## Existing implementation notes

- Phase 3 uses pooled/global quantile LightGBM models per quantile. If LightGBM is unavailable or the train/validation sample is too small, a deterministic seasonal/last-observation fallback is used.
- Phase 4 defaults are read from `config/config.yaml`; calibration intervals are fixed before the current outcome is appended to residual history.
- Phase 5 reliability-adjusted risk feeds the inventory simulation. Bootstrap resampling is clustered by series and Holm-adjusted results are exported.
- No clinical VEN or substitutability labels are inferred.

## Scientific limitations

**Inventory outcomes are SIMULATED. Stockout labels are not observed events.** SCMD contains no observed inventory or stockout outcomes. The legacy medicine-level pipeline is incompatible with the SCMD schema and must not be used for SCMD evaluation.

Passing unit tests does not establish scientific superiority. Research claims require successful full-cohort execution, held-out calibration, identical-observation comparisons and appropriate statistical evidence.
