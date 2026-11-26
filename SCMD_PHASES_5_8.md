# SCMD Phases 5–8 — canonical RADCSR pipeline

Run the complete supported workflow from the repository root:

```bash
python scripts/run_pipeline.py --source data/raw/combined_secondary_care_medicines.csv
```

The command performs Phase 1 ingestion and then invokes exactly one model orchestrator: `src.set_forecasting.pipeline.run_model_pipeline()`. Its runner prints the imported module path as an execution trace. Do not run the retired `src.scmd_phases_5_8` implementation.

If Phase 1 has already produced `data/processed/scmd_monthly_series.parquet`, the model pipeline can be rerun directly:

```bash
python run_phase5_8_scmd.py
```

Both commands execute the same canonical model pipeline. The second skips ingestion; it does not select a separate Phase 5–8 implementation.

## Pipeline stages

5. Reliability-adjusted RADCSR risk using the four-signal reliability index and monotonicity diagnostics.
6. Nine synthetic scenarios: stable, surge, drift, volatility, bias, lead-time shock, combined, intermittent demand and correlated surge; monthly periodic-review inventory simulation.
7. RADCSR comparisons against configured risk baselines and ablations on the generated evaluation streams.
8. Series-clustered bootstrap, paired statistical testing and Holm adjustment.

## Research limitation

**Inventory outcomes are SIMULATED. Stockout labels are not observed events.** SCMD has no stock-on-hand, receipts, purchase orders, supplier lead times or observed stockout data. Monte Carlo-style and Wasserstein-style methods are reference approximations, not full optimisation solvers.

## Interpretation and outputs

The adjusted-distribution tail and the monotone RADCSR uplift score are distinct quantities. The uplift score must not be interpreted as a calibrated probability unless validation-period calibration and held-out evaluation support that interpretation. The canonical pipeline records its resolved configuration and writes reports under `output/` and intermediate Parquet files under `data/processed/`.

The active reliability field is `R_UNRELIABILITY`; higher values mean worse forecast reliability. `R_REL` remains a compatibility alias for the RADCSR interface. Clinical/VEN analysis is omitted unless a provenance-complete, cohort-complete mapping is supplied.
