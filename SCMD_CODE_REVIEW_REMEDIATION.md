# SCMD full repository review remediation

The SCMD workflow is the supported path:
1. `python run_phase1_scmd.py`
2. `python run_phase2_scmd.py`
3. `python run_phase3_scmd.py`
4. `python run_phase4_scmd.py`
5. `python run_phase5_8_scmd.py`

Phase 3 now uses one pooled/global quantile LightGBM model per quantile, rather than fitting three models for every individual series. If LightGBM is unavailable or the train/validation sample is too small, it writes a deterministic seasonal/last-observation fallback prediction set. The test split is evaluated as a rolling one-step-ahead stream: earlier test outcomes can inform later forecasts only after they would have become observable.

Phase 4 defaults are read from `config/config.yaml` (90% target coverage, 8-month history). Calibration intervals are fixed before the current outcome is appended to the residual history.

Phase 5 risk radius is passed into Phase 6 replenishment targets. Bootstrap resampling is clustered by series after aggregating scenarios, and Holm-adjusted p-values and confidence intervals are exported.

**Data validity:** no clinical VEN or substitutability labels are inferred. The legacy medicine-level pipeline is disabled because its input schema is incompatible with SCMD. Inventory outcomes are SIMULATED. Stockout labels are not observed events.

The full dataset and all integration tests must still be run locally before research claims are made.
