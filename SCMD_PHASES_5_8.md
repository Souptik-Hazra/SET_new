# SCMD Phases 5–8

Run after Phases 1–4: `python run_phase5_8_scmd.py`.

5. Reliability-adjusted RADCSR risk radius and monotonicity sweep over reliability 0.00–1.00 in 0.05 increments.
6. Stable, surge, drift, volatility, bias, lead-time-shock and combined scenarios with monthly periodic-review inventory simulation.
7. Static, rolling-quantile, EWMA, ACI, Monte Carlo-style and Wasserstein-style interval baselines on identical synthetic streams.
8. Series-level paired bootstrap (2,000 resamples) and Holm adjustment on interval-score comparisons.

**Research limitation:** Inventory outcomes are SIMULATED. Stockout labels are not observed events. SCMD has no stock-on-hand, receipts, purchase orders, supplier lead times or observed stockout data. Monte Carlo and Wasserstein-style baselines are simplified sensitivity references, not full optimisation solvers.
