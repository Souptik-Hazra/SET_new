# SCMD Phases 5–8

Run after Phases 1–4: `python run_phase5_8_scmd.py`.

5. Reliability-adjusted RADCSR risk radius and monotonicity sweep over reliability 0.00–1.00 in 0.05 increments.
6. Stable, surge, drift, volatility, bias, lead-time-shock and combined scenarios with monthly periodic-review inventory simulation.
7. Static, rolling-quantile, EWMA, ACI, Monte Carlo-style and Wasserstein-style interval baselines on identical synthetic streams.
8. Series-level paired bootstrap (2,000 resamples) and Holm adjustment on interval-score comparisons.

**Research limitation:** Inventory outcomes are SIMULATED. Stockout labels are not observed events. SCMD has no stock-on-hand, receipts, purchase orders, supplier lead times or observed stockout data. Monte Carlo and Wasserstein-style baselines are simplified sensitivity references, not full optimisation solvers.


## Scenario implementation status

The active generator includes stable, surge, gradual drift, volatility, forecast-bias, lead-time shock, combined, intermittent-demand, and correlated-surge streams. Each series/scenario receives a deterministic independent random stream derived from the global seed and series key. Scenario metadata records the shift start index and shift magnitude. These remain synthetic stress tests, not observed hospital demand regimes.


## Research implementation notes (current)

- Phase 4 reports an analytic monotone reliability risk-uplift probability and enforces a held-out test acceptance gate across one representative case per series/scenario. There is no post-hoc cumulative-maximum envelope. The adjusted distribution tail is exported separately: the uplift probability is a monotone risk score/probability transform, not the unmodified tail probability of that adjusted distribution. Validation-only isotonic calibration is still required before interpreting it probabilistically.
- Reliability severity includes the capped effective-horizon ratio. The no-cap and no-horizon ablations therefore change the evaluated risk probability; tests guard against accidentally identical ablations.
- Phase 6 exports the distribution-based baseline methods and simulated labels separately. Monte Carlo and Wasserstein remain reference approximations; they are not claimed to be a full DRO optimisation solver.
- Phase 7 reports only ablations that are scientifically defined from available inputs. A clinical/VEN ablation is omitted unless a provenance-complete, cohort-complete mapping is supplied; it is never fabricated from medicine names or demand.
- Phase 8 resamples organisation × VMP × unit series as clusters, computes paired percentile-bootstrap confidence intervals, uses a paired sign-flip randomisation test, and applies one global Holm step-down correction across the scenario/method family.
- The final pipeline report records the monotonicity gate, scenario reliability/risk summaries, implemented ablations, statistical protocol, and simulation disclaimer.

These code-level checks do not establish external validity. All inventory, lead-time and stockout labels remain simulated because SCMD has no observed inventory or stockout outcomes.
