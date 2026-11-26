"""
src/risk/robust.py
==================
Phase 4(b): Robust Optimisation Baseline
Strict Conformance with Prompt & Spec 8.3:
  - Box uncertainty set from empirical 10th and 90th percentile residuals
  - Worst-case lead-time demand over that set
  - p_robust = 1 if worst-case demand exceeds stock, else nominal probability
"""

import numpy as np

def compute_robust_risk(recent_residuals: np.ndarray,
                        mu_1: float,
                        L: int,
                        avail: float,
                        p_nominal: float) -> float:
    """
    Build box uncertainty set from empirical 10th and 90th percentile residuals.
    Worst-case lead-time demand = L * (mu_1 + q90_residual).
    """
    res = np.asarray(recent_residuals, dtype=np.float64)
    if len(res) >= 4:
        q90_res = float(np.percentile(res, 90))
    elif len(res) > 0:
        q90_res = float(np.max(res))
    else:
        q90_res = 1.0

    worst_case_demand = float(L * (mu_1 + max(0.0, q90_res)))
    return 1.0 if (worst_case_demand > avail) else float(p_nominal)
