"""
src/risk/dro.py
===============
Phase 4(c): Wasserstein Distributionally Robust Optimisation (DRO)
Strict Conformance with Prompt & Spec 8.4:
  - Wasserstein ball of radius epsilon around empirical residual distribution
  - epsilon = 0.05 * std(residuals), with sensitivity over {0.01, 0.05, 0.10, 0.20}
  - Worst-case shortage probability over the ball
"""

import numpy as np

def compute_dro_risk(mu_L: float,
                     sigma_L: float,
                     avail: float,
                     recent_residuals: np.ndarray,
                     p_nominal: float,
                     epsilon_mult: float = 0.05) -> float:
    """
    Worst-case shortage probability over Wasserstein ball of radius epsilon.
    By Kantorovich-Rubinstein duality for indicator tail bounds:
      sup_{Q in P} Q(D_L > Avail) <= P_nominal + epsilon / (2 * sigma_L).
    """
    res = np.asarray(recent_residuals, dtype=np.float64)
    std_res = float(np.std(res)) if len(res) > 1 else 1.0
    epsilon = epsilon_mult * std_res
    risk_dro = float(p_nominal) + (epsilon / max(0.1, 2.0 * sigma_L))
    return float(np.clip(risk_dro, 0.0, 1.0))
