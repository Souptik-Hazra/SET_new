"""
src/risk/scenario_risk.py
=========================
Phase 4(a): Monte Carlo Scenario Risk Baseline
Strict Conformance with Prompt & Spec 8.2:
  - Block bootstrap sampling (block length b=7) from past residuals (up to 52 periods)
  - 1000 simulated lead-time demand trajectories
  - Fraction of paths exceeding available stock = p_mc
"""

import numpy as np

def compute_mc_scenario_risk(recent_residuals: np.ndarray,
                             mu_1: float,
                             L: int,
                             avail: float,
                             n_paths: int = 1000,
                             block_len: int = 7) -> float:
    """
    Draw 1000 paths by block bootstrap from past residuals, block length 7.
    Simulate lead-time demand and evaluate empirical stockout probability.
    """
    res = np.asarray(recent_residuals, dtype=np.float64)
    if len(res) == 0:
        return 1.0 if (L * mu_1 > avail) else 0.0

    if len(res) < block_len:
        res = np.pad(res, (0, max(0, block_len - len(res))), mode='edge')

    n_res = len(res)
    max_start = max(1, n_res - block_len + 1)

    # Fast block bootstrap
    sampled_demands = np.empty(n_paths, dtype=np.float64)
    for i in range(n_paths):
        path = []
        while len(path) < L:
            idx = np.random.randint(0, max_start)
            path.extend(res[idx : idx + block_len])
        sampled_demands[i] = np.sum(np.maximum(0.0, mu_1 + np.array(path[:L])))

    p_mc = float(np.mean(sampled_demands > avail))
    return float(np.clip(p_mc, 0.0, 1.0))
