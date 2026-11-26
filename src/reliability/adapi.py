"""
src/reliability/adapi.py
========================
Adaptive Prediction Intervals (AdaPI)
Spec Section 7.8:
  - Direct interval scaling using coverage feedback
  - F_t = g(C_t, L)
  - Variants: Linear, Logarithmic, and HLI
"""

import numpy as np

class AdaPI:
    def __init__(self, target_coverage: float = 0.90, window: int = 8, M: float = 0.5, variant: str = "linear"):
        self.target_coverage = target_coverage
        self.window = window
        self.M = M
        self.variant = variant
        self.coverage_history = []

    def get_scaling_factor(self) -> float:
        if len(self.coverage_history) < 2:
            return 1.0

        recent = self.coverage_history[-self.window:]
        c_t = np.mean(recent)
        l_nom = self.target_coverage * 100.0
        c_pct = c_t * 100.0

        if self.variant == "linear":
            # F_t = 1 - M * (L - 100*C_t) / (L - 100)
            denom = l_nom - 100.0
            if abs(denom) < 1e-5:
                return 1.0
            f_t = 1.0 - self.M * ((l_nom - c_pct) / denom)
        elif self.variant == "log":
            # Logarithmic dampening
            f_t = 1.0 + self.M * np.sign(l_nom - c_pct) * np.log1p(abs(l_nom - c_pct) / 10.0)
        else: # HLI (Heuristic Loss Inversion)
            f_t = max(0.5, 1.0 + 2.0 * (self.target_coverage - c_t))

        return float(np.clip(f_t, 0.5, 3.0))

    def update(self, y_true: float, lower: float, upper: float):
        covered = 1.0 if (y_true >= lower and y_true <= upper) else 0.0
        self.coverage_history.append(covered)
