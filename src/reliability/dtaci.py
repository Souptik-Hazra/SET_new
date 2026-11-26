"""
src/reliability/dtaci.py
========================
Dynamic Tracking Adaptive Conformal Inference (DtACI)
Per Gibbs & Candès (2024):
  - Online adaptation of the step-size / learning rate gamma_t
  - Higher gamma during detected non-stationary bursts; lower gamma during quiescent periods
"""

import numpy as np

class DtACI:
    def __init__(self, target_coverage: float = 0.90, gamma_min: float = 0.005, gamma_max: float = 0.10):
        self.alpha = 1.0 - target_coverage
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.alpha_t = self.alpha
        self.gamma_t = (gamma_min + gamma_max) / 2.0
        self.err_history = []

    def get_interval(self, y_hat: float, base_radius: float) -> tuple:
        # Scale factor based on accumulated coverage error
        scale_fac = max(0.5, 1.0 + (self.alpha - self.alpha_t))
        rad = max(0.1, base_radius * scale_fac)
        return max(0.0, y_hat - rad), y_hat + rad

    def update(self, y_true: float, lower: float, upper: float):
        err_t = 1.0 if (y_true < lower or y_true > upper) else 0.0
        self.err_history.append(err_t)

        # Dynamic learning rate: look at variance of errors in recent window
        if len(self.err_history) >= 4:
            recent_err = self.err_history[-4:]
            err_var = np.var(recent_err)
            # Increase learning rate if volatility in coverage errors spikes
            self.gamma_t = float(np.clip(self.gamma_min + 2.0 * err_var * (self.gamma_max - self.gamma_min),
                                         self.gamma_min, self.gamma_max))

        # Update nominal miscoverage rate
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma_t * (self.alpha - err_t), 0.01, 0.50))
