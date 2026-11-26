"""
src/reliability/enbpi.py
========================
Ensemble Batch Prediction Intervals (EnbPI)
Spec Section 7.9:
  - Online sequential conformal prediction using sliding non-conformity scores (residuals)
  - PI_t = [y_hat - Q_{1-alpha}(R_t), y_hat + Q_{1-alpha}(R_t)]
"""

import numpy as np

class EnbPI:
    def __init__(self, target_coverage: float = 0.90, window: int = 12):
        self.target_coverage = target_coverage
        self.window = window
        self.residual_buffer = []

    def get_interval(self, y_hat: float, fallback_radius: float) -> tuple:
        if len(self.residual_buffer) < 3:
            return max(0.0, y_hat - fallback_radius), y_hat + fallback_radius

        # Out-of-sample empirical nonconformity quantile
        scores = np.sort(self.residual_buffer)
        idx = int(np.ceil(self.target_coverage * len(scores))) - 1
        q_val = float(scores[min(idx, len(scores) - 1)])

        return max(0.0, y_hat - q_val), y_hat + q_val

    def update(self, y_true: float, y_hat: float):
        e_t = abs(y_true - y_hat)
        self.residual_buffer.append(e_t)
        if len(self.residual_buffer) > self.window:
            self.residual_buffer.pop(0)
