"""
src/evaluation/metrics.py
=========================
Forecast and Risk Evaluation Metrics.
Strict conformance with RADCSR Project Specification (Sections 4.6, 8.5, 10.1, 10.2, 10.3, 10.4, 10.5).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# 1. Point Forecast Metrics
# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%). Skips if sum of actuals == 0."""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    denom = np.sum(np.abs(y_true[mask]))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true[mask] - y_pred[mask])) / denom * 100.0)


# ---------------------------------------------------------------------------
# 2. Prediction Interval & Quantile Metrics
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile q in (0, 1)."""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    e = y_true[mask] - y_pred[mask]
    loss = np.where(e >= 0, q * e, (q - 1.0) * e)
    return float(np.mean(loss))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability."""
    mask = ~np.isnan(lower) & ~np.isnan(upper) & ~np.isnan(y_true)
    if not np.any(mask):
        return np.nan
    covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    return float(np.mean(covered))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width."""
    mask = ~np.isnan(lower) & ~np.isnan(upper)
    if not np.any(mask):
        return np.nan
    return float(np.mean(upper[mask] - lower[mask]))


def coverage_error(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   target: float = 0.90) -> float:
    """Absolute deviation of PICP from target coverage."""
    cov = picp(y_true, lower, upper)
    return float(abs(cov - target)) if not np.isnan(cov) else np.nan


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Continuous Ranked Probability Score (CRPS) under Gaussian assumption."""
    from scipy.stats import norm
    mask = ~np.isnan(y_true) & ~np.isnan(mu) & (sigma > 1e-6)
    if not np.any(mask):
        return np.nan
    y = y_true[mask]
    m = mu[mask]
    s = sigma[mask]
    z = (y - m) / s
    crps = s * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


# ---------------------------------------------------------------------------
# 3. Shortage Probability & Calibration Metrics
# ---------------------------------------------------------------------------
def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Brier score: mean squared error of predicted probabilities."""
    mask = ~np.isnan(y_true) & ~np.isnan(p_pred)
    if not np.any(mask):
        return np.nan
    return float(np.mean((p_pred[mask] - y_true[mask]) ** 2))


def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Binary Cross Entropy / Log Loss."""
    mask = ~np.isnan(y_true) & ~np.isnan(p_pred)
    if not np.any(mask):
        return np.nan
    p = np.clip(p_pred[mask], eps, 1.0 - eps)
    y = y_true[mask]
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for binary events."""
    mask = ~np.isnan(y_true) & ~np.isnan(p_pred)
    y = y_true[mask]
    p = p_pred[mask]
    if len(y) == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_n = len(y)

    for i in range(n_bins):
        bin_mask = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        bin_n = np.sum(bin_mask)
        if bin_n > 0:
            bin_acc = np.mean(y[bin_mask])
            bin_conf = np.mean(p[bin_mask])
            ece += (bin_n / total_n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# 4. Binary Detection Classification Metrics
# ---------------------------------------------------------------------------
def detection_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Computes Recall, Precision, F1, False Alarm Rate (FAR), and Specificity."""
    mask = ~np.isnan(y_true) & ~np.isnan(p_pred)
    y = y_true[mask].astype(int)
    p = p_pred[mask]
    pred = (p >= threshold).astype(int)

    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))

    recall = float(tp / max(1, (tp + fn)) * 100.0)
    precision = float(tp / max(1, (tp + fp)) * 100.0)
    f1 = float(2 * precision * recall / max(1e-5, precision + recall))
    far = float(fp / max(1, (fp + tn)) * 100.0)
    specificity = float(tn / max(1, (tn + fp)) * 100.0)

    return {
        "RECALL_%": recall,
        "PRECISION_%": precision,
        "F1_SCORE": f1,
        "FALSE_ALARM_RATE_%": far,
        "SPECIFICITY_%": specificity,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
    }


# ---------------------------------------------------------------------------
# 5. Inventory Consequence & Severity Metrics (Spec Section 8.5 & 10.5)
# ---------------------------------------------------------------------------
def expected_shortage_quantity(demands_L: np.ndarray, avail_L: np.ndarray) -> float:
    """ESQ: Expected Shortage Quantity over lead time."""
    shortages = np.maximum(0.0, demands_L - avail_L)
    return float(np.mean(shortages))


def cvar_shortage(demands_L: np.ndarray, avail_L: np.ndarray, beta: float = 0.95) -> float:
    """CVaR_beta of Shortage Quantity."""
    shortages = np.maximum(0.0, demands_L - avail_L)
    q_beta = np.quantile(shortages, beta)
    tail = shortages[shortages >= q_beta]
    return float(np.mean(tail)) if len(tail) > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Aggregate Forecast Evaluation Summary
# ---------------------------------------------------------------------------
def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        lower_90: Optional[np.ndarray] = None,
                        upper_90: Optional[np.ndarray] = None,
                        q10_pred: Optional[np.ndarray] = None,
                        q50_pred: Optional[np.ndarray] = None,
                        q90_pred: Optional[np.ndarray] = None) -> dict:
    """Compute all standard forecast baseline metrics."""
    result = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "N": int(len(y_true)),
    }

    if lower_90 is not None and upper_90 is not None:
        result["PICP_90"] = picp(y_true, lower_90, upper_90)
        result["MPIW_90"] = mpiw(lower_90, upper_90)
        result["COVERAGE_ERROR_90"] = coverage_error(y_true, lower_90, upper_90, 0.90)

    if q10_pred is not None:
        result["PINBALL_10"] = pinball_loss(y_true, q10_pred, 0.10)
    if q50_pred is not None:
        result["PINBALL_50"] = pinball_loss(y_true, q50_pred, 0.50)
    if q90_pred is not None:
        result["PINBALL_90"] = pinball_loss(y_true, q90_pred, 0.90)

    return result
