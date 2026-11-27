"""Prequential four-signal forecast-unreliability index.

All signals are bounded to [0, 1], where larger means worse reliability.
R_E = clip(mean(|e|) / scale, 0, 1).
R_B_SYMMETRIC = clip(2 * (sigmoid(2*|mean(e)|/scale)-0.5), 0, 1).
R_B (shortage-relevant) = clip(2 * (sigmoid(2*max(mean(e), 0)/scale)-0.5), 0, 1),
where e = actual - Q50. Positive e is under-forecasting and is penalised;
negative e is separately exposed as R_B_OVER and is not folded into shortage risk.
R_C = clip(|prior empirical interval coverage - target| / (1-target), 0, 1).
R_D is a two-sided Page-style CUSUM alarm: max(CUSUM+, |CUSUM-|) > 4.
The composite is the configured weighted sum of R_E, asymmetric R_B, R_C, R_D.
"""
import numpy as np
import pandas as pd
from scipy.special import expit

KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]


def _bias_score(magnitude, scale):
    return np.clip(2 * (expit(2 * np.maximum(magnitude, 0) / scale) - .5), 0, 1)


def build_reliability_index(predictions, target_coverage=.90, window=8, weights=None, min_history=3, period_col="PERIOD"):
    """Compute prequential signals with calendar-aware rolling state.

    A gap greater than one calendar month resets all rolling history, both CUSUM
    accumulators and the shift detector. The first observed row after a gap starts
    cold, so no statistic bridges unobserved calendar months.
    """
    required = KEYS + [period_col, "MODEL_DEMAND", "Q10", "Q50", "Q90"]
    missing = set(required) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing reliability inputs: {sorted(missing)}")
    w = weights or {"error": .35, "bias": .25, "coverage": .25, "drift": .15}
    expected = {"error", "bias", "coverage", "drift"}
    if set(w) != expected or any(not np.isfinite(float(v)) or float(v) < 0 for v in w.values()) or not np.isclose(sum(w.values()), 1):
        raise ValueError("Weights must define error, bias, coverage and drift; be finite, non-negative and sum to one")
    if predictions.duplicated(KEYS + [period_col]).any():
        raise ValueError("Reliability input must have one row per series and period")
    if not 0 < float(target_coverage) < 1 or int(window) < 1 or int(min_history) < 1:
        raise ValueError("target_coverage must be in (0,1); window and min_history must be positive")
    output = []
    for _, g in predictions.groupby(KEYS, sort=False, dropna=False):
        g = g.sort_values(period_col).copy()
        periods = pd.to_datetime(g[period_col], errors="coerce")
        if periods.isna().any():
            raise ValueError(f"Reliability {period_col} values must be valid calendar periods")
        month_ordinals = periods.dt.to_period("M").astype("int64").to_numpy()
        if len(month_ordinals) > 1 and np.any(np.diff(month_ordinals) <= 0):
            raise ValueError("Reliability periods must be strictly increasing within each series")
        y = g.MODEL_DEMAND.to_numpy(float)
        q10, q50, q90 = (g[c].to_numpy(float) for c in ("Q10", "Q50", "Q90"))
        err = y - q50
        ae = np.abs(err)
        n = len(g)
        re = np.zeros(n); rb = np.zeros(n); rbs = np.zeros(n)
        rbo = np.zeros(n); rc = np.zeros(n); rd = np.zeros(n)
        mae = np.full(n, np.nan); bias = np.full(n, np.nan); coverage = np.full(n, np.nan)
        cal_lo = np.zeros(n); cal_hi = np.zeros(n); cal_covered = np.zeros(n, dtype=bool)
        pos = neg = 0.0
        res_history = []
        segment_start = 0
        shift = np.zeros(n, dtype=bool)
        for t in range(n):
            if t and month_ordinals[t] - month_ordinals[t-1] != 1:
                # Calendar gap: deliberately reset all history-dependent state.
                segment_start = t
                pos = neg = 0.0
                res_history.clear()
            base_radius = max((q90[t] - q10[t]) / 2 * 1.282, 1e-6)
            prior_abs = np.asarray(ae[max(segment_start, t-window):t], float)
            radius = float(np.quantile(prior_abs, target_coverage, method="higher")) if len(prior_abs) >= min_history else base_radius
            cal_lo[t] = max(0.0, q50[t] - radius)
            cal_hi[t] = q50[t] + radius
            cal_covered[t] = (y[t] >= cal_lo[t]) and (y[t] <= cal_hi[t])
            if t > segment_start:
                start = max(segment_start, t-window)
                pm = float(np.mean(ae[start:t]))
                pb = float(np.mean(err[start:t]))
                pc = float(np.mean(cal_covered[start:t]))
                scale = max(float(np.median(np.abs(y[start:t]))), float(np.mean(ae[start:t])), 1e-6)
                mae[t] = pm; bias[t] = pb; coverage[t] = pc
                re[t] = np.clip(pm / scale, 0, 1)
                rbs[t] = _bias_score(abs(pb), scale)
                rb[t] = _bias_score(max(pb, 0.0), scale)
                rbo[t] = _bias_score(max(-pb, 0.0), scale)
                rc[t] = np.clip(abs(pc-target_coverage) / max(1-target_coverage, 1e-6), 0, 1)
                sigma = max(float(np.std(res_history[-window:])) if len(res_history) > 1 else 1.0, 1e-6)
                z = err[t-1] / sigma
                pos = max(0.0, pos + z - .5)
                neg = min(0.0, neg + z + .5)
                rd[t] = float(pos > 4 or abs(neg) > 4)
            res_history.append(err[t])
            if t - segment_start >= 2 * window:
                a = err[t-2*window:t-window]; b = err[t-window:t]
                scale = max(float(np.std(np.r_[a, b])), 1e-6)
                shift[t] = abs(float(np.mean(b)-np.mean(a))) > 1.5*scale
        g["CAL_LOWER_90"] = cal_lo; g["CAL_UPPER_90"] = cal_hi
        g["CAL_COVERED_90"] = cal_covered.astype("int8")
        g["R_E"] = re
        g["R_B"] = rb
        g["R_B_SYMMETRIC"] = rbs
        g["R_B_OVER"] = rbo
        g["R_B_UNDER"] = rb
        g["R_C"] = rc; g["R_D"] = rd
        g["R_UNRELIABILITY"] = np.clip(w["error"]*re + w["bias"]*rb + w["coverage"]*rc + w["drift"]*rd, 0, 1)
        g["R_REL"] = g["R_UNRELIABILITY"]
        g["ROLLING_MAE_PRIOR"] = mae; g["ROLLING_BIAS_PRIOR"] = bias
        g["ROLLING_COVERAGE_PRIOR"] = coverage
        g["SHIFT_DETECTOR_FLAG"] = shift
        output.append(g)
    return pd.concat(output, ignore_index=True) if output else predictions.copy()
