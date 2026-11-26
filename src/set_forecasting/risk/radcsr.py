"""RADCSR: reliability-adaptive demand-conditioned shortage risk."""
import numpy as np
import pandas as pd
from scipy.special import ndtr

KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events. SCMD contains no observed stock or supplier lead times."


def infer_lead_time_distribution(q10, q50, q90, lead):
    q10, q50, q90 = [np.maximum(0, np.asarray(v, dtype=float)) for v in (q10, q50, q90)]
    if any(np.any(~np.isfinite(v)) for v in (q10, q50, q90, np.asarray(lead, dtype=float))):
        raise ValueError("Quantiles and lead time must be finite")
    if np.any(q10 < 0) or np.any(q10 > q50) or np.any(q50 > q90):
        raise ValueError("Require 0 <= Q10 <= Q50 <= Q90")
    lead = np.asarray(lead, dtype=float)
    if np.any(lead <= 0):
        raise ValueError("Lead time must be strictly positive")
    mu = q50 * lead
    sigma = np.maximum((q90 - q10) / 2.563, 1e-6) * np.sqrt(lead)
    return mu, sigma


def _tail(mu, sigma, availability):
    return np.clip(1.0 - ndtr((np.asarray(availability) - mu) /
                              np.maximum(sigma, 1e-8)), 0.0, 1.0)


def _adjusted_distribution(q10, q50, q90, availability, lead, reliability,
                           kappa_sigma, kappa_mu, kappa_l, horizon_cap):
    """Apply conditional, availability-capped reliability adjustments.

    If A < mu_L, mean uplift and horizon extension are active, but the
    effective lead-time mean is capped at A. If A >= mu_L, both are disabled:
    only sigma_L is widened. This prevents mean overshoot from reversing the
    adjusted-distribution tail as reliability worsens.
    """
    mu, sigma = infer_lead_time_distribution(q10, q50, q90, lead)
    lead = np.asarray(lead, dtype=float)
    reliability = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    availability = np.asarray(availability, dtype=float)
    low_availability = availability < mu

    sigma_adj = sigma * (1.0 + kappa_sigma * reliability)
    # The specified cap is applied to the mean itself, before horizon scaling.
    mu_adj = np.where(
        low_availability,
        np.minimum(mu * (1.0 + kappa_mu * reliability), availability),
        mu,
    )
    # Horizon extension is conditional on A < mu_L only.
    lead_eff = np.where(
        low_availability,
        lead * (1.0 + kappa_l * reliability),
        lead,
    )
    if horizon_cap is not None:
        if float(horizon_cap) < 1.0:
            raise ValueError("horizon_cap must be >= 1 or None")
        lead_eff = np.minimum(lead_eff, float(horizon_cap) * lead)
    ratio = lead_eff / lead
    # Also cap the mean contribution from horizon extension at availability.
    mu_eff = np.where(low_availability, np.minimum(mu_adj * ratio, availability), mu)
    sigma_eff = sigma_adj * np.sqrt(ratio)
    return (mu_eff, sigma_eff, mu, sigma, sigma_adj, mu_adj, lead_eff)


def radcsr_probability(q10, q50, q90, availability, lead_time, r_rel,
                       kappa_sigma=0.75, kappa_mu=0.50, kappa_l=1.5,
                       horizon_cap=2.0, grid_size=101):
    """Return a monotone reliability-conditioned risk score in [0, 1].

    RADCSR_DISTRIBUTION_PROBABILITY is the adjusted normal tail. Its mean
    shift and horizon extension operate only when A < mu_L; the mean is capped
    at availability both before and after horizon scaling. When A >= mu_L,
    only sigma widens. This removes the mean-overshoot reversal observed at
    commit e0826409.

    RADCSR_PROBABILITY is a separate monotone uplift toward 0.5:
    p_adj + (0.5 - p_adj) * (1 - exp(-S)), where
    S=log(1+kappa_sigma*R_rel)
      + I(A<mu_L)*(log(1+kappa_mu*R_rel)+log(L_eff/L)).
    It is not the adjusted-distribution tail. Validation-only calibration
    and held-out evaluation are required before probability interpretation.
    """
    values = [np.asarray(v, dtype=float) for v in
              (q10, q50, q90, availability, lead_time, r_rel)]
    q10, q50, q90, availability, lead_time, reliability = np.broadcast_arrays(*values)
    if any(np.any(~np.isfinite(v)) for v in (q10, q50, q90, availability, lead_time, reliability)):
        raise ValueError("RADCSR inputs must be finite")
    if np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("R_REL must be within [0, 1]")
    coefficients = np.asarray([kappa_sigma, kappa_mu, kappa_l], dtype=float)
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0):
        raise ValueError("RADCSR coefficients must be finite and non-negative")
    if horizon_cap is not None and (not np.isfinite(float(horizon_cap)) or float(horizon_cap) < 1.0):
        raise ValueError("horizon_cap must be finite and >= 1 or None")
    if np.any(availability < 0):
        raise ValueError("Availability must be non-negative")
    mu_eff, sigma_eff, mu, sigma, sigma_adj, mu_adj, lead_eff = _adjusted_distribution(
        q10, q50, q90, availability, lead_time, reliability,
        kappa_sigma, kappa_mu, kappa_l, horizon_cap)
    baseline = _tail(mu, sigma, availability)
    adjusted = _tail(mu_eff, sigma_eff, availability)
    # The adjusted mean is always <= availability in the low-availability
    # regime and equals mu_L in the high-availability regime. Therefore
    # adjusted <= 0.5, and widening sigma moves the tail monotonically toward 0.5.
    low_availability = availability < mu
    lead_ratio = lead_eff / np.maximum(np.asarray(lead_time, dtype=float), 1e-12)
    severity = (np.log1p(kappa_sigma * reliability) +
                low_availability * (
                    np.log1p(kappa_mu * reliability) +
                    np.log(np.maximum(lead_ratio, 1.0))
                ))
    uplift_fraction = 1.0 - np.exp(-severity)
    # Separate monotone uplift: interpolate the adjusted tail toward its 0.5
    # ceiling. If adjusted is 0.5 (A < mu_L under the specified cap), it stays
    # 0.5. Otherwise both the adjusted tail and uplift fraction are nondecreasing.
    probability = adjusted + (0.5 - adjusted) * uplift_fraction
    probability = np.clip(probability, 0.0, 0.5)
    raw = np.asarray(adjusted, dtype=float)
    return pd.DataFrame({
        "MU_L": mu.reshape(-1), "SIGMA_L": sigma.reshape(-1),
        "SIGMA_ADJ": sigma_adj.reshape(-1), "MU_ADJ": mu_adj.reshape(-1),
        "MU_L_EFFECTIVE": mu_eff.reshape(-1), "L_EFF": lead_eff.reshape(-1),
        "RADCSR_DISTRIBUTION_PROBABILITY": raw.reshape(-1),
        "RADCSR_BASE_PROBABILITY": baseline.reshape(-1),
        "RADCSR_RAW_PROBABILITY": probability.reshape(-1),
        "RADCSR_PROBABILITY": probability.reshape(-1),
        "R_REL": reliability.reshape(-1),
        "STOCKOUTS_ARE_OBSERVED": 0, "OUTCOME_TYPE": "SIMULATED"
    })


def radcsr_ablations(frame, kappa_sigma=0.75, kappa_mu=0.50, kappa_l=1.5,
                     horizon_cap=2.0, weights=None, clinical_mapping=None):
    """Evaluate component ablations; clinical ablation requires verified VEN data."""
    required = KEYS + ["Q10", "Q50", "Q90", "AVAILABLE_STOCK",
                       "LEAD_TIME_MONTHS", "R_E", "R_B", "R_C", "R_D", "R_REL"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing ablation columns: {sorted(missing)}")
    x = frame.copy()
    signals = {"error": x.R_E.to_numpy(float), "bias": x.R_B.to_numpy(float),
               "coverage": x.R_C.to_numpy(float), "drift": x.R_D.to_numpy(float)}
    w = weights or {"error": .35, "bias": .25, "coverage": .25, "drift": .15}
    if set(w) != set(signals) or any(float(v) < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("Weights must define non-negative error, bias, coverage and drift weights")
    w = {k: float(v) / sum(w.values()) for k, v in w.items()}

    def without(signal):
        remaining = {k: v for k, v in w.items() if k != signal}
        denominator = sum(remaining.values())
        if denominator <= 0:
            raise ValueError(f"Cannot remove the only active signal: {signal}")
        return sum(remaining[k] * signals[k] for k in remaining) / denominator

    variants = {
        "RADCSR": (x.R_REL.to_numpy(float), kappa_sigma, kappa_mu, kappa_l, horizon_cap),
        "RADCSR_noBias": (without("bias"), kappa_sigma, kappa_mu, kappa_l, horizon_cap),
        "RADCSR_noCoverage": (without("coverage"), kappa_sigma, kappa_mu, kappa_l, horizon_cap),
        "RADCSR_noDrift": (without("drift"), kappa_sigma, kappa_mu, kappa_l, horizon_cap),
        "RADCSR_noCap": (x.R_REL.to_numpy(float), kappa_sigma, kappa_mu, kappa_l, None),
        "RADCSR_noHorizon": (x.R_REL.to_numpy(float), kappa_sigma, kappa_mu, 0.0, horizon_cap),
        "RADCSR_singleSignal": (signals["error"], kappa_sigma, kappa_mu, kappa_l, horizon_cap),
    }
    if "R_B_SYMMETRIC" in x.columns:
        symmetric_r = np.clip(
            w["error"] * signals["error"] +
            w["bias"] * x["R_B_SYMMETRIC"].to_numpy(float) +
            w["coverage"] * signals["coverage"] +
            w["drift"] * signals["drift"], 0.0, 1.0)
        variants["RADCSR_symmetricBias"] = (
            symmetric_r, kappa_sigma, kappa_mu, kappa_l, horizon_cap)

    if clinical_mapping is not None:
        from .clinical import REQUIRED as VEN_REQUIRED
        if not set(VEN_REQUIRED).issubset(clinical_mapping.columns):
            raise ValueError("Clinical ablation requires a verified, provenance-complete VEN mapping")
        mapped = x.merge(clinical_mapping, on="VMP_SNOMED_CODE", how="left", validate="many_to_one")
        if mapped.VEN_CLASS.isna().any():
            raise ValueError("Clinical ablation requires complete VEN mapping for the evaluated cohort")
        clinical_factor = mapped.VEN_CLASS.map({"V": 1.0, "E": .67, "N": .33}).to_numpy(float)
        clinical_r = np.clip(x.R_REL.to_numpy(float) * clinical_factor, 0, 1)
        variants["RADCSR_noClinical"] = (x.R_REL.to_numpy(float), kappa_sigma, kappa_mu, kappa_l, horizon_cap)
        variants["RADCSR_clinical"] = (clinical_r, kappa_sigma, kappa_mu, kappa_l, horizon_cap)

    outputs = []
    for name, (r, ks, km, kl, cap) in variants.items():
        result = radcsr_probability(x.Q10, x.Q50, x.Q90, x.AVAILABLE_STOCK,
                                    x.LEAD_TIME_MONTHS, r, ks, km, kl, cap)
        outputs.append(pd.DataFrame({"METHOD": name,
                                     "SHORTAGE_PROBABILITY": result.RADCSR_PROBABILITY}))
    return pd.concat(outputs, ignore_index=True)


def monotonicity_audit(q10, q50, q90, availability, lead_time,
                       ks=.75, km=.50, kl=1.5, horizon_cap=2.0):
    grid = np.linspace(0.0, 1.0, 101)
    result = radcsr_probability(np.full(101, q10), np.full(101, q50),
        np.full(101, q90), np.full(101, availability), np.full(101, lead_time),
        grid, ks, km, kl, horizon_cap)
    curve = result.RADCSR_PROBABILITY.to_numpy()
    steps = np.diff(curve)
    passed = bool(np.all(steps >= -1e-10))
    return {"raw_pass": passed, "envelope_pass": passed,
            "minimum_step": float(steps.min()), "probability_curve": curve.tolist(),
            "gate_status": "PASS" if passed else "FAIL",
            "method": "analytic_monotone_risk_uplift_no_posthoc_envelope"}


def block_bootstrap_lead_time_samples(q50_path, residual_history, lead, n_boot=500,
                                      block_length=None, seed=42):
    """Simulate lead-time demand from historical forecast-error blocks.

    q50_path contains one-step forecasts for the future lead horizon.
    residual_history must contain errors observed strictly before the forecast origin.
    Circular moving blocks preserve short-range monthly dependence; no future
    validation/test residual is admitted into the bootstrap pool.
    """
    forecasts = np.maximum(0.0, np.asarray(q50_path, dtype=float).reshape(-1))
    errors = np.asarray(residual_history, dtype=float).reshape(-1)
    if int(lead) < 1 or len(forecasts) < int(lead):
        raise ValueError("q50_path must contain at least lead finite forecasts")
    if len(errors) < 2 or np.any(~np.isfinite(errors)) or np.any(~np.isfinite(forecasts)):
        raise ValueError("At least two finite historical residuals and finite forecasts are required")
    lead = int(lead)
    if block_length is None:
        block_length = max(1, int(round(len(errors) ** (1.0 / 3.0))))
    block_length = min(max(1, int(block_length)), len(errors))
    rng = np.random.default_rng(seed)
    n_boot = int(n_boot)
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20")
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        path = []
        while len(path) < lead:
            start = int(rng.integers(0, len(errors)))
            path.extend(errors[(start + np.arange(block_length)) % len(errors)].tolist())
        samples[b] = np.maximum(0.0, forecasts[:lead] + np.asarray(path[:lead])).sum()
    return samples
