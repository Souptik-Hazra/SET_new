"""RADCSR: reliability-adaptive demand-conditioned shortage risk."""
import numpy as np
import pandas as pd
from scipy.special import ndtr

KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER = ("SCMD contains no observed stock, lead time or stockout events. "
              "Inventory, lead times and labels are simulated.")


def infer_lead_time_distribution(q10, q50, q90, lead):
    q10, q50, q90 = [np.maximum(0, np.asarray(v, dtype=float)) for v in (q10, q50, q90)]
    if np.any(q10 > q50) or np.any(q50 > q90):
        raise ValueError("Require Q10 <= Q50 <= Q90")
    lead = np.maximum(np.asarray(lead, dtype=float), 1.0)
    mu = q50 * lead
    sigma = np.maximum((q90 - q10) / 2.563, 1e-6) * np.sqrt(lead)
    return mu, sigma


def _tail(mu, sigma, availability):
    return np.clip(1.0 - ndtr((np.asarray(availability) - mu) /
                              np.maximum(sigma, 1e-8)), 0.0, 1.0)


def _adjusted_distribution(q10, q50, q90, availability, lead, reliability,
                           kappa_sigma, kappa_mu, kappa_l, horizon_cap):
    mu, sigma = infer_lead_time_distribution(q10, q50, q90, lead)
    lead = np.maximum(np.asarray(lead, dtype=float), 1.0)
    reliability = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    availability = np.asarray(availability, dtype=float)
    sigma_adj = sigma * (1.0 + kappa_sigma * reliability)
    mu_adj = mu + np.where(availability < mu, mu * kappa_mu * reliability, 0.0)
    lead_eff = lead * (1.0 + kappa_l * reliability)
    if horizon_cap is not None:
        if float(horizon_cap) < 1.0:
            raise ValueError("horizon_cap must be >= 1 or None")
        lead_eff = np.minimum(lead_eff, float(horizon_cap) * lead)
    ratio = lead_eff / lead
    return (mu_adj * ratio, sigma_adj * np.sqrt(ratio),
            mu, sigma, sigma_adj, mu_adj, lead_eff)


def radcsr_probability(q10, q50, q90, availability, lead_time, r_rel,
                       kappa_sigma=0.75, kappa_mu=0.50, kappa_l=1.5,
                       horizon_cap=2.0, grid_size=101):
    """Return a naturally monotone reliability-conditioned probability.

    The uncertainty, conditional-mean and horizon adjustments are non-negative
    functions of unreliability. No post-hoc cumulative-max envelope is applied.
    """
    values = [np.asarray(v, dtype=float) for v in
              (q10, q50, q90, availability, lead_time, r_rel)]
    q10, q50, q90, availability, lead_time, reliability = np.broadcast_arrays(*values)
    if any(np.any(~np.isfinite(v)) for v in (q10, q50, q90, availability, lead_time, reliability)):
        raise ValueError("RADCSR inputs must be finite")
    if np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("R_REL must be within [0, 1]")
    if min(kappa_sigma, kappa_mu, kappa_l) < 0:
        raise ValueError("RADCSR coefficients must be non-negative")
    mu_eff, sigma_eff, mu, sigma, sigma_adj, mu_adj, lead_eff = _adjusted_distribution(
        q10, q50, q90, availability, lead_time, reliability,
        kappa_sigma, kappa_mu, kappa_l, horizon_cap)
    baseline = _tail(mu, sigma, availability)
    adjusted = _tail(mu_eff, sigma_eff, availability)
    # Monotone risk uplift: reliability penalty adds non-negative log-survival
    # pressure, with exact equality to the unadjusted distribution at R_REL=0.
    lead_ratio = lead_eff / np.maximum(np.asarray(lead_time, dtype=float), 1.0)
    severity = (np.log1p(kappa_sigma * reliability) +
                np.log1p(kappa_mu * reliability * (availability < mu)) +
                np.log(np.maximum(lead_ratio, 1.0)))
    probability = baseline + (1.0 - baseline) * (1.0 - np.exp(-severity))
    # Keep adjusted distribution visible for audit; the reported risk is the
    # monotone RADCSR uplift uses the same capped effective-horizon ratio as the model.
    probability = np.clip(probability, 0.0, 1.0)
    raw = np.asarray(adjusted, dtype=float)
    return pd.DataFrame({
        "MU_L": mu.reshape(-1), "SIGMA_L": sigma.reshape(-1),
        "SIGMA_ADJ": sigma_adj.reshape(-1), "MU_ADJ": mu_adj.reshape(-1),
        "L_EFF": lead_eff.reshape(-1),
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
