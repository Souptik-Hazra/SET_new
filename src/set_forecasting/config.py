"""Single source of truth for SETReliable SCMD runtime configuration.

Edit values here instead of maintaining a separate YAML configuration file.
Call get_config() for an isolated copy so callers cannot mutate global defaults.
"""
from copy import deepcopy

CONFIG = {
    "seed": 42,
    "data": {
        "source": "data/raw/combined_secondary_care_medicines.csv",
        "min_periods": 18,
        "min_nonzero": 6,
        "train_frac": 0.60,
        "validation_frac": 0.20,
        "test_frac": 0.20,
        "winsor_quantile": 0.995,
    },
    "forecast": {
        "model": "quantile_lightgbm",
        "quantiles": (0.10, 0.50, 0.90),
        "seasonal_period": 12,
    },
    "reliability": {
        "target_coverage": 0.90,
        "window": 8,
        "weights": {"error": 0.35, "bias": 0.25, "coverage": 0.25, "drift": 0.15},
    },
    "radcsr": {
        "kappa_sigma": 0.50,
        "kappa_mu": 0.75,
        "kappa_l": 1.50,
        "horizon_cap": 2.0,
        "monotonicity_gate": True,
        "monotonicity_tolerance": 1e-10,
        "probability_adjustment": "strictly_monotone_dual_channel_risk_uplift",
    },
    "inventory": {
        "outcomes_are_simulated": True,
        "initial_cover_months": 2,
        "target_cover_months": 3,
        "lead_time_months": (1, 2, 3),
        "service_level": 0.95,
    },
    "evaluation": {
        "bootstrap_resamples": 2000,
        "bootstrap_unit": "series",
        "calibration_bins": 10,
    },
}


def get_config():
    """Return a deep copy of the canonical configuration."""
    return deepcopy(CONFIG)


def validate_config(config=None):
    """Validate configuration invariants before pipeline execution."""
    cfg = get_config() if config is None else deepcopy(config)
    data = cfg["data"]
    if int(data["min_periods"]) < 2 or int(data["min_nonzero"]) < 1:
        raise ValueError("Cohort thresholds must be positive and min_periods >= 2")
    if not 0 < float(data["train_frac"]) < 1:
        raise ValueError("train_frac must be between zero and one")
    if not 0 < float(data["validation_frac"]) < 1:
        raise ValueError("validation_frac must be between zero and one")
    if not 0 < float(data["test_frac"]) < 1:
        raise ValueError("test_frac must be between zero and one")
    if not abs(float(data["train_frac"]) + float(data["validation_frac"]) + float(data["test_frac"]) - 1.0) <= 1e-9:
        raise ValueError("train_frac + validation_frac + test_frac must sum to 1.0")
    rel = cfg["reliability"]
    if not 0 < float(rel["target_coverage"]) < 1 or int(rel["window"]) < 1:
        raise ValueError("Invalid reliability coverage/window")
    risk = cfg["radcsr"]
    ks, km = float(risk["kappa_sigma"]), float(risk["kappa_mu"])
    if ks <= 0 or km < ks or float(risk["kappa_l"]) <= 0:
        raise ValueError("RADCSR requires kappa_sigma > 0, kappa_mu >= kappa_sigma and kappa_l > 0")
    if float(risk["horizon_cap"]) < 1:
        raise ValueError("RADCSR horizon_cap must be >= 1")
    inv = cfg["inventory"]
    if not 0 < float(inv["service_level"]) < 1:
        raise ValueError("Inventory service_level must be between zero and one")
    if not inv["lead_time_months"] or any(int(x) < 1 for x in inv["lead_time_months"]):
        raise ValueError("Inventory lead times must be positive")
    if int(cfg["evaluation"]["bootstrap_resamples"]) < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    return cfg
