"""Full-cohort empirical monotonicity audit for saved held-out test observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..risk.radcsr import KEYS, radcsr_probability, infer_lead_time_distribution

REQUIRED = KEYS + [
    "PERIOD", "SCENARIO_ID", "SPLIT", "Q10", "Q50", "Q90",
    "AVAILABLE_STOCK", "LEAD_TIME_MONTHS", "R_REL",
]


def audit_frame(frame, kappa_sigma=0.50, kappa_mu=0.75,
                kappa_l=1.50, horizon_cap=2.0, grid_points=101,
                batch_size=4096):
    """Check both RADCSR probability curves for every supplied test observation."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Empirical monotonicity input missing columns: {missing}")
    data = frame.loc[frame["SPLIT"].astype(str).str.lower() == "test"].copy()
    if data.empty:
        raise ValueError("Empirical monotonicity audit received no held-out test observations")
    identity = KEYS + ["PERIOD", "SCENARIO_ID"]
    if data[identity].isna().any().any():
        raise ValueError("Empirical monotonicity keys contain null values")
    if data.duplicated(identity).any():
        raise ValueError("Empirical monotonicity input contains duplicate observation keys")
    numeric = ["Q10", "Q50", "Q90", "AVAILABLE_STOCK", "LEAD_TIME_MONTHS", "R_REL"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("Empirical monotonicity inputs contain non-finite values")
    if not 2 <= int(grid_points):
        raise ValueError("grid_points must be at least 2")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")

    grid = np.linspace(0.0, 1.0, int(grid_points))
    rows = []
    for start in range(0, len(data), int(batch_size)):
        batch = data.iloc[start:start + int(batch_size)].reset_index(drop=True)
        n = len(batch)
        expanded = {
            col: np.repeat(batch[col].to_numpy(float), len(grid))
            for col in numeric if col != "R_REL"
        }
        expanded["R_REL"] = np.tile(grid, n)
        result = radcsr_probability(
            expanded["Q10"], expanded["Q50"], expanded["Q90"],
            expanded["AVAILABLE_STOCK"], expanded["LEAD_TIME_MONTHS"],
            expanded["R_REL"], kappa_sigma=float(kappa_sigma),
            kappa_mu=float(kappa_mu), kappa_l=float(kappa_l),
            horizon_cap=horizon_cap,
        )
        mu, _ = infer_lead_time_distribution(
            batch["Q10"].to_numpy(float), batch["Q50"].to_numpy(float),
            batch["Q90"].to_numpy(float), batch["LEAD_TIME_MONTHS"].to_numpy(float),
        )
        low = batch["AVAILABLE_STOCK"].to_numpy(float) < np.asarray(mu, dtype=float)
        for source, target in (
            ("RADCSR_DISTRIBUTION_PROBABILITY", "distribution"),
            ("RADCSR_PROBABILITY", "radcsr"),
        ):
            curves = result[source].to_numpy(float).reshape(n, len(grid))
            passed = np.all(np.diff(curves, axis=1) > 0.0, axis=1)
            batch[target + "_pass"] = passed
            batch[target + "_minimum_increment"] = np.diff(curves, axis=1).min(axis=1)
        batch["LOW_AVAILABILITY"] = low
        rows.append(batch[identity + [
            "LOW_AVAILABILITY", "distribution_pass", "radcsr_pass",
            "distribution_minimum_increment", "radcsr_minimum_increment",
        ]])

    checked = pd.concat(rows, ignore_index=True)
    scenarios = {}
    for scenario, group in checked.groupby("SCENARIO_ID", sort=True, dropna=False):
        scenarios[str(scenario)] = {
            "observations": int(len(group)),
            "distribution_pass": int(group.distribution_pass.sum()),
            "distribution_fail": int((~group.distribution_pass).sum()),
            "radcsr_pass": int(group.radcsr_pass.sum()),
            "radcsr_fail": int((~group.radcsr_pass).sum()),
            "low_availability_observations": int(group.LOW_AVAILABILITY.sum()),
            "low_availability_failures": int((
                group.LOW_AVAILABILITY &
                (~group.distribution_pass | ~group.radcsr_pass)
            ).sum()),
        }

    distribution_fail = checked.loc[~checked.distribution_pass]
    radcsr_fail = checked.loc[~checked.radcsr_pass]
    low_fail = checked.loc[
        checked.LOW_AVAILABILITY &
        (~checked.distribution_pass | ~checked.radcsr_pass)
    ]
    violations = checked.loc[
        ~checked.distribution_pass | ~checked.radcsr_pass
    ].copy()
    violation_keys = violations[identity].astype(str).to_dict(orient="records")
    return {
        "status": "PASS" if violations.empty else "FAIL",
        "total_test_observations": int(len(checked)),
        "distribution_pass": int(checked.distribution_pass.sum()),
        "distribution_fail": int(len(distribution_fail)),
        "radcsr_pass": int(checked.radcsr_pass.sum()),
        "radcsr_fail": int(len(radcsr_fail)),
        "both_pass": int((checked.distribution_pass & checked.radcsr_pass).sum()),
        "both_fail": int((~checked.distribution_pass | ~checked.radcsr_pass).sum()),
        "low_availability_observations": int(checked.LOW_AVAILABILITY.sum()),
        "low_availability_failures": int(len(low_fail)),
        "per_scenario": scenarios,
        "violating_observation_keys": violation_keys,
        "grid_points": int(grid_points),
        "reliability_grid": [float(x) for x in grid],
        "coefficients": {
            "kappa_sigma": float(kappa_sigma), "kappa_mu": float(kappa_mu),
            "kappa_l": float(kappa_l), "horizon_cap": horizon_cap,
        },
        "method": "Every saved held-out test observation; both raw probability curves swept over R_REL in [0,1]; strict positive increments required.",
        "outcome_boundary": "Inventory and stockout outcomes are simulated; this audit checks mathematical monotonicity, not observed stockout prediction.",
    }


def run_saved_audit(processed_dir="data/processed", output_dir="output"):
    root = Path(__file__).resolve().parents[3]
    processed = Path(processed_dir)
    output = Path(output_dir)
    if not processed.is_absolute():
        processed = root / processed
    if not output.is_absolute():
        output = root / output
    scenarios = pd.read_parquet(processed / "scenario_streams.parquet")
    inventory = pd.read_parquet(processed / "simulated_inventory.parquet")
    join_keys = KEYS + ["PERIOD", "SCENARIO_ID"]
    required_inventory = join_keys + ["AVAILABLE_STOCK"]
    missing = sorted(set(required_inventory) - set(inventory.columns))
    if missing:
        raise ValueError(f"Saved simulated inventory missing columns: {missing}")
    frame = scenarios.merge(
        inventory[required_inventory], on=join_keys, how="inner",
        validate="one_to_one",
    )
    config_path = root / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    risk = (config or {}).get("radcsr", {})
    result = audit_frame(
        frame,
        kappa_sigma=float(risk.get("kappa_sigma", 0.50)),
        kappa_mu=float(risk.get("kappa_mu", 0.75)),
        kappa_l=float(risk.get("kappa_l", 1.50)),
        horizon_cap=risk.get("horizon_cap", 2.0),
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "empirical_monotonicity_audit.json"
    target.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "PASS":
        raise RuntimeError(
            "Empirical monotonicity audit failed; "
            f"low_availability_failures={result['low_availability_failures']}; "
            f"report={target}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    run_saved_audit(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
