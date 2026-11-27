"""Post-hoc empirical monotonicity verification over every held-out RADCSR row.

For each saved test observation, hold quantiles, availability and lead time fixed,
sweep R_REL from 0 to 1, and verify strict monotonicity of both RADCSR channels.
This standalone check runs after the pipeline and does not modify pipeline outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..risk.radcsr import KEYS, radcsr_probability

PROBABILITY_COLUMNS = ("RADCSR_DISTRIBUTION_PROBABILITY", "RADCSR_PROBABILITY")
OBSERVATION_KEYS = KEYS + ["PERIOD", "SCENARIO_ID"]


def audit_frame(risk_frame, *, kappa_sigma=.50, kappa_mu=.75, kappa_l=1.50,
                horizon_cap=2.0, grid_size=101, batch_size=10000):
    """Return per-scenario pass/fail counts and all violating observation keys."""
    required = set(OBSERVATION_KEYS + [
        "SPLIT", "Q10", "Q50", "Q90", "AVAILABLE_STOCK", "LEAD_TIME_MONTHS", "R_REL",
    ])
    missing = required - set(risk_frame.columns)
    if missing:
        raise ValueError(f"Risk detail artifact is missing columns: {sorted(missing)}")
    if int(grid_size) < 3 or int(batch_size) < 1:
        raise ValueError("grid_size must be >= 3 and batch_size must be positive")
    test = risk_frame.loc[risk_frame["SPLIT"].astype(str).eq("test")].copy()
    if test.empty:
        raise ValueError("Risk detail artifact contains no held-out test observations")
    if test.duplicated(OBSERVATION_KEYS).any():
        raise ValueError("Risk detail artifact has duplicate test observation keys")
    numeric = ["Q10", "Q50", "Q90", "AVAILABLE_STOCK", "LEAD_TIME_MONTHS"]
    if not np.isfinite(test[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Test risk inputs must be finite")

    grid = np.linspace(0.0, 1.0, int(grid_size))
    summaries, violations = [], []
    for scenario, group in test.groupby("SCENARIO_ID", sort=True, dropna=False):
        distribution_pass = probability_pass = both_pass = 0
        for offset in range(0, len(group), int(batch_size)):
            batch = group.iloc[offset:offset + int(batch_size)].reset_index(drop=True)
            n = len(batch)
            result = radcsr_probability(
                np.repeat(batch["Q10"].to_numpy(float), len(grid)),
                np.repeat(batch["Q50"].to_numpy(float), len(grid)),
                np.repeat(batch["Q90"].to_numpy(float), len(grid)),
                np.repeat(batch["AVAILABLE_STOCK"].to_numpy(float), len(grid)),
                np.repeat(batch["LEAD_TIME_MONTHS"].to_numpy(float), len(grid)),
                np.tile(grid, n), kappa_sigma=kappa_sigma, kappa_mu=kappa_mu,
                kappa_l=kappa_l, horizon_cap=horizon_cap,
            )
            distribution = result[PROBABILITY_COLUMNS[0]].to_numpy(float).reshape(n, len(grid))
            final = result[PROBABILITY_COLUMNS[1]].to_numpy(float).reshape(n, len(grid))
            distribution_steps = np.diff(distribution, axis=1)
            final_steps = np.diff(final, axis=1)
            distribution_ok = np.all(distribution_steps > 0.0, axis=1)
            final_ok = np.all(final_steps > 0.0, axis=1)
            distribution_pass += int(distribution_ok.sum())
            probability_pass += int(final_ok.sum())
            both_pass += int((distribution_ok & final_ok).sum())
            for i in np.flatnonzero(~(distribution_ok & final_ok)):
                record = {key: batch.iloc[i][key] for key in OBSERVATION_KEYS}
                record.update({
                    "OBSERVED_R_REL": float(batch.iloc[i]["R_REL"]),
                    "OUTCOME_TYPE": "SIMULATED",
                    "STOCKOUTS_ARE_OBSERVED": 0,
                    "DISCLAIMER": "Inventory outcomes are SIMULATED. Stockout labels are not observed events.",
                    "DISTRIBUTION_MONOTONE": bool(distribution_ok[i]),
                    "RADCSR_MONOTONE": bool(final_ok[i]),
                    "DISTRIBUTION_MIN_INCREMENT": float(distribution_steps[i].min()),
                    "RADCSR_MIN_INCREMENT": float(final_steps[i].min()),
                })
                violations.append(record)
        total = len(group)
        summaries.append({
            "SCENARIO_ID": str(scenario), "TEST_OBSERVATIONS": total,
            "DISTRIBUTION_PASS": distribution_pass, "DISTRIBUTION_FAIL": total-distribution_pass,
            "RADCSR_PASS": probability_pass, "RADCSR_FAIL": total-probability_pass,
            "BOTH_PASS": both_pass, "BOTH_FAIL": total-both_pass,
        })
    violation_columns = OBSERVATION_KEYS + [
        "OBSERVED_R_REL", "DISTRIBUTION_MONOTONE", "RADCSR_MONOTONE",
        "DISTRIBUTION_MIN_INCREMENT", "RADCSR_MIN_INCREMENT",
        "OUTCOME_TYPE", "STOCKOUTS_ARE_OBSERVED", "DISCLAIMER",
    ]
    violation_frame = pd.DataFrame(violations, columns=violation_columns)
    report = {
        "OUTCOME_TYPE": "SIMULATED",
        "STOCKOUTS_ARE_OBSERVED": 0,
        "DISCLAIMER": "Inventory outcomes are SIMULATED. Stockout labels are not observed events.",
        "status": "PASS" if not violations else "FAIL",
        "method": "per-observation counterfactual R_REL sweep across all held-out test observations; strict positive increments",
        "grid_size": int(grid_size), "kappa_sigma": float(kappa_sigma),
        "kappa_mu": float(kappa_mu), "kappa_l": float(kappa_l),
        "horizon_cap": None if horizon_cap is None else float(horizon_cap),
        "test_observations": int(len(test)),
        "distribution_pass": int(sum(x["DISTRIBUTION_PASS"] for x in summaries)),
        "distribution_fail": int(sum(x["DISTRIBUTION_FAIL"] for x in summaries)),
        "radcsr_pass": int(sum(x["RADCSR_PASS"] for x in summaries)),
        "radcsr_fail": int(sum(x["RADCSR_FAIL"] for x in summaries)),
        "both_pass": int(sum(x["BOTH_PASS"] for x in summaries)),
        "both_fail": int(sum(x["BOTH_FAIL"] for x in summaries)),
        "by_scenario": summaries, "violating_observation_keys": violations,
    }
    return report, violation_frame



def _synthetic_fixture():
    """Small saved-artifact-shaped fixture spanning low/high availability regimes."""
    rows = []
    for scenario, availability in (("S1_Stable", 8.0), ("S2_Surge", 30.0)):
        for month in range(2):
            rows.append({
                "ODS_CODE": "ORG", "VMP_SNOMED_CODE": "000123",
                "UNIT_OF_MEASURE_IDENTIFIER": "U1",
                "PERIOD": pd.Timestamp("2025-01-01") + pd.offsets.MonthBegin(month),
                "SCENARIO_ID": scenario, "SPLIT": "test",
                "Q10": 5.0, "Q50": 10.0, "Q90": 15.0,
                "AVAILABLE_STOCK": availability, "LEAD_TIME_MONTHS": 2,
                "R_REL": .4,
            })
    return pd.DataFrame(rows)

def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-path", type=Path, default=root/"data/processed/radcsr_risk_details.parquet")
    parser.add_argument("--config-path", type=Path, default=root/"output/resolved_config.json")
    parser.add_argument("--report-path", type=Path, default=root/"output/empirical_monotonicity_audit.json")
    parser.add_argument("--violations-path", type=Path, default=root/"output/tables/empirical_monotonicity_violations.csv")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--synthetic-fixture", action="store_true", help="Run the verifier on a deterministic low/high-availability test fixture.")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.synthetic_fixture:
        config = {}
        risk_frame = _synthetic_fixture()
    else:
        config = json.loads(args.config_path.read_text(encoding="utf-8")).get("radcsr", {}) if args.config_path.is_file() else {}
        risk_frame = pd.read_parquet(args.risk_path)
    report, violations = audit_frame(
        risk_frame,
        kappa_sigma=float(config.get("kappa_sigma", .50)),
        kappa_mu=float(config.get("kappa_mu", .75)),
        kappa_l=float(config.get("kappa_l", 1.50)),
        horizon_cap=config.get("horizon_cap", 2.0),
        grid_size=args.grid_size, batch_size=args.batch_size,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.violations_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    violations.to_csv(args.violations_path, index=False)
    print(json.dumps(report, indent=2, default=str))
    print(f"Violating observation keys: {len(violations)}")
    print(f"Report: {args.report_path}")
    print(f"Violations: {args.violations_path}")
    return report


if __name__ == "__main__":
    result = main()
    if result["status"] != "PASS":
        raise SystemExit(1)
