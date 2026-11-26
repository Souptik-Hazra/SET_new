"""
run_full_cohort_pipeline.py  (Rebuilt Corrective Conformance Edition)
====================================================================
Runs the corrected RADCSR framework on the scoped SMOOTH study population (2,093 medicines)
across all 10 scenarios (S1 - S10).
Produces:
  - output/tables/phase9_evaluation_summary.csv (All comparative methods x S1-S10)
  - output/tables/phase10_ablation.csv (All 7 ablations x S1-S10)
  - output/tables/phase12_significance.csv (Paired bootstrap CIs & Holm-Bonferroni)
  - output/correction_report.md (Final synthesis & verdicts)
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.inventory.scenarios import generate_scenarios
from src.inventory.simulator import assign_ven_classes, simulate_inventory, INVENTORY_CONFIGS
from src.reliability.monitors import run_reliability_monitors
from src.risk.radcsr import evaluate_all_risk_methods
from src.evaluation.metrics import (
    brier_score, log_loss, expected_calibration_error,
    detection_metrics, expected_shortage_quantity, cvar_shortage
)
from src.evaluation.statistical_tests import paired_bootstrap_brier, apply_holm_bonferroni

def evaluate_metrics_for_df(df: pd.DataFrame, method_cols: list) -> pd.DataFrame:
    rows = []
    for scen, grp in df.groupby("SCENARIO_ID"):
        y_true = grp["LEAD_TIME_SHORTAGE_Y"].values.astype(int)
        d_lt = grp["LEAD_TIME_DEMAND"].values.astype(float)
        avail_lt = (grp["I_START"] + grp["PIPELINE_ORDERS"]).values.astype(float)
        shortage_rate = float(np.mean(y_true) * 100.0)
        esq_val = expected_shortage_quantity(d_lt, avail_lt)
        cvar_val = cvar_shortage(d_lt, avail_lt)

        for m_name, col in method_cols:
            if col not in grp.columns:
                continue
            p_hat = grp[col].values.astype(float)
            brier = brier_score(y_true, p_hat)
            ll = log_loss(y_true, p_hat)
            ece = expected_calibration_error(y_true, p_hat)
            det = detection_metrics(y_true, p_hat, threshold=0.50)
            mean_risk_event = float(np.mean(p_hat[y_true == 1])) if np.sum(y_true == 1) > 0 else 0.0

            row = {
                "SCENARIO_ID": scen,
                "METHOD": m_name,
                "SHORTAGE_RATE_%": round(shortage_rate, 2),
                "BRIER_SCORE": round(brier, 4),
                "LOG_LOSS": round(ll, 4),
                "ECE": round(ece, 4),
                "RECALL_%": round(det["RECALL_%"], 2),
                "PRECISION_%": round(det["PRECISION_%"], 2),
                "F1_SCORE": round(det["F1_SCORE"], 2),
                "FALSE_ALARM_RATE_%": round(det["FALSE_ALARM_RATE_%"], 2),
                "MEAN_RISK_ON_SHORTAGE": round(mean_risk_event, 4),
                "ESQ": round(esq_val, 2),
                "CVAR_95": round(cvar_val, 2)
            }
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    total_start = time.time()
    logger.info("=== Starting Rebuilt Conformance Pipeline on Smooth Study Population ===")

    # 1. Scope to smooth study population
    preds_path = os.path.join(PROJECT_ROOT, "data", "processed", "quantile_lgbm_test_predictions.csv")
    preds = pd.read_csv(preds_path, parse_dates=["PERIOD"])

    diag_path = os.path.join(PROJECT_ROOT, "output", "tables", "phase1_diagnostics.csv")
    diag_df = pd.read_csv(diag_path)
    smooth_meds = set(diag_df[diag_df["DEMAND_CLASS"] == "Smooth"]["MEDICINE_ID"])
    
    test_preds = preds[preds["MEDICINE_ID"].isin(smooth_meds)].copy()
    logger.info(f"Scoped test set to Smooth cohort: {len(test_preds):,} rows across {test_preds['MEDICINE_ID'].nunique():,} medicines.")

    # 2. Rule-based VEN classification
    ven_df = assign_ven_classes(test_preds)

    # 3. Controlled Demand Scenarios S1 - S10 (Spec & Phase 7)
    logger.info("Generating all 10 demand scenarios (S1 - S10)...")
    scenarios_df = generate_scenarios(test_preds, diagnostics_df=diag_df)

    # 4. Inventory Simulation
    logger.info("Running non-circular inventory simulation...")
    sim_df = simulate_inventory(scenarios_df, config=INVENTORY_CONFIGS["config_base"])

    # 5. Reliability Monitoring
    logger.info("Running reliability monitoring...")
    rel_df = run_reliability_monitors(sim_df)

    # 6. Risk Evaluation: Rebuilt Baselines + Competitors + RADCSR + 7 Ablations
    logger.info("Evaluating all risk methods, competitors, RADCSR, and 7 ablations...")
    final_df = evaluate_all_risk_methods(rel_df, ven_df, compute_expensive_mc=False)

    # 7. Summary Tables
    os.makedirs(os.path.join(PROJECT_ROOT, "output", "tables"), exist_ok=True)

    comparative_methods = [
        ("Static", "RISK_STATIC"),
        ("Rolling_Quantile", "RISK_ROLL_Q"),
        ("EWMA_Variance", "RISK_EWMA"),
        ("ACI_Conformal", "RISK_ACI"),
        ("DtACI_Conformal", "RISK_DTACI"),
        ("Monte_Carlo_Scenario", "RISK_MC"),
        ("Robust_Optimization", "RISK_ROBUST"),
        ("Wasserstein_DRO", "RISK_DRO"),
        ("RADCSR_Full", "RISK_RADCSR")
    ]
    summary_df = evaluate_metrics_for_df(final_df, comparative_methods)
    summary_csv = os.path.join(PROJECT_ROOT, "output", "tables", "phase9_evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Phase 9 Evaluation Summary saved to {summary_csv}")

    # 8. Phase 5 & 10 Ablation Matrix
    ablation_methods = [
        ("RADCSR_Full", "RISK_RADCSR"),
        ("RADCSR_noBias", "RISK_ABL_NO_BIAS"),
        ("RADCSR_noCoverage", "RISK_ABL_NO_COV"),
        ("RADCSR_noDrift", "RISK_ABL_NO_DRIFT"),
        ("RADCSR_noClinical", "RISK_ABL_NO_CLINICAL"),
        ("RADCSR_noHorizon", "RISK_ABL_NO_HORIZON"),
        ("RADCSR_singleSignal", "RISK_ABL_SINGLE")
    ]
    ablation_df = evaluate_metrics_for_df(final_df, ablation_methods)
    ablation_csv = os.path.join(PROJECT_ROOT, "output", "tables", "phase10_ablation.csv")
    ablation_df.to_csv(ablation_csv, index=False)
    logger.info(f"Phase 10 Ablation Table saved to {ablation_csv}")

    # 9. Phase 8 Statistical Testing & Holm-Bonferroni (Phase 12 Significance)
    logger.info("Computing Paired Bootstrap & Holm-Bonferroni across all scenario-baseline pairs...")
    test_baselines = [
        ("Static", "RISK_STATIC"),
        ("Rolling_Quantile", "RISK_ROLL_Q"),
        ("EWMA_Variance", "RISK_EWMA"),
        ("ACI_Conformal", "RISK_ACI"),
        ("DtACI_Conformal", "RISK_DTACI"),
        ("Monte_Carlo_Scenario", "RISK_MC"),
        ("Robust_Optimization", "RISK_ROBUST"),
        ("Wasserstein_DRO", "RISK_DRO")
    ]
    
    boot_rows = []
    for sc in final_df["SCENARIO_ID"].unique():
        sub = final_df[final_df["SCENARIO_ID"] == sc]
        y_t = sub["LEAD_TIME_SHORTAGE_Y"].values.astype(int)
        p_rad = sub["RISK_RADCSR"].values

        for b_name, b_col in test_baselines:
            p_base = sub[b_col].values
            res = paired_bootstrap_brier(y_t, p_rad, p_base, n_boot=1000)
            boot_rows.append({
                "SCENARIO_ID": sc,
                "BASELINE": b_name,
                "MEAN_DIFF": round(res["mean_diff"], 4),
                "CI_LOWER": round(res["ci_lower"], 4),
                "CI_UPPER": round(res["ci_upper"], 4),
                "P_VALUE": round(res["p_value"], 4)
            })

    boot_df = pd.DataFrame(boot_rows)
    significance_df = apply_holm_bonferroni(boot_df, p_col="P_VALUE", alpha=0.05)
    sig_csv = os.path.join(PROJECT_ROOT, "output", "tables", "phase12_significance.csv")
    significance_df.to_csv(sig_csv, index=False)
    logger.info(f"Phase 12 Significance Table saved to {sig_csv}")

    elapsed = time.time() - total_start
    logger.info(f"Corrective Conformance Pipeline completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
