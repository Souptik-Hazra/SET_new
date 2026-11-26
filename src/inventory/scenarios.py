"""
src/inventory/scenarios.py
==========================
Phase 5 & Phase 7: Controlled Demand Scenarios (S1 - S10)
Strict Conformance with Project Specification & Phase 7 Circularity Breaker:
  - S1: Stable (unmodified history)
  - S2: Abrupt surge (1.75x)
  - S3: Gradual drift ((1+g)^t, g=0.02)
  - S4: Volatility increase (residual std multiplied by 2.5)
  - S5: Bias shift (add +std(y))
  - S6: Intermittent (zero stretches with burst multiplier)
  - S7: Lead-time disruption (demand unchanged, lead time extended)
  - S8: Correlated surge (AR(1) shock)
  - S9: Downward drift (demand declines by 2% per period after change point)
  - S10: Variance shift (mean unchanged, residual std doubles after change point)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
SEED = 42

def generate_scenarios(test_df: pd.DataFrame,
                       diagnostics_df: pd.DataFrame = None,
                       demand_col: str = "DEMAND_CLEAN",
                       seed: int = SEED) -> pd.DataFrame:
    np.random.seed(seed)
    base_df = test_df.sort_values(["MEDICINE_ID", "PERIOD"]).reset_index(drop=True)
    n_total = len(base_df)

    med_ids = base_df["MEDICINE_ID"].values
    unique_meds, med_starts, med_counts = np.unique(med_ids, return_index=True, return_counts=True)
    y_raw = base_df[demand_col].values.astype(np.float64)

    scenarios_list = []

    # 1. S1_Stable
    s1 = base_df.copy()
    s1["SCENARIO_ID"] = "S1_Stable"
    s1["SCENARIO_DEMAND"] = y_raw.copy()
    s1["SHIFT_START_INDEX"] = -1
    s1["SHIFT_MAGNITUDE"] = 1.0
    scenarios_list.append(s1)

    # 2. Configs for S2 - S10
    all_scen_configs = [
        ("S2_Surge", 1.75),
        ("S3_Drift", 0.02),
        ("S4_Volatility", 2.5),
        ("S5_Bias", 1.0),
        ("S7_LeadTime", 1.0),
        ("S8_CorrelatedSurge", 1.5),
        ("S6_Intermittent", 2.0),
        ("S9_DownwardDrift", -0.02),
        ("S10_VarianceShift", 2.0)
    ]

    for scen_id, def_mag in all_scen_configs:
        y_scen = y_raw.copy()
        shift_starts = np.empty(n_total, dtype=np.int32)
        mags = np.empty(n_total, dtype=np.float32)

        for start, count in zip(med_starts, med_counts):
            end = start + count
            y_m = y_scen[start:end]
            s_start = max(1, count // 3)
            shift_starts[start:end] = s_start
            mags[start:end] = def_mag

            if scen_id == "S2_Surge":
                dur = min(3, count - s_start)
                y_m[s_start : s_start + dur] *= def_mag
            elif scen_id == "S3_Drift":
                t_steps = np.arange(count - s_start)
                y_m[s_start:] *= ((1.0 + def_mag) ** t_steps)
            elif scen_id == "S4_Volatility":
                mean_y = float(np.mean(y_m))
                res = y_m[s_start:] - mean_y
                y_m[s_start:] = np.maximum(0.0, mean_y + res * def_mag)
            elif scen_id == "S5_Bias":
                std_y = float(np.std(y_m))
                mag_val = std_y if std_y > 0 else 1.0
                mags[start:end] = mag_val
                y_m[s_start:] += mag_val
            elif scen_id == "S7_LeadTime":
                pass # Lead time changed in simulator
            elif scen_id == "S8_CorrelatedSurge":
                dur = min(4, count - s_start)
                shock = 0.0
                for k in range(dur):
                    shock = 0.7 * shock + np.random.normal(0, 0.5)
                    y_m[s_start + k] *= (def_mag + max(0.0, shock))
            elif scen_id == "S6_Intermittent":
                for t in range(s_start, count):
                    if t % 3 != 0:
                        y_m[t] = 0.0
                    else:
                        y_m[t] = y_m[t] * def_mag if y_m[t] > 0 else float(np.mean(y_m[:s_start]) * def_mag)
            elif scen_id == "S9_DownwardDrift":
                # Phase 7: Demand declines by 2% per period after change point
                t_steps = np.arange(count - s_start)
                y_m[s_start:] = np.maximum(0.0, y_m[s_start:] * ((1.0 - 0.02) ** t_steps))
            elif scen_id == "S10_VarianceShift":
                # Phase 7: Mean unchanged, residual std doubles after change point
                mean_y = float(np.mean(y_m[:s_start])) if s_start > 0 else float(np.mean(y_m))
                res = y_m[s_start:] - mean_y
                y_m[s_start:] = np.maximum(0.0, mean_y + res * 2.0)

            y_scen[start:end] = y_m

        s_df = base_df.copy()
        s_df["SCENARIO_ID"] = scen_id
        s_df["SCENARIO_DEMAND"] = y_scen
        s_df["SHIFT_START_INDEX"] = shift_starts
        s_df["SHIFT_MAGNITUDE"] = mags
        scenarios_list.append(s_df)

    out = pd.concat(scenarios_list, ignore_index=True)
    logger.info(f"Generated all 10 scenarios (S1-S10): Total rows: {len(out):,}")
    return out
