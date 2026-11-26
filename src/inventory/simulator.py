"""
src/inventory/simulator.py
==========================
Phase 4: Inventory & Shortage Simulation - Fully Pre-allocated & Fast
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

INVENTORY_CONFIGS = {
    "config_lean": {
        "name": "Lean_JIT",
        "init_stock_mult": 1.5,
        "reorder_mult": 1.0,
        "target_cover_mult": 2.0,
        "lead_time_min": 1,
        "lead_time_max": 2,
        "service_level": 0.90
    },
    "config_base": {
        "name": "Base_Standard",
        "init_stock_mult": 2.0,
        "reorder_mult": 1.2,
        "target_cover_mult": 3.0,
        "lead_time_min": 1,
        "lead_time_max": 2,
        "service_level": 0.95
    },
    "config_buffered": {
        "name": "Buffered_Conservative",
        "init_stock_mult": 3.0,
        "reorder_mult": 1.8,
        "target_cover_mult": 4.0,
        "lead_time_min": 2,
        "lead_time_max": 3,
        "service_level": 0.99
    }
}

def assign_ven_classes(df: pd.DataFrame, med_col: str = "MEDICINE_ID", demand_col: str = "DEMAND_CLEAN", seed: int = 42) -> pd.DataFrame:
    """Clinical VEN classes cannot be inferred from demand or randomly assigned."""
    raise ValueError(
        "Synthetic VEN/substitutability assignment is disabled. Supply a verified "
        "clinical classification source with medicine-level provenance instead."
    )


def simulate_inventory(scenario_df: pd.DataFrame,
                       config: Dict[str, Any] = None,
                       seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    cfg = INVENTORY_CONFIGS["config_base"].copy()
    if config:
        cfg.update(config)

    z_sl = 1.645 if cfg.get("service_level", 0.95) == 0.95 else (1.282 if cfg.get("service_level", 0.95) == 0.90 else 2.326)

    # Sort once
    df_sorted = scenario_df.sort_values(["SCENARIO_ID", "MEDICINE_ID", "PERIOD"]).reset_index(drop=True)
    n_total = len(df_sorted)

    # Pre-allocate output arrays
    out_i_start = np.empty(n_total, dtype=np.float32)
    out_receipts = np.empty(n_total, dtype=np.float32)
    out_stockout = np.empty(n_total, dtype=np.int8)
    out_shortage_qty = np.empty(n_total, dtype=np.float32)
    out_i_end = np.empty(n_total, dtype=np.float32)
    out_order_placed = np.empty(n_total, dtype=np.float32)
    out_lead_time = np.empty(n_total, dtype=np.int8)
    out_pipeline = np.empty(n_total, dtype=np.float32)
    out_lt_demand = np.empty(n_total, dtype=np.float32)
    out_lt_shortage = np.empty(n_total, dtype=np.int8)

    demands_all = df_sorted["SCENARIO_DEMAND"].values.astype(np.float64)
    q50_all = df_sorted["Q50"].values.astype(np.float64)
    scen_ids = df_sorted["SCENARIO_ID"].values
    med_ids = df_sorted["MEDICINE_ID"].values

    # Find contiguous boundaries (much faster than groupby)
    # Combine scen and med into keys
    change_mask = np.empty(n_total, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = (scen_ids[1:] != scen_ids[:-1]) | (med_ids[1:] != med_ids[:-1])
    group_starts = np.where(change_mask)[0]
    group_ends = np.append(group_starts[1:], n_total)

    init_mult = cfg["init_stock_mult"]
    reorder_mult = cfg["reorder_mult"]
    target_mult = cfg["target_cover_mult"]
    base_l_min = cfg["lead_time_min"]
    base_l_max = cfg["lead_time_max"]

    # Main contiguous simulation loop
    for start, end in zip(group_starts, group_ends):
        n = end - start
        demands = demands_all[start:end]
        q50 = q50_all[start:end]
        scen = scen_ids[start]

        mean_forecast = max(0.5, float(np.mean(q50)))
        std_forecast = max(0.2, float(np.std(q50)))

        is_disruption = (scen == "S7_LeadTime")
        l_min = (base_l_min + 1) if is_disruption else base_l_min
        l_max = (base_l_max + 2) if is_disruption else base_l_max

        current_inv = float(mean_forecast * init_mult)
        # Circular pipeline array buffer: arrival_time -> qty
        # max simulation length n + max_lead_time
        max_t = n + l_max + 5
        order_arrivals = np.zeros(max_t, dtype=np.float64)
        l_times = np.random.randint(l_min, l_max + 1, size=n).astype(np.int8)

        reorder_point = mean_forecast * reorder_mult
        target_level = mean_forecast * target_mult + z_sl * std_forecast

        for t in range(n):
            idx = start + t
            lt_curr = l_times[t]
            out_lead_time[idx] = lt_curr

            # 1. Start inventory
            i_start = current_inv
            out_i_start[idx] = i_start

            # 2. Receipts
            arrived = order_arrivals[t]
            order_arrivals[t] = 0.0 # reset
            out_receipts[idx] = arrived

            i_avail = i_start + arrived
            d_t = demands[t]

            if d_t > i_avail:
                stk = 1
                shortage = d_t - i_avail
                i_end = 0.0
            else:
                stk = 0
                shortage = 0.0
                i_end = i_avail - d_t

            out_stockout[idx] = stk
            out_shortage_qty[idx] = shortage
            out_i_end[idx] = i_end

            # Pipeline open orders due within lead time:
            # sum of orders in buffer between t+1 and t+lt_curr
            pipeline_o = float(np.sum(order_arrivals[t + 1 : t + lt_curr + 1]))
            out_pipeline[idx] = pipeline_o

            # Reorder
            total_open = float(np.sum(order_arrivals[t + 1 :]))
            inv_position = i_end + total_open

            if inv_position <= reorder_point:
                o_qty = max(0.0, target_level - inv_position)
                order_arrivals[t + lt_curr] += o_qty
                out_order_placed[idx] = o_qty
            else:
                out_order_placed[idx] = 0.0

            current_inv = i_end

        # Forward lead time actual demand & shortage event
        for t in range(n):
            idx = start + t
            lt = int(l_times[t])
            f_end = min(n, t + 1 + lt)
            if f_end > t + 1:
                f_slice = demands[t + 1 : f_end]
                d_lt = float(np.sum(f_slice)) * (float(lt) / float(len(f_slice)))
            else:
                d_lt = float(demands[t]) * float(lt)

            out_lt_demand[idx] = d_lt
            avail_lt = out_i_start[idx] + out_pipeline[idx]
            out_lt_shortage[idx] = 1 if d_lt > avail_lt else 0

    df_sorted["I_START"] = out_i_start
    df_sorted["RECEIPTS"] = out_receipts
    df_sorted["STOCKOUT"] = out_stockout
    df_sorted["SHORTAGE_QTY"] = out_shortage_qty
    df_sorted["I_END"] = out_i_end
    df_sorted["ORDER_PLACED"] = out_order_placed
    df_sorted["LEAD_TIME"] = out_lead_time
    df_sorted["PIPELINE_ORDERS"] = out_pipeline
    df_sorted["LEAD_TIME_DEMAND"] = out_lt_demand
    df_sorted["LEAD_TIME_SHORTAGE_Y"] = out_lt_shortage

    return df_sorted
