"""
src/reliability/monitors.py
===========================
Phase 6: Reliability & Calibration Baselines - Contiguous Vectorized Implementation
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_reliability_monitors(sim_df: pd.DataFrame, window: int = 4, target_cov: float = 0.90) -> pd.DataFrame:
    df_sorted = sim_df.sort_values(["SCENARIO_ID", "MEDICINE_ID", "PERIOD"]).reset_index(drop=True)
    n_total = len(df_sorted)

    alpha = 1.0 - target_cov
    gamma_aci = 0.05

    # Pre-allocate output arrays
    out_wape_alarm = np.zeros(n_total, dtype=np.int8)
    out_cusum_alarm = np.zeros(n_total, dtype=np.int8)
    out_adwin_alarm = np.zeros(n_total, dtype=np.int8)

    out_aci_lower = np.empty(n_total, dtype=np.float32)
    out_aci_upper = np.empty(n_total, dtype=np.float32)
    out_dtaci_lower = np.empty(n_total, dtype=np.float32)
    out_dtaci_upper = np.empty(n_total, dtype=np.float32)
    out_adapi_lower = np.empty(n_total, dtype=np.float32)
    out_adapi_upper = np.empty(n_total, dtype=np.float32)
    out_enbpi_lower = np.empty(n_total, dtype=np.float32)
    out_enbpi_upper = np.empty(n_total, dtype=np.float32)
    out_ewma_lower = np.empty(n_total, dtype=np.float32)
    out_ewma_upper = np.empty(n_total, dtype=np.float32)
    out_roll_q_lower = np.empty(n_total, dtype=np.float32)
    out_roll_q_upper = np.empty(n_total, dtype=np.float32)

    y_all = df_sorted["SCENARIO_DEMAND"].values.astype(np.float64)
    y_hat_all = df_sorted["Q50"].values.astype(np.float64)
    q10_all = df_sorted["Q10"].values.astype(np.float64)
    q90_all = df_sorted["Q90"].values.astype(np.float64)

    scen_ids = df_sorted["SCENARIO_ID"].values
    med_ids = df_sorted["MEDICINE_ID"].values

    change_mask = np.empty(n_total, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = (scen_ids[1:] != scen_ids[:-1]) | (med_ids[1:] != med_ids[:-1])
    group_starts = np.where(change_mask)[0]
    group_ends = np.append(group_starts[1:], n_total)

    for start, end in zip(group_starts, group_ends):
        n = end - start
        y = y_all[start:end]
        y_hat = y_hat_all[start:end]
        q10 = q10_all[start:end]
        q90 = q90_all[start:end]

        residuals = y - y_hat
        abs_res = np.abs(residuals)

        base_n = max(2, min(4, n // 3))
        mu_0 = np.mean(abs_res[:base_n])
        sigma_0 = np.std(abs_res[:base_n]) if np.std(abs_res[:base_n]) > 1e-4 else 1.0

        cusum_s = 0.0
        ewma_m = 0.0
        ewma_v = sigma_0 ** 2

        aci_alpha = alpha
        dtaci_alpha = alpha
        dtaci_gamma = 0.02
        err_history = []
        cov_history = []
        res_buffer = []

        for t in range(n):
            idx = start + t
            yt = y[t]
            yht = y_hat[t]
            et = abs_res[t]
            base_rad = max(0.1, (q90[t] - q10[t]) / 2.0)

            res_buffer.append(et)
            if len(res_buffer) > 8:
                res_buffer.pop(0)

            # 1. Rolling WAPE
            recent_y = y[max(0, t - window + 1) : t + 1]
            recent_e = abs_res[max(0, t - window + 1) : t + 1]
            wape_t = (np.sum(recent_e) / max(1e-3, np.sum(recent_y)))
            out_wape_alarm[idx] = 1 if (wape_t > 0.80 and t >= window) else 0

            # 2. CUSUM
            zt = (et - mu_0) / sigma_0
            cusum_s = max(0.0, cusum_s + zt - 0.5)
            out_cusum_alarm[idx] = 1 if (cusum_s > 4.0) else 0

            # 3. Fast ADWIN window split proxy
            if t >= window * 2:
                w0 = res_buffer[: len(res_buffer) // 2]
                w1 = res_buffer[len(res_buffer) // 2 :]
                out_adwin_alarm[idx] = 1 if abs(np.mean(w1) - np.mean(w0)) > (1.5 * sigma_0) else 0
            else:
                out_adwin_alarm[idx] = 0

            # 4. Rolling Quantile interval
            q_rad = float(np.quantile(res_buffer, target_cov)) if len(res_buffer) > 1 else (q90[t] - yht)
            out_roll_q_lower[idx] = max(0.0, yht - q_rad)
            out_roll_q_upper[idx] = yht + q_rad

            # 5. EWMA
            ewma_m = 0.8 * ewma_m + 0.2 * residuals[t]
            ewma_v = 0.8 * ewma_v + 0.2 * (residuals[t] - ewma_m) ** 2
            ewma_rad = 1.645 * np.sqrt(max(1e-4, ewma_v))
            out_ewma_lower[idx] = max(0.0, yht - ewma_rad)
            out_ewma_upper[idx] = yht + ewma_rad

            # 6. ACI
            scale_fac = max(0.5, 1.0 + (alpha - aci_alpha))
            c_low = max(0.0, yht - base_rad * scale_fac)
            c_high = yht + base_rad * scale_fac
            out_aci_lower[idx] = c_low
            out_aci_upper[idx] = c_high
            err_t = 1.0 if (yt < c_low or yt > c_high) else 0.0
            aci_alpha = max(0.01, min(0.50, aci_alpha + gamma_aci * (alpha - err_t)))

            # 7. DtACI
            dt_scale = max(0.5, 1.0 + (alpha - dtaci_alpha))
            dt_low = max(0.0, yht - base_rad * dt_scale)
            dt_high = yht + base_rad * dt_scale
            out_dtaci_lower[idx] = dt_low
            out_dtaci_upper[idx] = dt_high
            err_dt = 1.0 if (yt < dt_low or yt > dt_high) else 0.0
            err_history.append(err_dt)
            if len(err_history) >= 4:
                dtaci_gamma = float(np.clip(0.005 + 2.0 * np.var(err_history[-4:]) * 0.095, 0.005, 0.10))
            dtaci_alpha = float(np.clip(dtaci_alpha + dtaci_gamma * (alpha - err_dt), 0.01, 0.50))

            # 8. AdaPI
            cov_history.append(1.0 if (yt >= c_low and yt <= c_high) else 0.0)
            if len(cov_history) >= 2:
                c_pct = np.mean(cov_history[-window:]) * 100.0
                ada_scale = max(0.5, min(3.0, 1.0 - 0.5 * ((target_cov * 100.0 - c_pct) / -10.0)))
            else:
                ada_scale = 1.0
            out_adapi_lower[idx] = max(0.0, yht - base_rad * ada_scale)
            out_adapi_upper[idx] = yht + base_rad * ada_scale

            # 9. EnbPI
            if len(res_buffer) >= 3:
                s_sorted = np.sort(res_buffer)
                q_enb = float(s_sorted[min(int(np.ceil(target_cov * len(s_sorted))) - 1, len(s_sorted) - 1)])
            else:
                q_enb = base_rad
            out_enbpi_lower[idx] = max(0.0, yht - q_enb)
            out_enbpi_upper[idx] = yht + q_enb

    df_sorted["WAPE_ALARM"] = out_wape_alarm
    df_sorted["CUSUM_ALARM"] = out_cusum_alarm
    df_sorted["ADWIN_ALARM"] = out_adwin_alarm
    df_sorted["ACI_LOWER"] = out_aci_lower
    df_sorted["ACI_UPPER"] = out_aci_upper
    df_sorted["DTACI_LOWER"] = out_dtaci_lower
    df_sorted["DTACI_UPPER"] = out_dtaci_upper
    df_sorted["ADAPI_LOWER"] = out_adapi_lower
    df_sorted["ADAPI_UPPER"] = out_adapi_upper
    df_sorted["ENBPI_LOWER"] = out_enbpi_lower
    df_sorted["ENBPI_UPPER"] = out_enbpi_upper
    df_sorted["EWMA_LOWER"] = out_ewma_lower
    df_sorted["EWMA_UPPER"] = out_ewma_upper
    df_sorted["ROLL_Q_LOWER"] = out_roll_q_lower
    df_sorted["ROLL_Q_UPPER"] = out_roll_q_upper

    return df_sorted
