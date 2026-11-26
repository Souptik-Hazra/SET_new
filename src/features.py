"""
features.py
===========
Phase 2: Feature Engineering

Creates all features with strict leakage guard:
  - Every rolling/lag value at row t uses ONLY data from t-1 and earlier.
  - Features computed per medicine independently.

Outputs: data/processed/features.parquet
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


def create_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: long-format demand_series.csv with columns:
        PERIOD, MEDICINE_ID, DEMAND_CLEAN, GRANULARITY
    Output: feature-augmented DataFrame (same rows, many more columns)
    """
    logger.info("=== PHASE 2: Feature Engineering ===")
    all_dfs = []

    for med, grp in series_df.groupby("MEDICINE_ID"):
        grp = grp.sort_values("PERIOD").copy().reset_index(drop=True)
        d = grp["DEMAND_CLEAN"].values
        gran = grp["GRANULARITY"].iloc[0]

        # Lag config: 7 for daily, 4 for weekly
        n_lags = 7 if gran == "daily" else 4
        # Rolling windows: (7, 14) for daily; (4, 8) for weekly
        windows = [7, 14] if gran == "daily" else [4, 8]

        # --- Lag features (shifted by 1 so row t uses t-1, t-2, ...) ---
        for lag in range(1, n_lags + 1):
            grp[f"lag_{lag}"] = grp["DEMAND_CLEAN"].shift(lag)

        # --- Rolling features (shift(1) ensures t uses up to t-1) ---
        for w in windows:
            shifted = grp["DEMAND_CLEAN"].shift(1)
            grp[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            grp[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std()
            grp[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()
            grp[f"zero_ratio_{w}"] = (shifted == 0).rolling(w, min_periods=1).mean()

        # --- Calendar features ---
        grp["month"] = grp["PERIOD"].dt.month
        grp["quarter"] = grp["PERIOD"].dt.quarter
        grp["dow"] = grp["PERIOD"].dt.dayofweek
        grp["trend_index"] = np.arange(len(grp))

        # Annual seasonality (sinusoidal encoding)
        doy = grp["PERIOD"].dt.dayofyear
        grp["sin_annual"] = np.sin(2 * np.pi * doy / 365.25)
        grp["cos_annual"] = np.cos(2 * np.pi * doy / 365.25)

        # --- Intermittency features (computed from entire series up to t-1) ---
        for w in windows:
            shifted_w = grp["DEMAND_CLEAN"].shift(1)
            roll_vals = shifted_w.rolling(w, min_periods=2)
            grp[f"adi_{w}"] = roll_vals.apply(
                lambda x: _compute_adi(x), raw=True
            )
            roll_mean = shifted_w.rolling(w, min_periods=1).mean()
            roll_std = shifted_w.rolling(w, min_periods=2).std()
            grp[f"cv2_{w}"] = np.where(
                roll_mean > 0,
                (roll_std / roll_mean) ** 2,
                0.0
            )

        # Days since last non-zero issue
        grp["days_since_last_issue"] = _days_since_last_nonzero(grp["DEMAND_CLEAN"])

        all_dfs.append(grp)

    out = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Feature engineering complete: {out.shape[0]:,} rows, {out.shape[1]} columns")
    return out


def _compute_adi(x: np.ndarray) -> float:
    nonzero_pos = np.where(x > 0)[0]
    if len(nonzero_pos) < 2:
        return float(len(x))
    return float(np.mean(np.diff(nonzero_pos)))


def _days_since_last_nonzero(series: pd.Series) -> pd.Series:
    result = []
    last_nonzero = np.nan
    for i, val in enumerate(series.values):
        if i == 0:
            result.append(np.nan)
        else:
            prev = series.iloc[i - 1]
            if prev > 0:
                last_nonzero = i - 1
            if np.isnan(last_nonzero):
                result.append(np.nan)
            else:
                result.append(float(i - last_nonzero))
    return pd.Series(result, index=series.index)


def leakage_check(features_df: pd.DataFrame, demand_col: str = "DEMAND_CLEAN") -> bool:
    lag_cols = [c for c in features_df.columns if c.startswith("lag_")]
    roll_cols = [c for c in features_df.columns if c.startswith("roll_mean_")]
    check_cols = lag_cols[:3] + roll_cols[:2]

    for col in check_cols:
        clean = features_df[[col, demand_col]].dropna()
        if len(clean) < 10:
            continue
        corr = abs(clean[col].corr(clean[demand_col]))
        if corr >= 0.9999:
            logger.error(f"LEAKAGE DETECTED in column '{col}': correlation = {corr:.6f}")
            return False

    for med, grp in features_df.groupby("MEDICINE_ID"):
        grp = grp.sort_values("PERIOD").reset_index(drop=True)
        if "lag_1" in grp.columns and len(grp) > 2:
            shifted_demand = grp["DEMAND_CLEAN"].shift(1)
            diff = (grp["lag_1"] - shifted_demand).abs().max()
            if diff > 1e-9:
                logger.error(f"LEAKAGE CHECK FAILED for medicine '{med}': lag_1 != demand.shift(1)")
                return False
        break

    logger.info("Leakage check PASSED: no immediate leakage detected.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    series_path = os.path.join("data", "processed", "demand_series.csv")
    out_path = os.path.join("data", "processed", "features.parquet")

    series_df = pd.read_csv(series_path, parse_dates=["PERIOD"])
    features_df = create_features(series_df)

    passed = leakage_check(features_df)
    if not passed:
        raise RuntimeError("Leakage detected! Halting.")

    features_df.to_parquet(out_path, index=False)
    logger.info(f"Features saved to {out_path}")
    print(f"\nFeatures shape: {features_df.shape}")
    print(features_df.head(3))
