"""Vectorised past-only SCMD feature engineering."""
import numpy as np
import pandas as pd

SERIES_KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
TARGET = "MODEL_DEMAND"
LAGS = (1, 2, 3, 6, 12)
WINDOWS = (3, 6, 12)


def _rolling(grouped, column, window, operation):
    values = getattr(grouped[column].rolling(window, min_periods=1), operation)()
    return values.reset_index(level=list(range(len(SERIES_KEYS))), drop=True).sort_index()


def create_features(data):
    """Build all lag/rolling features without a Python loop over 389k+ series."""
    required = set(SERIES_KEYS + ["PERIOD", TARGET, "OBSERVATION_MISSING"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing SCMD columns: {sorted(missing)}")
    if data.empty:
        return data.copy()

    g = data.copy()
    g["PERIOD"] = pd.to_datetime(g["PERIOD"], errors="raise")
    g = g.sort_values(SERIES_KEYS + ["PERIOD"], kind="mergesort").reset_index(drop=True)
    grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
    missing_obs = g["OBSERVATION_MISSING"].fillna(True).astype(bool)
    y = pd.to_numeric(g[TARGET], errors="coerce").where(~missing_obs)
    g["_Y"] = y
    g["_PAST"] = grouped["_Y"].shift(1)
    grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)

    for lag in LAGS:
        g[f"LAG_{lag}"] = grouped["_Y"].shift(lag)

    for window in WINDOWS:
        mean = _rolling(grouped, "_PAST", window, "mean")
        median = _rolling(grouped, "_PAST", window, "median")
        std = _rolling(grouped, "_PAST", window, "std")
        minimum = _rolling(grouped, "_PAST", window, "min")
        maximum = _rolling(grouped, "_PAST", window, "max")
        count = _rolling(grouped, "_PAST", window, "count")
        zeros = g["_PAST"].eq(0).astype("float64")
        g["_ZERO"] = zeros
        grouped_zero = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
        zero_count = _rolling(grouped_zero, "_ZERO", window, "sum")
        g[f"ROLL_MEAN_{window}"] = mean
        g[f"ROLL_MEDIAN_{window}"] = median
        g[f"ROLL_STD_{window}"] = std.fillna(0.0)
        g[f"ROLL_MIN_{window}"] = minimum
        g[f"ROLL_MAX_{window}"] = maximum
        g[f"ROLL_CV_{window}"] = g[f"ROLL_STD_{window}"] / mean.replace(0, np.nan)
        g[f"ZERO_RATIO_{window}"] = zero_count / count.replace(0, np.nan)
        g["_OBSERVED"] = (~missing_obs).astype("float64")
        g["_OBSERVED_PAST"] = g.groupby(
            SERIES_KEYS, sort=False, dropna=False, observed=True
        )["_OBSERVED"].shift(1)
        observed_grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
        g[f"OBSERVED_RATIO_{window}"] = _rolling(
            observed_grouped.assign if False else observed_grouped,
            "_OBSERVED_PAST", window, "mean"
        )

    row_number = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True).cumcount()
    last_nonzero = row_number.where(g["_PAST"].gt(0)).groupby(
        [g[key] for key in SERIES_KEYS], sort=False, dropna=False
    ).ffill()
    g["MONTHS_SINCE_NONZERO"] = row_number - last_nonzero
    g["_POSITIVE_PAST"] = g["_PAST"].where(g["_PAST"] > 0)
    positive_grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
    g["PRIOR_NONZERO_MEAN_6"] = _rolling(positive_grouped, "_POSITIVE_PAST", 6, "mean")
    g["MOMENTUM_3M"] = g["ROLL_MEAN_3"] / g["ROLL_MEAN_6"].replace(0, np.nan)
    g["SEASONAL_RATIO_12"] = g["LAG_12"] / g["ROLL_MEAN_12"].replace(0, np.nan)

    month = g["PERIOD"].dt.month
    g["MONTH"] = month
    g["QUARTER"] = g["PERIOD"].dt.quarter
    g["YEAR"] = g["PERIOD"].dt.year
    g["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    g["MONTH_COS"] = np.cos(2 * np.pi * month / 12)
    g["TREND_INDEX"] = row_number.astype("int32")
    g["TARGET_OBSERVED"] = ~missing_obs

    if {"TOTAL_INDICATIVE_COST", "POSITIVE_QUANTITY"}.issubset(g.columns):
        cost = pd.to_numeric(g["TOTAL_INDICATIVE_COST"], errors="coerce")
        quantity = pd.to_numeric(g["POSITIVE_QUANTITY"], errors="coerce")
        unit_cost = (cost / quantity.where(quantity > 0)).replace([np.inf, -np.inf], np.nan)
        g["_UNIT_COST"] = unit_cost
        unit_grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
        lagged_cost = unit_grouped["_UNIT_COST"].shift(1)
        g["_LAGGED_UNIT_COST"] = lagged_cost
        cost_grouped = g.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
        g["LAG_UNIT_COST_1"] = lagged_cost
        g["ROLL_UNIT_COST_3"] = _rolling(cost_grouped, "_LAGGED_UNIT_COST", 3, "mean")

    return g.drop(columns=[
        "_Y", "_PAST", "_ZERO", "_OBSERVED", "_OBSERVED_PAST",
        "_POSITIVE_PAST", "_UNIT_COST", "_LAGGED_UNIT_COST"
    ], errors="ignore")


def leakage_check(features):
    required = set(SERIES_KEYS + ["PERIOD", TARGET, "OBSERVATION_MISSING", "LAG_1"])
    if required - set(features.columns):
        return False
    ordered = features.sort_values(SERIES_KEYS + ["PERIOD"], kind="mergesort")
    grouped = ordered.groupby(SERIES_KEYS, sort=False, dropna=False, observed=True)
    expected = grouped[TARGET].shift(1).where(
        ~grouped["OBSERVATION_MISSING"].shift(1).fillna(True).astype(bool)
    )
    actual = pd.to_numeric(ordered["LAG_1"], errors="coerce")
    return bool(np.allclose(actual.to_numpy(float), expected.to_numpy(float), equal_nan=True))
