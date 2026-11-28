"""Memory-conscious global quantile LightGBM and naive forecasting baselines."""
import os
import numpy as np
import pandas as pd
from ..features.engineering import create_features, SERIES_KEYS, TARGET

FEATURES = [
    "LAG_1", "LAG_2", "LAG_3", "LAG_6", "LAG_12",
    "ROLL_MEAN_3", "ROLL_MEDIAN_3", "ROLL_STD_3", "ROLL_CV_3",
    "ROLL_MEAN_6", "ROLL_MEDIAN_6", "ROLL_STD_6", "ROLL_CV_6",
    "ROLL_MIN_6", "ROLL_MAX_6", "ZERO_RATIO_6", "OBSERVED_RATIO_6",
    "MONTHS_SINCE_NONZERO", "PRIOR_NONZERO_MEAN_6", "MOMENTUM_3M",
    "SEASONAL_RATIO_12", "MONTH", "QUARTER", "YEAR", "MONTH_SIN",
    "MONTH_COS", "TREND_INDEX", "LAG_UNIT_COST_1", "ROLL_UNIT_COST_3",
]


def _score(frame, name):
    y = frame["MODEL_DEMAND"].to_numpy(float)
    low = frame["Q10"].to_numpy(float)
    median = frame["Q50"].to_numpy(float)
    high = frame["Q90"].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(low) & np.isfinite(median) & np.isfinite(high)
    y, low, median, high = y[valid], low[valid], median[valid], high[valid]
    if not len(y):
        return {"MODEL": name, "N": 0}
    pinball = lambda q, p: float(np.mean(np.maximum(q * (y - p), (q - 1) * (y - p))))
    return {
        "MODEL": name, "N": len(y),
        "MAE": float(np.mean(np.abs(y - median))),
        "WAPE": float(np.abs(y - median).sum() / max(np.abs(y).sum(), 1e-9) * 100),
        "PICP_80": float(np.mean((y >= low) & (y <= high))),
        "MPIW_80": float(np.mean(high - low)),
        "PINBALL_10": pinball(.1, low),
        "PINBALL_50": pinball(.5, median),
        "PINBALL_90": pinball(.9, high),
    }


def _winsorise_by_series(frame, quantile):
    """Calculate train-only positive-demand thresholds with one grouped aggregation."""
    train_mask = frame["SPLIT"].eq("train") & ~frame["OBSERVATION_MISSING"].fillna(True).astype(bool)
    positive_train = pd.to_numeric(frame.loc[train_mask, "MODEL_DEMAND"], errors="coerce")
    positive_train = positive_train[positive_train > 0]
    keys = frame.loc[positive_train.index, SERIES_KEYS]
    thresholds = positive_train.groupby(
        [keys[column] for column in SERIES_KEYS], sort=False, dropna=False
    ).quantile(quantile)
    row_keys = pd.MultiIndex.from_frame(frame[SERIES_KEYS])
    limits = thresholds.reindex(row_keys).to_numpy(dtype="float64")
    limits[~np.isfinite(limits)] = np.inf
    values = pd.to_numeric(frame["MODEL_DEMAND"], errors="coerce").to_numpy(dtype="float64")
    winsorised = np.where(values > 0, np.minimum(values, limits), values)
    return limits, winsorised, values > limits


def train_and_predict(series, min_train_rows=100, random_state=42, winsor_quantile=.995):
    """Train quantile models with bounded thread use and avoid per-series Python loops."""
    required = set(SERIES_KEYS + ["PERIOD", "MODEL_DEMAND", "OBSERVATION_MISSING", "SPLIT"])
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"Monthly series is missing columns: {sorted(missing)}")

    # Keep only source columns needed downstream; avoid an extra full-frame copy.
    keep = list(dict.fromkeys(
        SERIES_KEYS + [
            "PERIOD", "MODEL_DEMAND", "OBSERVATION_MISSING", "SPLIT",
            "YEAR_MONTH", "TRAIN_END", "VALIDATION_END", "TEST_START",
            "TOTAL_INDICATIVE_COST", "POSITIVE_QUANTITY", "NEGATIVE_QUANTITY",
            "NET_QUANTITY", "SOURCE_RECORDS", "VALID_QUANTITY_RECORDS",
            "INVALID_QUANTITY_RECORDS", "MISSING_OR_INVALID_COST_RECORDS",
            "NEGATIVE_COST_RECORDS", "COHORT_INCLUDED",
        ]
    ))
    x = series.loc[:, [column for column in keep if column in series.columns]].copy()
    x["PERIOD"] = pd.to_datetime(x["PERIOD"], errors="raise")
    original = pd.to_numeric(x["MODEL_DEMAND"], errors="coerce")
    limits, winsorised, outliers = _winsorise_by_series(x, winsor_quantile)
    x["MODEL_DEMAND_WINSORIZED"] = winsorised
    x["OUTLIER_995_FLAG"] = outliers
    x["WINSOR_LIMIT_995"] = limits
    x["_ORIGINAL_MODEL_DEMAND"] = original
    x[TARGET] = x["MODEL_DEMAND_WINSORIZED"]

    features = create_features(x)
    features["MODEL_DEMAND"] = features["_ORIGINAL_MODEL_DEMAND"]
    features.drop(columns=["_ORIGINAL_MODEL_DEMAND"], inplace=True)
    features["SERIES_ID"] = features[SERIES_KEYS].astype(str).agg("|".join, axis=1)
    categorical = []
    for column in SERIES_KEYS:
        name = column + "_CAT"
        features[name] = features[column].astype("category")
        categorical.append(name)
    columns = [column for column in FEATURES if column in features.columns] + categorical
    for column in FEATURES:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    train_mask = features["SPLIT"].eq("train") & features["TARGET_OBSERVED"]
    val_mask = features["SPLIT"].eq("validation") & features["TARGET_OBSERVED"]
    validation_mask = features["SPLIT"].eq("validation")
    test_mask = features["SPLIT"].eq("test")
    train = features.loc[train_mask & features["MODEL_DEMAND_WINSORIZED"].notna()]
    val = features.loc[val_mask & features["MODEL_DEMAND_WINSORIZED"].notna()]
    validation_calendar = features.loc[validation_mask]
    test = features.loc[test_mask]

    models = []
    status = "naive_fallback"
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor

        if len(train) >= min_train_rows and len(val) >= 20:
            available_cpus = os.cpu_count() or 2
            model_threads = max(1, min(4, available_cpus))
            train_x = train[columns]
            train_y = train["MODEL_DEMAND_WINSORIZED"]
            val_x = val[columns]
            val_y = val["MODEL_DEMAND_WINSORIZED"]
            for quantile in (.1, .5, .9):
                estimator = LGBMRegressor(
                    objective="quantile", alpha=quantile, n_estimators=800,
                    learning_rate=.03, num_leaves=31, min_child_samples=30,
                    subsample=.85, subsample_freq=1, colsample_bytree=.85, max_bin=127,
                    force_col_wise=True, verbosity=-1, random_state=random_state,
                    n_jobs=model_threads,
                )
                estimator.fit(
                    train_x, train_y, categorical_feature=categorical,
                    eval_set=[(val_x, val_y)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                models.append(estimator)
            status = "quantile_lightgbm"
    except ImportError:
        pass

    def predict(part):
        result = part.copy()
        if result.empty:
            return result
        if models:
            raw = np.column_stack([
                np.maximum(0, estimator.predict(result[columns])) for estimator in models
            ])
            result[["Q10", "Q50", "Q90"]] = np.sort(raw, axis=1)
        else:
            result["Q50"] = result["LAG_1"].fillna(result["LAG_12"]).fillna(0).clip(lower=0)
            result["Q10"] = result["Q50"] * .5
            result["Q90"] = result["Q50"] * 1.5
        result["FORECAST_MODEL"] = status
        result["NAIVE_LAST"] = result["LAG_1"].fillna(0).clip(lower=0)
        result["NAIVE_SEASONAL_12"] = result["LAG_12"].fillna(result["NAIVE_LAST"]).clip(lower=0)
        return result

    validation_pred = predict(validation_calendar)
    test_pred = predict(test)
    all_predictions = pd.concat([validation_pred, test_pred], ignore_index=True)
    metrics = []
    if len(test_pred):
        metrics.append(_score(
            test_pred, "QuantileLightGBM" if status == "quantile_lightgbm" else "NaiveFallback"
        ))
        for column, name in (
            ("NAIVE_LAST", "PreviousObservationNaive"),
            ("NAIVE_SEASONAL_12", "SeasonalNaive12"),
        ):
            baseline = test_pred[["MODEL_DEMAND", column]].copy()
            baseline["Q50"] = baseline[column]
            baseline["Q10"] = baseline[column] * .5
            baseline["Q90"] = baseline[column] * 1.5
            metrics.append(_score(baseline, name))

    report = {
        "model": status, "train_rows": len(train), "validation_rows": len(val),
        "validation_calendar_rows": len(validation_calendar), "test_rows": len(test_pred),
        "test_observed_rows": int(test_pred["TARGET_OBSERVED"].sum()),
        "validation_predictions": len(validation_pred),
        "series_count": int(features.groupby(SERIES_KEYS, sort=False, dropna=False).ngroups),
        "feature_rows": int(len(features)),
        "lightgbm_threads": max(1, min(4, os.cpu_count() or 2)),
    }
    return features, all_predictions, pd.DataFrame(metrics), report
