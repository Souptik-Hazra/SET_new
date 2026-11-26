"""
xgboost_model.py
================
XGBoost point-forecast baseline.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("xgboost not available.")


FEATURE_COLS_WEEKLY = [
    "lag_1", "lag_2", "lag_3", "lag_4",
    "roll_mean_4", "roll_std_4", "roll_max_4", "zero_ratio_4",
    "roll_mean_8", "roll_std_8", "roll_max_8", "zero_ratio_8",
    "month", "quarter", "sin_annual", "cos_annual", "trend_index",
    "adi_4", "cv2_4", "days_since_last_issue"
]

FEATURE_COLS_DAILY = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "roll_mean_7", "roll_std_7", "roll_max_7", "zero_ratio_7",
    "roll_mean_14", "roll_std_14", "roll_max_14", "zero_ratio_14",
    "month", "quarter", "dow", "sin_annual", "cos_annual", "trend_index",
    "adi_7", "cv2_7", "days_since_last_issue"
]


class XGBoostModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model_ = None

    def _get_feature_cols(self, features_df: pd.DataFrame) -> list:
        gran = features_df["GRANULARITY"].iloc[0] if "GRANULARITY" in features_df.columns else "weekly"
        candidates = FEATURE_COLS_DAILY if gran == "daily" else FEATURE_COLS_WEEKLY
        return [c for c in candidates if c in features_df.columns]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            demand_col: str = "DEMAND_CLEAN") -> "XGBoostModel":
        if not XGB_OK:
            return self
        feature_cols = self._get_feature_cols(train_df)
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[demand_col].values
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[demand_col].values

        self.model_ = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self._feature_cols = feature_cols
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not XGB_OK or self.model_ is None:
            return np.full(len(test_df), np.nan)
        X = test_df[self._feature_cols].fillna(0)
        preds = self.model_.predict(X)
        return np.maximum(preds, 0)
