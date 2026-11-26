"""
quantile_lgbm.py
================
Quantile LightGBM — PRIMARY probabilistic forecasting model.
Trains three models at alpha in {0.10, 0.50, 0.90}.
This model is frozen for all subsequent reliability experiments.
"""

import numpy as np
import pandas as pd
import hashlib
import logging
import pickle
import os

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LGB_OK = True
except ImportError:
    LGB_OK = False
    logger.warning("lightgbm not available.")


QUANTILES = [0.10, 0.50, 0.90]

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


class QuantileLGBM:
    """
    Three separate LightGBM models for quantile regression at q10, q50, q90.
    After fitting, models are frozen (not retrained in reliability phases).
    """

    def __init__(self, seed: int = 42, quantiles: list = None):
        self.seed = seed
        self.quantiles = quantiles or QUANTILES
        self.models_ = {}
        self._feature_cols = None
        self.version_hash_ = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        gran = df["GRANULARITY"].iloc[0] if "GRANULARITY" in df.columns else "weekly"
        candidates = FEATURE_COLS_DAILY if gran == "daily" else FEATURE_COLS_WEEKLY
        return [c for c in candidates if c in df.columns]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            demand_col: str = "DEMAND_CLEAN") -> "QuantileLGBM":
        if not LGB_OK:
            return self

        self._feature_cols = self._get_feature_cols(train_df)
        X_train = train_df[self._feature_cols].fillna(0)
        y_train = train_df[demand_col].values
        X_val = val_df[self._feature_cols].fillna(0)
        y_val = val_df[demand_col].values

        for q in self.quantiles:
            model = LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="quantile",
                alpha=q,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1)
                ]
            )
            self.models_[q] = model
            logger.info(f"  Quantile LGB q={q:.2f} fitted. Best iter: {model.best_iteration_}")

        # Compute version hash for reproducibility record
        self.version_hash_ = self._compute_hash(X_train)
        logger.info(f"Quantile LGB version hash: {self.version_hash_}")
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with columns: q10, q50, q90 — clipped to >= 0.
        """
        if not LGB_OK or not self.models_:
            nan_arr = np.full(len(test_df), np.nan)
            return pd.DataFrame({"q10": nan_arr, "q50": nan_arr, "q90": nan_arr})

        X = test_df[self._feature_cols].fillna(0)
        results = {}
        for q in self.quantiles:
            preds = self.models_[q].predict(X)
            results[f"q{int(q*100):02d}"] = np.maximum(preds, 0)

        return pd.DataFrame(results, index=test_df.index)

    def _compute_hash(self, X_train) -> str:
        h = hashlib.sha256(pd.util.hash_pandas_object(X_train, index=True).values.tobytes())
        return h.hexdigest()[:12]

    def save(self, path: str):
        """Save all models to disk for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"QuantileLGBM saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QuantileLGBM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"QuantileLGBM loaded from {path}, hash={obj.version_hash_}")
        return obj
