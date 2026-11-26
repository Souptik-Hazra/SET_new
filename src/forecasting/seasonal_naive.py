"""
seasonal_naive.py
=================
Seasonal Naive baseline forecaster.
D_hat(t+1) = D(t+1-s) where s=7 (daily) or s=52 (weekly).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SeasonalNaive:
    def __init__(self, granularity: str = "weekly"):
        self.granularity = granularity
        self.s = 7 if granularity == "daily" else 52

    def predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Produce one-step-ahead seasonal naive forecasts for each period.
        Returns array same length as series; first s values are NaN.
        """
        n = len(series)
        preds = np.full(n, np.nan)
        for t in range(self.s, n):
            # Forecast for period t is the value s periods ago
            preds[t] = series[t - self.s]
        return preds

    def predict_series(self, series_df: pd.DataFrame,
                       demand_col: str = "DEMAND_CLEAN",
                       period_col: str = "PERIOD") -> pd.DataFrame:
        """
        Apply seasonal naive to a single-medicine series DataFrame.
        Returns DataFrame with FORECAST column added.
        """
        df = series_df.sort_values(period_col).copy().reset_index(drop=True)
        df["FORECAST"] = self.predict(df[demand_col].values)
        df["MODEL"] = "SeasonalNaive"
        return df
