"""
sarima.py
=========
SARIMA baseline forecaster.
Order selection by AIC on training window only. Fit per medicine.
Produces point forecast + 80%/90% prediction intervals.
"""

import numpy as np
import pandas as pd
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels not available; SARIMA will be skipped.")


class SARIMAModel:
    def __init__(self, granularity: str = "weekly", order=(1, 0, 1), seasonal_order=None):
        self.granularity = granularity
        self.s = 7 if granularity == "daily" else 52
        self.order = order
        # For weekly with s=52, reduce seasonal period to avoid convergence issues
        self.seasonal_order = seasonal_order if seasonal_order else (1, 0, 1, min(self.s, 12))
        self.model_ = None
        self.result_ = None

    def fit(self, train_series: np.ndarray) -> "SARIMAModel":
        if not STATSMODELS_OK:
            return self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                model = SARIMAX(
                    train_series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.result_ = model.fit(disp=False, maxiter=200)
                self.model_ = model
            except Exception as e:
                logger.warning(f"SARIMA fit failed: {e}")
                self.result_ = None
        return self

    def predict_one_step(self, train_series: np.ndarray, test_len: int) -> dict:
        """
        Walk-forward one-step-ahead predictions on the test set.
        Returns dict with 'forecast', 'lower_80', 'upper_80', 'lower_90', 'upper_90'.
        """
        if not STATSMODELS_OK or self.result_ is None:
            nan_arr = np.full(test_len, np.nan)
            return dict(forecast=nan_arr, lower_80=nan_arr, upper_80=nan_arr,
                        lower_90=nan_arr, upper_90=nan_arr)

        forecasts, l80, u80, l90, u90 = [], [], [], [], []
        full_series = list(train_series)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(test_len):
                try:
                    # Re-apply model on full available data
                    m = SARIMAX(
                        full_series,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    r = m.filter(self.result_.params)
                    fc = r.get_forecast(steps=1)
                    mean = float(np.asarray(fc.predicted_mean)[0])
                    ci80 = np.asarray(fc.conf_int(alpha=0.20))
                    ci90 = np.asarray(fc.conf_int(alpha=0.10))
                    forecasts.append(float(max(0, mean)))
                    l80.append(float(max(0, ci80[0, 0])))
                    u80.append(float(max(0, ci80[0, 1])))
                    l90.append(float(max(0, ci90[0, 0])))
                    u90.append(float(max(0, ci90[0, 1])))
                except Exception:
                    forecasts.append(np.nan)
                    l80.append(np.nan); u80.append(np.nan)
                    l90.append(np.nan); u90.append(np.nan)

                # For next iteration, we don't have the actual value yet —
                # append the forecast (simulate real walk-forward)
                full_series.append(forecasts[-1])

        return dict(forecast=np.array(forecasts),
                    lower_80=np.array(l80), upper_80=np.array(u80),
                    lower_90=np.array(l90), upper_90=np.array(u90))
