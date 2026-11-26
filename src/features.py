"""Compatibility import. New code should import src.set_forecasting.features."""
from src.set_forecasting.features.engineering import SERIES_KEYS as KEY, TARGET, create_features, leakage_check
__all__=["KEY","TARGET","create_features","leakage_check"]
