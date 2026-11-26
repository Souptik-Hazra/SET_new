"""Validation contract for independently sourced, observed inventory outcomes.

This module deliberately does not infer inventory or stockout labels from SCMD.
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER", "PERIOD"]
REQUIRED = KEYS + [
    "OPENING_STOCK", "RECEIPTS", "ISSUES", "LEAD_TIME_DAYS", "STOCKOUT_FLAG",
    "SOURCE_NAME", "SOURCE_VERSION", "SOURCE_URL",
]
NUMERIC = ["OPENING_STOCK", "RECEIPTS", "ISSUES", "LEAD_TIME_DAYS"]


def load_observed_inventory(path):
    """Load only observed, provenance-backed inventory records.

    Expected grain: organisation × VMP × unit × month. A STOCKOUT_FLAG of 1
    must originate from the source system's observed stockout definition.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={
        "ODS_CODE": str,
        "VMP_SNOMED_CODE": str,
        "UNIT_OF_MEASURE_IDENTIFIER": str,
        "STOCKOUT_FLAG": str,
    })
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Observed inventory file missing columns: {missing}")
    for column in ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER",
                   "SOURCE_NAME", "SOURCE_VERSION", "SOURCE_URL"]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any() or frame[column].str.lower().eq("nan").any():
            raise ValueError(f"{column} contains blank values")
    if not frame.SOURCE_URL.str.match(r"^https?://", case=False).all():
        raise ValueError("SOURCE_URL must contain HTTP(S) provenance URLs")
    frame["PERIOD"] = pd.to_datetime(frame["PERIOD"], errors="coerce")
    if frame.PERIOD.isna().any():
        raise ValueError("PERIOD contains invalid dates")
    frame["PERIOD"] = frame.PERIOD.dt.to_period("M").dt.to_timestamp()
    if frame.duplicated(KEYS).any():
        raise ValueError("Observed inventory grain is not unique")
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
        if (frame[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")
    frame["STOCKOUT_FLAG"] = pd.to_numeric(frame.STOCKOUT_FLAG, errors="coerce")
    if frame.STOCKOUT_FLAG.isna().any() or not frame.STOCKOUT_FLAG.isin([0, 1]).all():
        raise ValueError("STOCKOUT_FLAG must be observed binary values 0 or 1")
    if "CLOSING_STOCK" in frame:
        frame["CLOSING_STOCK"] = pd.to_numeric(frame.CLOSING_STOCK, errors="coerce")
        expected = frame.OPENING_STOCK + frame.RECEIPTS - frame.ISSUES
        frame["STOCK_BALANCE_RESIDUAL"] = frame.CLOSING_STOCK - expected
    frame["OUTCOME_TYPE"] = "OBSERVED"
    frame["STOCKOUTS_ARE_OBSERVED"] = 1
    return frame
