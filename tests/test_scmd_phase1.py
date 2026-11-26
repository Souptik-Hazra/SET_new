import numpy as np
import pandas as pd
import pytest
from src.scmd_phase1 import compute_adi_cv2, classify_demand

def test_scmd_demand_classification_all_four_patterns():
    assert classify_demand(1.0, 0.2) == "Smooth"
    assert classify_demand(1.0, 0.8) == "Erratic"
    assert classify_demand(2.0, 0.2) == "Intermittent"
    assert classify_demand(2.0, 0.8) == "Lumpy"

def test_adi_uses_observed_periods_over_nonzero_periods():
    adi, cv2 = compute_adi_cv2([10, 0, 0, 20, 0, 30])
    assert adi == pytest.approx(2.0)
    assert cv2 >= 0

def test_all_zero_series_has_infinite_adi():
    adi, cv2 = compute_adi_cv2([0, 0, 0])
    assert np.isinf(adi)
    assert cv2 == 0
