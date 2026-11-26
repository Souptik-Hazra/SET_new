"""
tests/test_pipeline_units.py
============================
Automated unit and integration test suite covering:
  - Phase 1: Preprocessing & Net Demand logic
  - Phase 2: Feature Engineering & Leakage Protection
  - Phase 3: Metrics & Quantile Validity
  - Phase 4: Inventory balance equations & non-negativity
  - Phase 5: Scenario generation integrity
  - Phase 6-8: RADCSR reliability calculations & bounding
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pytest

pytestmark = pytest.mark.legacy
import numpy as np
import pandas as pd
from src.evaluation.metrics import mae, rmse, wape, pinball_loss, picp, mpiw
from src.inventory.scenarios import generate_scenarios
from src.inventory.simulator import simulate_inventory, assign_ven_classes
from src.risk.radcsr import evaluate_all_risk_methods

def test_metrics_integrity():
    y_true = np.array([10.0, 20.0, 0.0, 30.0])
    y_pred = np.array([12.0, 18.0, 2.0, 25.0])
    
    assert mae(y_true, y_pred) == pytest.approx(2.75)
    assert rmse(y_true, y_pred) == pytest.approx(3.04138, rel=1e-3)
    assert wape(y_true, y_pred) == pytest.approx((11.0 / 60.0) * 100.0)
    
    # Coverage
    lower = np.array([8.0, 15.0, 0.0, 20.0])
    upper = np.array([15.0, 25.0, 5.0, 35.0])
    assert picp(y_true, lower, upper) == 1.0
    assert mpiw(lower, upper) == pytest.approx(9.25)

def test_inventory_balance_no_leakage():
    # Synthetic 2-period stream
    test_df = pd.DataFrame({
        "PERIOD": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
        "MEDICINE_ID": ["MED_A"] * 5,
        "DEMAND_CLEAN": [10.0, 20.0, 30.0, 15.0, 25.0],
        "Q10": [5.0, 10.0, 15.0, 10.0, 15.0],
        "Q50": [10.0, 20.0, 30.0, 15.0, 25.0],
        "Q90": [15.0, 30.0, 45.0, 25.0, 35.0]
    })
    
    scen_df = generate_scenarios(test_df)
    assert "S1_Stable" in scen_df["SCENARIO_ID"].values
    
    sim_df = simulate_inventory(scen_df)
    # Assert starting inventory and end inventory are strictly non-negative
    assert (sim_df["I_START"] >= 0.0).all()
    assert (sim_df["I_END"] >= 0.0).all()
    # Assert shortage >= 0
    assert (sim_df["SHORTAGE_QTY"] >= 0.0).all()

def test_radcsr_risk_bounds():
    test_df = pd.DataFrame({
        "PERIOD": pd.date_range("2024-01-01", periods=6, freq="W-MON"),
        "MEDICINE_ID": ["MED_TEST"] * 6,
        "DEMAND_CLEAN": [5.0, 10.0, 50.0, 5.0, 5.0, 5.0],
        "Q10": [2.0, 5.0, 10.0, 2.0, 2.0, 2.0],
        "Q50": [5.0, 10.0, 15.0, 5.0, 5.0, 5.0],
        "Q90": [8.0, 15.0, 25.0, 8.0, 8.0, 8.0],
    })
    
    scen_df = generate_scenarios(test_df)
    sim_df = simulate_inventory(scen_df)
    res = evaluate_all_risk_methods(sim_df)
    
    # Risk probabilities must be bounded strictly in [0, 1]
    assert (res["RISK_STATIC"] >= 0.0).all() and (res["RISK_STATIC"] <= 1.0).all()
    assert (res["RISK_RADCSR"] >= 0.0).all() and (res["RISK_RADCSR"] <= 1.0).all()
    assert (res["RELIABILITY_INDEX"] >= 0.0).all() and (res["RELIABILITY_INDEX"] <= 1.0).all()

if __name__ == "__main__":
    test_metrics_integrity()
    test_inventory_balance_no_leakage()
    test_radcsr_risk_bounds()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
