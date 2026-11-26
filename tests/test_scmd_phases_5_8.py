import numpy as np
import pandas as pd
from src.scmd_phases_5_8 import adaptive_risk,monotonicity_sweep,simulate_scenarios,inventory_simulation,baseline_metrics,paired_bootstrap_holm
K=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]
def sample():
 n=10; d={k:[v]*n for k,v in zip(K,["ORG","000123","U"])}
 d.update(PERIOD=pd.date_range("2025-01-01",periods=n,freq="MS"),MODEL_DEMAND=np.arange(n)+5.,Q10=np.arange(n)+4.,Q50=np.arange(n)+5.,Q90=np.arange(n)+6.,ABS_ERROR=np.ones(n),SIGNED_ERROR=np.ones(n),CALIBRATED_COVERED=[1]*n,ROLLING_MAE=[1.]*n,ROLLING_BIAS=[.2]*n,ROLLING_COVERAGE=[.8]*n,DRIFT_SIGNAL=[False]*n)
 return pd.DataFrame(d)
def test_monotonicity(): assert monotonicity_sweep().RISK_MULTIPLIER.is_monotonic_decreasing
def test_simulation_and_statistics():
 x=adaptive_risk(sample()); s=simulate_scenarios(x); assert len(s)==70
 inv=inventory_simulation(s); assert set(inv.INVENTORY_OUTCOME)=={"SIMULATED"}
 b=baseline_metrics(s); assert b.BASELINE.nunique()==6
 assert len(paired_bootstrap_holm(b[["SERIES_ID","BASELINE","INTERVAL_SCORE"]],n_boot=100))==5
