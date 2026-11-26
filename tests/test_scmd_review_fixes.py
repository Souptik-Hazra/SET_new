import numpy as np
import pandas as pd
import pytest
from src.features import create_features,leakage_check
from src.scmd_reliability import calibrate_intervals
from src.scmd_phases_5_8 import adaptive_risk,simulate_scenarios,inventory_simulation,baseline_metrics,paired_bootstrap_holm
from src.inventory.simulator import assign_ven_classes

K=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]
def sample(n=24):
 d={k:[v]*n for k,v in zip(K,["ORG","000123","U"])}
 d.update(PERIOD=pd.date_range("2024-01-01",periods=n,freq="MS"),MODEL_DEMAND=np.arange(n,dtype=float)+5,
  OBSERVATION_MISSING=[False]*n,SPLIT=["train"]*14+["validation"]*5+["test"]*(n-19))
 return pd.DataFrame(d)

def test_features_are_invariant_to_future_target_changes():
 x=sample();f=create_features(x)
 changed=x.copy();changed.loc[20:,"MODEL_DEMAND"]+=10000
 f2=create_features(changed)
 cols=["LAG_1","LAG_2","LAG_3","LAG_6","LAG_12","ROLL_MEAN_3","ROLL_STD_3","ROLL_MEAN_6","ROLL_STD_6","ROLL_MAX_6"]
 pd.testing.assert_frame_equal(f.loc[:19,cols],f2.loc[:19,cols])

def test_missing_month_is_not_zero_filled():
 x=sample();x.loc[5,"OBSERVATION_MISSING"]=True
 f=create_features(x)
 assert pd.isna(f.loc[6,"LAG_1"])
 assert leakage_check(f)

def test_prequential_calibration_is_prior_only():
 x=sample().tail(8).copy()
 x["Q10"]=x.MODEL_DEMAND-1;x["Q50"]=x.MODEL_DEMAND-0.5;x["Q90"]=x.MODEL_DEMAND+1
 c=calibrate_intervals(x,target_coverage=.9,window=4,min_history=2)
 changed=x.copy();changed.loc[changed.index[-1],"MODEL_DEMAND"]+=5000
 c2=calibrate_intervals(changed,target_coverage=.9,window=4,min_history=2)
 assert c.iloc[:-1].CAL_Q10.tolist()==c2.iloc[:-1].CAL_Q10.tolist()
 assert c.iloc[:-1].CAL_Q90.tolist()==c2.iloc[:-1].CAL_Q90.tolist()

def test_adaptive_risk_and_inventory_are_connected():
 x=sample().tail(10).copy()
 x["Q10"]=x.MODEL_DEMAND-1;x["Q50"]=x.MODEL_DEMAND;x["Q90"]=x.MODEL_DEMAND+1
 x["ROLLING_MAE"]=2.;x["ROLLING_BIAS"]=1.;x["ROLLING_COVERAGE"]=.7;x["DRIFT_SIGNAL"]=False
 r=adaptive_risk(x);s=simulate_scenarios(r);inv=inventory_simulation(s)
 assert "RISK_UPPER" in s and "ORDER_PLACED" in inv
 assert set(inv.INVENTORY_OUTCOME)=={"SIMULATED"}

def test_bootstrap_is_series_clustered_and_holm_adjusted():
 rows=[]
 for sid in range(12):
  for name,val in [("A",1.),("B",2.),("C",3.)]:
   rows.append({"SERIES_ID":str(sid),"BASELINE":name,"INTERVAL_SCORE":val+sid*.01})
 out=paired_bootstrap_holm(pd.DataFrame(rows),n_boot=200,seed=1)
 assert len(out)==2 and out.HOLM_ADJUSTED_P.between(0,1).all()

def test_synthetic_clinical_ven_assignment_is_rejected():
 with pytest.raises(ValueError,match="Clinical VEN"):
  assign_ven_classes(pd.DataFrame({"MEDICINE_ID":["a"],"DEMAND_CLEAN":[1]}))
