import numpy as np
import pandas as pd
from src.set_forecasting.features.engineering import create_features,leakage_check
from src.set_forecasting.reliability.index import build_reliability_index
from src.set_forecasting.reliability.baselines import interval_baselines
from src.set_forecasting.risk.radcsr import radcsr_probability,monotonicity_audit
from src.set_forecasting.risk.simulation import make_scenarios,simulate_inventory,risk_baseline_probabilities

KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]

def sample(n=36):
    x={k:[v]*n for k,v in zip(KEYS,["ORG01","000123","U1"])}
    y=np.array([10,12,0,11,14,9]*6,dtype=float)
    x.update(PERIOD=pd.date_range("2023-01-01",periods=n,freq="MS"),
      MODEL_DEMAND=y,OBSERVATION_MISSING=[False]*n,
      SPLIT=["train"]*21+["validation"]*7+["test"]*8,
      POSITIVE_QUANTITY=y,TOTAL_INDICATIVE_COST=y*2)
    return pd.DataFrame(x)

def test_feature_engineering_is_past_only():
    x=sample();f=create_features(x)
    assert f.loc[1,"LAG_1"]==10
    assert f.loc[0,"LAG_1"]!=f.loc[0,"MODEL_DEMAND"]
    assert leakage_check(f)
    changed=x.copy();changed.loc[20,"MODEL_DEMAND"]=999999
    f2=create_features(changed)
    # Changing current target cannot alter its own predictor features.
    assert f.loc[20,"LAG_1"]==f2.loc[20,"LAG_1"]
    assert f.loc[20,"ROLL_MEAN_6"]==f2.loc[20,"ROLL_MEAN_6"]

def test_reliability_score_is_prequential():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND-2;x["Q50"]=x.MODEL_DEMAND-1;x["Q90"]=x.MODEL_DEMAND+2
    a=build_reliability_index(x)
    x.loc[x.index[5],"MODEL_DEMAND"]+=100
    b=build_reliability_index(x)
    assert a.loc[5,"R_REL"]==b.loc[5,"R_REL"]
    assert set(["R_E","R_B","R_C","R_D","R_REL"]).issubset(a.columns)

def test_reliability_baselines_are_all_exported():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND-2;x["Q50"]=x.MODEL_DEMAND-1;x["Q90"]=x.MODEL_DEMAND+2
    out=interval_baselines(x)
    assert set(out.METHOD.unique())=={"Static","RollingQuantile","EWMA","ACI","DtACI","AdaPI","EnbPI"}

def test_radcsr_monotone_probability_for_fixed_case():
    r=np.linspace(0,1,21)
    out=radcsr_probability(np.repeat(8.,len(r)),np.repeat(10.,len(r)),np.repeat(13.,len(r)),
      np.repeat(15.,len(r)),np.repeat(2.,len(r)),r)
    assert out.RADCSR_PROBABILITY.is_monotonic_increasing
    audit=monotonicity_audit(8,10,13,15,2)
    assert audit["envelope_pass"]

def test_seven_scenarios_are_isolated_and_inventory_is_simulated():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND*.8;x["Q50"]=x.MODEL_DEMAND;x["Q90"]=x.MODEL_DEMAND*1.2
    x["R_REL"]=.2;x["R_E"]=.1;x["R_B"]=.1;x["R_C"]=.1;x["R_D"]=0
    s=make_scenarios(x)
    assert len(s.SCENARIO_ID.unique())==7
    stable=s[s.SCENARIO_ID=="S1_Stable"]
    bias=s[s.SCENARIO_ID=="S5_Bias"]
    assert np.allclose(stable.Q50,x.Q50)
    assert np.allclose(bias.Q50,x.Q50*1.2)
    inv=simulate_inventory(s)
    assert (inv.STOCKOUTS_ARE_OBSERVED==0).all()
    assert set(inv.OUTCOME_TYPE)=={"SIMULATED"}

def test_risk_baseline_interface_returns_six_methods():
    x=sample().tail(10).copy()
    x["Q10"]=x.MODEL_DEMAND*.8;x["Q50"]=x.MODEL_DEMAND;x["Q90"]=x.MODEL_DEMAND*1.2
    x["R_REL"]=.2
    s=make_scenarios(x);inv=simulate_inventory(s)
    frame=s.merge(inv[KEYS+["PERIOD","SCENARIO_ID","AVAILABLE_STOCK","DEMAND"]],on=KEYS+["PERIOD","SCENARIO_ID"])
    p=risk_baseline_probabilities(frame)
    assert set(p.METHOD.unique())=={"Static","RollingQuantile","EWMA","ACI","MonteCarlo","WassersteinDRO"}
    assert p.SHORTAGE_PROBABILITY.between(0,1).all()

def test_end_to_end_small_cohort(tmp_path):
    from src.set_forecasting.pipeline import run_model_pipeline
    frames=[]
    for j in range(4):
        x=sample(36)
        x["ODS_CODE"]=f"ORG{j:02d}"
        x["MODEL_DEMAND"]=x.MODEL_DEMAND*(j+1)
        frames.append(x)
    series=pd.concat(frames,ignore_index=True)
    source=tmp_path/"series.parquet"
    series.to_parquet(source,index=False)
    report=run_model_pipeline(series_path=str(source),output_dir=str(tmp_path/"output"),
        processed_dir=str(tmp_path/"processed"),n_boot=100)
    assert report["prediction_rows"]>0
    assert report["inventory_rows"]>0
    assert report["ablation_rows"]>0
    assert (tmp_path/"output/tables/s5_bias_radcsr_vs_static_wasserstein.csv").exists()
