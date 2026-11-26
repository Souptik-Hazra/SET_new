"""Phase 4: sequential reliability monitoring and online interval calibration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.scmd_phase1 import KEY, DISCLAIMER

def calibrate_intervals(predictions: pd.DataFrame, target_coverage=.80, window=12, min_history=3):
    required=KEY+["PERIOD","MODEL_DEMAND","Q10","Q50","Q90"]
    missing=set(required)-set(predictions.columns)
    if missing: raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    outputs=[]
    for _,g in predictions.groupby(KEY,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy()
        lows=[]; highs=[]; errors=[]; cover=[]; bias=[]
        for row in g.itertuples():
            # Calibrate using residuals strictly before the current timestamp.
            hist=np.asarray(errors[-window:],float)
            radius=float(np.quantile(hist,target_coverage)) if len(hist)>=min_history else max(float(row.Q90-row.Q50),float(row.Q50-row.Q10))
            lows.append(max(0.0,float(row.Q50)-radius)); highs.append(float(row.Q50)+radius)
            err=float(row.MODEL_DEMAND-row.Q50)
            errors.append(abs(err)); cover.append(float(lows[-1]<=row.MODEL_DEMAND<=highs[-1])); bias.append(err)
        g["CAL_Q10"]=lows;g["CAL_Q90"]=highs
        g["ABS_ERROR"]=np.abs(g.MODEL_DEMAND-g.Q50)
        g["SIGNED_ERROR"]=g.MODEL_DEMAND-g.Q50
        g["CALIBRATED_COVERED"]=((g.MODEL_DEMAND>=g.CAL_Q10)&(g.MODEL_DEMAND<=g.CAL_Q90)).astype(int)
        g["ROLLING_MAE"]=g.ABS_ERROR.shift(1).rolling(window,min_periods=1).mean()
        g["ROLLING_BIAS"]=g.SIGNED_ERROR.shift(1).rolling(window,min_periods=1).mean()
        g["ROLLING_COVERAGE"]=g.CALIBRATED_COVERED.shift(1).rolling(window,min_periods=1).mean()
        g["DRIFT_SIGNAL"]=(g.ROLLING_BIAS.abs()>g.ROLLING_MAE).fillna(False)
        outputs.append(g)
    return pd.concat(outputs,ignore_index=True) if outputs else predictions.copy()

def run_reliability(prediction_path="data/processed/scmd_test_predictions.parquet", output_dir="output", target_coverage=.80, window=12):
    p=pd.read_parquet(prediction_path)
    calibrated=calibrate_intervals(p,target_coverage,window)
    out=Path(output_dir);(out/"tables").mkdir(parents=True,exist_ok=True)
    calibrated.to_parquet("data/processed/scmd_reliability.parquet",index=False)
    summary=calibrated.groupby(KEY,dropna=False).agg(N=("MODEL_DEMAND","size"),MAE=("ABS_ERROR","mean"),BIAS=("SIGNED_ERROR","mean"),RAW_COVERAGE=("MODEL_DEMAND",lambda x:float(np.mean((x>=calibrated.loc[x.index,"Q10"])&(x<=calibrated.loc[x.index,"Q90"])))),CALIBRATED_COVERAGE=("CALIBRATED_COVERED","mean"),DRIFT_RATE=("DRIFT_SIGNAL","mean")).reset_index()
    summary.to_csv(out/"tables/phase4_reliability_summary.csv",index=False)
    report={"series_evaluated":len(summary),"prediction_rows":len(calibrated),"target_coverage":target_coverage,"calibration":"past-residual absolute-error quantile; no current/future residual used for current interval","disclaimer":DISCLAIMER}
    (out/"phase4_reliability_report.json").write_text(json.dumps(report,indent=2))
    return calibrated,summary,report

if __name__=="__main__": run_reliability()
