"""Phase 4: prequential interval calibration for monthly SCMD forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from src.scmd_phase1 import KEY,DISCLAIMER

def calibrate_intervals(predictions,target_coverage=.90,window=8,min_history=3):
    if not 0<target_coverage<1: raise ValueError("target_coverage must be between 0 and 1")
    if window<1 or min_history<1: raise ValueError("window and min_history must be positive")
    required=KEY+["PERIOD","MODEL_DEMAND","Q10","Q50","Q90"]
    missing=set(required)-set(predictions.columns)
    if missing: raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        return predictions.assign(CAL_Q10=pd.Series(dtype=float),CAL_Q90=pd.Series(dtype=float),
            ABS_ERROR=pd.Series(dtype=float),SIGNED_ERROR=pd.Series(dtype=float),
            CALIBRATED_COVERED=pd.Series(dtype="int8"),ROLLING_MAE=pd.Series(dtype=float),
            ROLLING_BIAS=pd.Series(dtype=float),ROLLING_COVERAGE=pd.Series(dtype=float),
            DRIFT_SIGNAL=pd.Series(dtype=bool))
    outputs=[]
    for _,g in predictions.groupby(KEY,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy()
        lows=[];highs=[];errors=[]
        for row in g.itertuples():
            hist=np.asarray(errors[-window:],float)
            radius=float(np.quantile(hist,target_coverage,method="higher")) if len(hist)>=min_history else max(float(row.Q90-row.Q50),float(row.Q50-row.Q10),0.)
            lows.append(max(0.,float(row.Q50)-radius));highs.append(float(row.Q50)+radius)
            # Observation is added only after the interval for this timestamp is fixed.
            errors.append(abs(float(row.MODEL_DEMAND)-float(row.Q50)))
        g["CAL_Q10"]=lows;g["CAL_Q90"]=highs
        g["ABS_ERROR"]=np.abs(g.MODEL_DEMAND-g.Q50);g["SIGNED_ERROR"]=g.MODEL_DEMAND-g.Q50
        g["CALIBRATED_COVERED"]=((g.MODEL_DEMAND>=g.CAL_Q10)&(g.MODEL_DEMAND<=g.CAL_Q90)).astype("int8")
        g["ROLLING_MAE"]=g.ABS_ERROR.shift(1).rolling(window,min_periods=1).mean()
        g["ROLLING_BIAS"]=g.SIGNED_ERROR.shift(1).rolling(window,min_periods=1).mean()
        g["ROLLING_COVERAGE"]=g.CALIBRATED_COVERED.shift(1).rolling(window,min_periods=1).mean()
        g["DRIFT_SIGNAL"]=(g.ROLLING_BIAS.abs()>g.ROLLING_MAE).fillna(False)
        outputs.append(g)
    return pd.concat(outputs,ignore_index=True) if outputs else predictions.copy()

def run_reliability(prediction_path="data/processed/scmd_test_predictions.parquet",output_dir="output",processed_dir="data/processed",target_coverage=None,window=None):
    config_path=Path("config/config.yaml")
    cfg=yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    rcfg=cfg.get("reliability",{})
    target_coverage=float(target_coverage if target_coverage is not None else rcfg.get("target_coverage",.90))
    window=int(window if window is not None else rcfg.get("window",8))
    p=pd.read_parquet(prediction_path)
    calibrated=calibrate_intervals(p,target_coverage,window)
    out=Path(output_dir);proc=Path(processed_dir);(out/"tables").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    calibrated.to_parquet(proc/"scmd_reliability.parquet",index=False)
    if len(calibrated):
        calibrated["RAW_COVERED"]=(calibrated.MODEL_DEMAND>=calibrated.Q10)&(calibrated.MODEL_DEMAND<=calibrated.Q90)
        summary=calibrated.groupby(KEY,dropna=False).agg(N=("MODEL_DEMAND","size"),MAE=("ABS_ERROR","mean"),
          BIAS=("SIGNED_ERROR","mean"),RAW_COVERAGE=("RAW_COVERED","mean"),
          CALIBRATED_COVERAGE=("CALIBRATED_COVERED","mean"),DRIFT_RATE=("DRIFT_SIGNAL","mean")).reset_index()
    else:
        summary=pd.DataFrame(columns=KEY+["N","MAE","BIAS","RAW_COVERAGE","CALIBRATED_COVERAGE","DRIFT_RATE"])
    summary.to_csv(out/"tables/phase4_reliability_summary.csv",index=False)
    report={"series_evaluated":len(summary),"prediction_rows":len(calibrated),"target_coverage":target_coverage,"window":window,
      "calibration":"prequential absolute-residual quantile; current residual only updates later intervals",
      "disclaimer":DISCLAIMER}
    (out/"phase4_reliability_report.json").write_text(json.dumps(report,indent=2));return calibrated,summary,report
if __name__=="__main__":run_reliability()
