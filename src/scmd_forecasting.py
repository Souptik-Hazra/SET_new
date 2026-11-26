"""Phase 3: monthly SCMD probabilistic forecasting and honest holdout evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import create_features
from src.scmd_phase1 import KEY, DISCLAIMER

FEATURES=["LAG_1","LAG_2","LAG_3","LAG_6","LAG_12","ROLL_MEAN_3","ROLL_STD_3","ROLL_MEAN_6","ROLL_STD_6","ROLL_MAX_6","ZERO_RATIO_6","OBSERVED_RATIO_6","MONTH","QUARTER","YEAR","MONTH_SIN","MONTH_COS","TREND_INDEX"]

def seasonal_naive(g, period=12):
    y=g["MODEL_DEMAND"].where(~g["OBSERVATION_MISSING"])
    return y.shift(period)

def _metrics(y,lo,mid,hi):
    y=np.asarray(y,float); lo=np.asarray(lo,float); mid=np.asarray(mid,float); hi=np.asarray(hi,float)
    denom=max(float(np.abs(y).sum()),1e-9)
    pin=lambda q,p: np.maximum(q*(y-p),(q-1)*(y-p)).mean()
    return {"MAE":float(np.mean(np.abs(y-mid))),"RMSE":float(np.sqrt(np.mean((y-mid)**2))),
      "WAPE":float(np.abs(y-mid).sum()/denom*100),"PICP_80":float(np.mean((y>=lo)&(y<=hi))),
      "MPIW_80":float(np.mean(hi-lo)),"PINBALL_10":float(pin(.1,lo)),
      "PINBALL_50":float(pin(.5,mid)),"PINBALL_90":float(pin(.9,hi)),
      "QUANTILE_CROSSING_RATE":float(np.mean((lo>mid)|(mid>hi)))}

def run_forecasting(series_path="data/processed/scmd_monthly_series.parquet", output_dir="output"):
    df=pd.read_parquet(series_path)
    df["PERIOD"]=pd.to_datetime(df["PERIOD"])
    feat=create_features(df)
    feat.to_parquet(Path("data/processed/scmd_features.parquet"),index=False)
    # Optional LightGBM; deterministic seasonal-naive fallback keeps pipeline runnable.
    try:
        from lightgbm import LGBMRegressor
        import lightgbm as lgb
        model_available=True
    except ImportError: model_available=False
    rows=[]; predictions=[]
    for key,g in feat.groupby(KEY,sort=False,dropna=False):
        g=g.sort_values("PERIOD").copy()
        train=g[(g.SPLIT=="train") & g.TARGET_OBSERVED].dropna(subset=FEATURES+["MODEL_DEMAND"])
        val=g[(g.SPLIT=="validation") & g.TARGET_OBSERVED].dropna(subset=FEATURES+["MODEL_DEMAND"])
        test=g[(g.SPLIT=="test") & g.TARGET_OBSERVED].dropna(subset=FEATURES+["MODEL_DEMAND"])
        if len(train)<12 or len(test)==0: continue
        test=test.copy()
        if model_available and len(val)>=3:
            qs=[]
            for q in (.1,.5,.9):
                m=LGBMRegressor(objective="quantile",alpha=q,n_estimators=500,learning_rate=.03,num_leaves=15,verbosity=-1,random_state=42)
                m.fit(train[FEATURES],train.MODEL_DEMAND,eval_set=[(val[FEATURES],val.MODEL_DEMAND)],callbacks=[lgb.early_stopping(40,verbose=False)])
                qs.append(np.maximum(0,m.predict(test[FEATURES])))
            lo,mid,hi=qs
            model_name="QuantileLightGBM"
        else:
            pred=seasonal_naive(g).reindex(test.index).to_numpy()
            fallback=test[FEATURES].filter(regex="^LAG_1$").iloc[:,0].to_numpy()
            mid=np.where(np.isfinite(pred),pred,fallback); lo=np.maximum(0,mid*.5); hi=mid*1.5
            model_name="SeasonalNaive12_Fallback"
        y=test.MODEL_DEMAND.to_numpy(float)
        rec=dict(zip(KEY,key)); rec.update({"MODEL":model_name,"N_TEST":len(test),**_metrics(y,lo,mid,hi)})
        rows.append(rec)
        p=test[KEY+["PERIOD","MODEL_DEMAND"]].copy(); p["Q10"]=lo;p["Q50"]=mid;p["Q90"]=hi;p["MODEL"]=model_name
        predictions.append(p)
    out=Path(output_dir); (out/"tables").mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(rows); pred=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()
    metrics.to_csv(out/"tables/phase3_forecast_metrics.csv",index=False)
    pred.to_parquet("data/processed/scmd_test_predictions.parquet",index=False)
    report={"model_rows":len(metrics),"prediction_rows":len(pred),"model_available":"QuantileLightGBM" in set(metrics.MODEL) if len(metrics) else False,"disclaimer":DISCLAIMER,"target_note":"Positive quantity is a modelling proxy, not verified consumption. Missing months are excluded from observed-target training/evaluation."}
    (out/"phase3_forecasting_report.json").write_text(json.dumps(report,indent=2))
    return metrics,pred,report

if __name__=="__main__": run_forecasting()
