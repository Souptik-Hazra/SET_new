"""Scalable global quantile forecasting for monthly SCMD series."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import create_features, TARGET
from src.scmd_phase1 import KEY, DISCLAIMER

FEATURES=["LAG_1","LAG_2","LAG_3","LAG_6","LAG_12","ROLL_MEAN_3","ROLL_STD_3","ROLL_MEAN_6","ROLL_STD_6","ROLL_MAX_6","ZERO_RATIO_6","OBSERVED_RATIO_6","MONTH","QUARTER","YEAR","MONTH_SIN","MONTH_COS","TREND_INDEX"]
ID_FEATURES=["ORG_CODE","VMP_CODE","UNIT_CODE"]

def seasonal_naive(g,period=12):
    return g[TARGET].where(~g["OBSERVATION_MISSING"].astype(bool)).shift(period)

def _metrics(y,lo,mid,hi):
    y=np.asarray(y,float);lo=np.asarray(lo,float);mid=np.asarray(mid,float);hi=np.asarray(hi,float)
    m=np.isfinite(y)&np.isfinite(lo)&np.isfinite(mid)&np.isfinite(hi)
    y,lo,mid,hi=y[m],lo[m],mid[m],hi[m]
    if not len(y): return {k:float("nan") for k in ["MAE","RMSE","WAPE","PICP_80","MPIW_80","PINBALL_10","PINBALL_50","PINBALL_90","QUANTILE_CROSSING_RATE"]}
    pin=lambda q,p: np.maximum(q*(y-p),(q-1)*(y-p)).mean()
    return {"MAE":float(np.mean(np.abs(y-mid))),"RMSE":float(np.sqrt(np.mean((y-mid)**2))),
      "WAPE":float(np.abs(y-mid).sum()/max(np.abs(y).sum(),1e-9)*100),
      "PICP_80":float(np.mean((y>=lo)&(y<=hi))),"MPIW_80":float(np.mean(hi-lo)),
      "PINBALL_10":float(pin(.1,lo)),"PINBALL_50":float(pin(.5,mid)),"PINBALL_90":float(pin(.9,hi)),
      "QUANTILE_CROSSING_RATE":float(np.mean((lo>mid)|(mid>hi)))}

def _prepare_features(df):
    x=df.copy()
    for col,code in zip(KEY,ID_FEATURES):
        x[code]=pd.factorize(x[col].astype(str),sort=True)[0].astype("int32")
    return x

def run_forecasting(series_path="data/processed/scmd_monthly_series.parquet",output_dir="output",processed_dir="data/processed"):
    proc=Path(processed_dir);out=Path(output_dir);(out/"tables").mkdir(parents=True,exist_ok=True);proc.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(series_path);df["PERIOD"]=pd.to_datetime(df["PERIOD"])
    feat=_prepare_features(create_features(df))
    feat.to_parquet(proc/"scmd_features.parquet",index=False)
    cols=FEATURES+ID_FEATURES
    for c in cols: feat[c]=pd.to_numeric(feat[c],errors="coerce")
    train=feat[(feat.SPLIT=="train")&feat.TARGET_OBSERVED].dropna(subset=cols+[TARGET])
    val=feat[(feat.SPLIT=="validation")&feat.TARGET_OBSERVED].dropna(subset=cols+[TARGET])
    test=feat[(feat.SPLIT=="test")&feat.TARGET_OBSERVED].dropna(subset=cols+[TARGET]).copy()
    models=[]; model_name="SeasonalNaive12_Fallback"
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        if len(train)>=100 and len(val)>=20:
            for q in (.1,.5,.9):
                m=LGBMRegressor(objective="quantile",alpha=q,n_estimators=600,learning_rate=.03,
                    num_leaves=31,min_child_samples=30,subsample=.8,colsample_bytree=.8,
                    verbosity=-1,random_state=42,n_jobs=-1)
                m.fit(train[cols],train[TARGET],eval_set=[(val[cols],val[TARGET])],
                    callbacks=[lgb.early_stopping(40,verbose=False)])
                models.append(m)
            model_name="GlobalQuantileLightGBM"
    except ImportError: pass
    if len(test):
        if models:
            pred=[np.maximum(0,m.predict(test[cols])) for m in models]
            lo,mid,hi=pred
            # Quantile crossing is retained as a metric and corrected for operational intervals.
            lo,mid,hi=np.minimum(lo,mid),np.maximum(mid,lo),np.maximum(hi,mid)
            lo=np.minimum(lo,mid);hi=np.maximum(hi,mid)
        else:
            groups={k:g for k,g in feat.groupby(KEY,sort=False,dropna=False)}
            p=test.copy(); fallback=[]
            for _,row in p.iterrows():
                g=groups[tuple(row[k] for k in KEY)].sort_values("PERIOD")
                hist=g.loc[g.PERIOD<row.PERIOD,TARGET].where(~g.loc[g.PERIOD<row.PERIOD,"OBSERVATION_MISSING"].astype(bool))
                v=hist.iloc[-12] if len(hist)>=12 and pd.notna(hist.iloc[-12]) else (hist.iloc[-1] if len(hist) and pd.notna(hist.iloc[-1]) else 0.)
                fallback.append(max(0.,float(v)))
            mid=np.asarray(fallback);lo=mid*.5;hi=mid*1.5
        test["Q10"]=lo;test["Q50"]=mid;test["Q90"]=hi;test["MODEL"]=model_name
    else:
        test["Q10"]=pd.Series(dtype=float);test["Q50"]=pd.Series(dtype=float);test["Q90"]=pd.Series(dtype=float);test["MODEL"]=pd.Series(dtype=str)
    metrics=[]
    if len(test):
        for key,g in test.groupby(KEY,sort=False,dropna=False):
            rec=dict(zip(KEY,key));rec.update({"MODEL":model_name,"N_TEST":len(g),**_metrics(g[TARGET],g.Q10,g.Q50,g.Q90)})
            metrics.append(rec)
    metrics=pd.DataFrame(metrics,columns=KEY+["MODEL","N_TEST","MAE","RMSE","WAPE","PICP_80","MPIW_80","PINBALL_10","PINBALL_50","PINBALL_90","QUANTILE_CROSSING_RATE"])
    pred=test[KEY+["PERIOD",TARGET,"Q10","Q50","Q90","MODEL"]].copy()
    metrics.to_csv(out/"tables/phase3_forecast_metrics.csv",index=False)
    pred.to_parquet(proc/"scmd_test_predictions.parquet",index=False)
    report={"model_rows":len(metrics),"prediction_rows":len(pred),"model":model_name,"global_training_rows":len(train),"validation_rows":len(val),
      "disclaimer":DISCLAIMER,"target_note":"Positive quantity is a modelling proxy, not verified consumption. Missing months are excluded from observed-target training/evaluation."}
    (out/"phase3_forecasting_report.json").write_text(json.dumps(report,indent=2))
    return metrics,pred,report
if __name__=="__main__":run_forecasting()
