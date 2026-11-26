"""Global quantile LightGBM with previous-observation and seasonal-naive baselines."""
import numpy as np
import pandas as pd
from ..features.engineering import create_features,SERIES_KEYS,TARGET
FEATURES=["LAG_1","LAG_2","LAG_3","LAG_6","LAG_12","ROLL_MEAN_3","ROLL_MEDIAN_3","ROLL_STD_3","ROLL_CV_3","ROLL_MEAN_6","ROLL_MEDIAN_6","ROLL_STD_6","ROLL_CV_6","ROLL_MIN_6","ROLL_MAX_6","ZERO_RATIO_6","OBSERVED_RATIO_6","MONTHS_SINCE_NONZERO","PRIOR_NONZERO_MEAN_6","MOMENTUM_3M","SEASONAL_RATIO_12","MONTH","QUARTER","YEAR","MONTH_SIN","MONTH_COS","TREND_INDEX","LAG_UNIT_COST_1","ROLL_UNIT_COST_3"]

def _score(g,name):
    y=g.MODEL_DEMAND.to_numpy(float);lo=g.Q10.to_numpy(float);mid=g.Q50.to_numpy(float);hi=g.Q90.to_numpy(float)
    m=np.isfinite(y)&np.isfinite(lo)&np.isfinite(mid)&np.isfinite(hi);y,lo,mid,hi=y[m],lo[m],mid[m],hi[m]
    if not len(y):return {"MODEL":name,"N":0}
    pin=lambda q,p:float(np.mean(np.maximum(q*(y-p),(q-1)*(y-p))))
    return {"MODEL":name,"N":len(y),"MAE":float(np.mean(abs(y-mid))),
      "WAPE":float(abs(y-mid).sum()/max(abs(y).sum(),1e-9)*100),
      "PICP_80":float(np.mean((y>=lo)&(y<=hi))),"MPIW_80":float(np.mean(hi-lo)),
      "PINBALL_10":pin(.1,lo),"PINBALL_50":pin(.5,mid),"PINBALL_90":pin(.9,hi)}

def train_and_predict(series,min_train_rows=100,random_state=42):
    x=series.copy()
    # Train-only 99.5th percentile winsorisation; preserve original target for evaluation.
    x["MODEL_DEMAND_WINSORIZED"]=np.nan;x["OUTLIER_995_FLAG"]=False
    for _,idx in x.groupby(SERIES_KEYS,sort=False,dropna=False).groups.items():
        g=x.loc[idx];train=g[(g.SPLIT=="train")&(~g.OBSERVATION_MISSING.astype(bool))]
        positive=pd.to_numeric(train.MODEL_DEMAND,errors="coerce");positive=positive[positive>0]
        threshold=float(positive.quantile(.995)) if len(positive) else np.inf
        values=pd.to_numeric(g.MODEL_DEMAND,errors="coerce")
        x.loc[idx,"OUTLIER_995_FLAG"]=values>threshold
        x.loc[idx,"MODEL_DEMAND_WINSORIZED"]=np.where(values>0,np.minimum(values,threshold),values)
    # Feature lags are generated from winsorised history, never from the full target.
    original=x.MODEL_DEMAND.copy();x[TARGET]=x.MODEL_DEMAND_WINSORIZED
    features=create_features(x)
    features["MODEL_DEMAND"]=original.loc[features.index].to_numpy()
    features["SERIES_ID"]=features[SERIES_KEYS].astype(str).agg("|".join,axis=1)
    categorical=[]
    for c in SERIES_KEYS:
        name=c+"_CAT";features[name]=features[c].astype("category");categorical.append(name)
    cols=[c for c in FEATURES if c in features]+categorical
    for c in FEATURES:
        if c in features:features[c]=pd.to_numeric(features[c],errors="coerce")
    train=features[(features.SPLIT=="train")&features.TARGET_OBSERVED].dropna(subset=cols+[TARGET])
    val=features[(features.SPLIT=="validation")&features.TARGET_OBSERVED].dropna(subset=cols+[TARGET])
    test=features[(features.SPLIT=="test")&features.TARGET_OBSERVED].dropna(subset=["MODEL_DEMAND"])
    models=[];status="naive_fallback"
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        if len(train)>=min_train_rows and len(val)>=20:
            for q in (.1,.5,.9):
                m=LGBMRegressor(objective="quantile",alpha=q,n_estimators=800,learning_rate=.03,
                  num_leaves=31,min_child_samples=30,subsample=.85,colsample_bytree=.85,
                  verbosity=-1,random_state=random_state,n_jobs=-1)
                m.fit(train[cols],train[TARGET],categorical_feature=categorical,eval_set=[(val[cols],val[TARGET])],
                  callbacks=[lgb.early_stopping(50,verbose=False)])
                models.append(m)
            status="quantile_lightgbm"
    except ImportError:pass
    if len(test):
        if models:
            raw=np.column_stack([np.maximum(0,m.predict(test[cols])) for m in models])
            test[["Q10","Q50","Q90"]]=np.sort(raw,axis=1)
        else:
            test["Q50"]=test.LAG_1.fillna(test.LAG_12).fillna(0).clip(lower=0)
            test["Q10"]=test.Q50*.5;test["Q90"]=test.Q50*1.5
        test["FORECAST_MODEL"]=status
        test["NAIVE_LAST"]=test.LAG_1.fillna(0).clip(lower=0)
        test["NAIVE_SEASONAL_12"]=test.LAG_12.fillna(test.NAIVE_LAST).clip(lower=0)
    metrics=[]
    if len(test):
        metrics.append(_score(test,"QuantileLightGBM" if status=="quantile_lightgbm" else "NaiveFallback"))
        for col,name in [("NAIVE_LAST","PreviousObservationNaive"),("NAIVE_SEASONAL_12","SeasonalNaive12")]:
            z=test.copy();z["Q10"]=z[col]*.5;z["Q50"]=z[col];z["Q90"]=z[col]*1.5;metrics.append(_score(z,name))
    return features,test,pd.DataFrame(metrics),{"model":status,"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test)}
