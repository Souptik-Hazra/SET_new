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
    x["MODEL_DEMAND_WINSORIZED"]=np.nan;x["OUTLIER_995_FLAG"]=False
    x["_ORIGINAL_MODEL_DEMAND"]=pd.to_numeric(x.MODEL_DEMAND,errors="coerce")
    for _,idx in x.groupby(SERIES_KEYS,sort=False,dropna=False).groups.items():
        g=x.loc[idx];train_g=g[(g.SPLIT=="train")&(~g.OBSERVATION_MISSING.astype(bool))]
        positive=pd.to_numeric(train_g.MODEL_DEMAND,errors="coerce");positive=positive[positive>0]
        threshold=float(positive.quantile(.995)) if len(positive) else np.inf
        values=pd.to_numeric(g.MODEL_DEMAND,errors="coerce")
        x.loc[idx,"OUTLIER_995_FLAG"]=values>threshold
        x.loc[idx,"MODEL_DEMAND_WINSORIZED"]=np.where(values>0,np.minimum(values,threshold),values)
    x[TARGET]=x.MODEL_DEMAND_WINSORIZED
    features=create_features(x)
    features["MODEL_DEMAND"]=features["_ORIGINAL_MODEL_DEMAND"]
    features["SERIES_ID"]=features[SERIES_KEYS].astype(str).agg("|".join,axis=1)
    categorical=[]
    for col in SERIES_KEYS:
        name=col+"_CAT";features[name]=features[col].astype("category");categorical.append(name)
    cols=[c for c in FEATURES if c in features.columns]+categorical
    for col in FEATURES:
        if col in features:features[col]=pd.to_numeric(features[col],errors="coerce")
    train=features[(features.SPLIT=="train")&features.TARGET_OBSERVED].dropna(subset=["MODEL_DEMAND_WINSORIZED"])
    val=features[(features.SPLIT=="validation")&features.TARGET_OBSERVED].dropna(subset=["MODEL_DEMAND_WINSORIZED"])
    test=features[(features.SPLIT=="test")&features.TARGET_OBSERVED].dropna(subset=["MODEL_DEMAND"])
    models=[];status="naive_fallback"
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        if len(train)>=min_train_rows and len(val)>=20:
            for quantile in (.1,.5,.9):
                model=LGBMRegressor(objective="quantile",alpha=quantile,n_estimators=800,
                    learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.85,
                    colsample_bytree=.85,verbosity=-1,random_state=random_state,n_jobs=-1)
                model.fit(train[cols],train["MODEL_DEMAND_WINSORIZED"],categorical_feature=categorical,
                    eval_set=[(val[cols],val["MODEL_DEMAND_WINSORIZED"])],
                    callbacks=[lgb.early_stopping(50,verbose=False)])
                models.append(model)
            status="quantile_lightgbm"
    except ImportError:
        pass
    def predict(part):
        part=part.copy()
        if part.empty:return part
        if models:
            raw=np.column_stack([np.maximum(0,m.predict(part[cols])) for m in models])
            part[["Q10","Q50","Q90"]]=np.sort(raw,axis=1)
        else:
            part["Q50"]=part.LAG_1.fillna(part.LAG_12).fillna(0).clip(lower=0)
            part["Q10"]=part.Q50*.5;part["Q90"]=part.Q50*1.5
        part["FORECAST_MODEL"]=status
        part["NAIVE_LAST"]=part.LAG_1.fillna(0).clip(lower=0)
        part["NAIVE_SEASONAL_12"]=part.LAG_12.fillna(part.NAIVE_LAST).clip(lower=0)
        return part
    validation_pred=predict(val);test_pred=predict(test)
    all_predictions=pd.concat([validation_pred,test_pred],ignore_index=True)
    metrics=[]
    if len(test_pred):
        metrics.append(_score(test_pred,"QuantileLightGBM" if status=="quantile_lightgbm" else "NaiveFallback"))
        for col,name in [("NAIVE_LAST","PreviousObservationNaive"),("NAIVE_SEASONAL_12","SeasonalNaive12")]:
            z=test_pred.copy();z["Q10"]=z[col]*.5;z["Q50"]=z[col];z["Q90"]=z[col]*1.5
            metrics.append(_score(z,name))
    report={"model":status,"train_rows":len(train),"validation_rows":len(val),
      "test_rows":len(test_pred),"validation_predictions":len(validation_pred)}
    return features,all_predictions,pd.DataFrame(metrics),report
