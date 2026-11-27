"""
run_phase3_forecasting.py  (v2 — global ML strategy)
======================================================
Phase 3: Train all 5 forecasting baselines and evaluate on the test set.

Strategy:
  - Seasonal Naive:    per-medicine (look-back only)
  - SARIMA:            per-medicine, top-50 by total demand only
  - XGBoost:           GLOBAL model (all medicines, medicine features included)
  - LightGBM:          GLOBAL model
  - Quantile LightGBM: GLOBAL model — frozen primary probabilistic model

Chronological split (per medicine): 60% train / 20% val / 20% test
No shuffling. No future leakage.
"""

import os, sys, json, logging, warnings, hashlib, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
SEED = 42
OBSERVED_DEMAND_DISCLAIMER = "Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.forecasting.seasonal_naive import SeasonalNaive
from src.forecasting.sarima import SARIMAModel
from src.evaluation.metrics import compute_all_metrics

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LGB_OK = True
except ImportError:
    LGB_OK = False

# ---------------------------------------------------------------------------
# Feature columns (must match features.py output)
# ---------------------------------------------------------------------------
FEATURE_COLS_WEEKLY = [
    "lag_1","lag_2","lag_3","lag_4",
    "roll_mean_4","roll_std_4","roll_max_4","zero_ratio_4",
    "roll_mean_8","roll_std_8","roll_max_8","zero_ratio_8",
    "month","quarter","sin_annual","cos_annual","trend_index",
    "adi_4","cv2_4","days_since_last_issue"
]
FEATURE_COLS_DAILY = [
    "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6","lag_7",
    "roll_mean_7","roll_std_7","roll_max_7","zero_ratio_7",
    "roll_mean_14","roll_std_14","roll_max_14","zero_ratio_14",
    "month","quarter","dow","sin_annual","cos_annual","trend_index",
    "adi_7","cv2_7","days_since_last_issue"
]
DEMAND_COL = "DEMAND_CLEAN"


# ---------------------------------------------------------------------------
# 1. Chronological split per medicine, then pool for global ML
# ---------------------------------------------------------------------------
def make_global_splits(features_df: pd.DataFrame,
                       train_frac=0.60, val_frac=0.20):
    """
    Per-medicine chronological split.
    Returns train_df, val_df, test_df  (all medicines pooled, with MEDICINE_ID column).
    Also returns a dict: {medicine_id -> (n_train, n_val, n_test)}.
    """
    train_parts, val_parts, test_parts = [], [], []
    split_map = {}

    for med, grp in features_df.groupby("MEDICINE_ID"):
        grp = grp.sort_values("PERIOD").reset_index(drop=True)
        n = len(grp)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        tr = grp.iloc[:n_train]
        va = grp.iloc[n_train:n_train+n_val]
        te = grp.iloc[n_train+n_val:]
        if len(te) < 2:
            continue
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        split_map[med] = (len(tr), len(va), len(te))

    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(val_parts,  ignore_index=True),
            pd.concat(test_parts, ignore_index=True),
            split_map)


def get_feature_cols(df: pd.DataFrame) -> list:
    # Majority granularity
    gran = df["GRANULARITY"].mode()[0] if "GRANULARITY" in df.columns else "weekly"
    cands = FEATURE_COLS_DAILY if gran == "daily" else FEATURE_COLS_WEEKLY
    return [c for c in cands if c in df.columns]


# ---------------------------------------------------------------------------
# 2. Seasonal Naive (per-medicine)
# ---------------------------------------------------------------------------
def run_seasonal_naive(features_df: pd.DataFrame, split_map: dict) -> pd.DataFrame:
    rows = []
    for med, grp in features_df.groupby("MEDICINE_ID"):
        if med not in split_map:
            continue
        grp = grp.sort_values("PERIOD").reset_index(drop=True)
        gran = grp["GRANULARITY"].iloc[0]
        n_tr, n_va, n_te = split_map[med]
        sn = SeasonalNaive(granularity=gran)
        all_preds = sn.predict(grp[DEMAND_COL].values)
        test_preds = all_preds[n_tr+n_va:n_tr+n_va+n_te]
        y_test = grp[DEMAND_COL].values[n_tr+n_va:n_tr+n_va+n_te]
        m = compute_all_metrics(y_test, test_preds)
        m.update({"MEDICINE_ID":med,"MODEL":"SeasonalNaive","GRANULARITY":gran,
                  "N_TRAIN":n_tr,"N_VAL":n_va,"N_TEST":n_te})
        rows.append(m)
    logger.info(f"SeasonalNaive: evaluated {len(rows)} medicines")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. SARIMA (top-50 by total demand, per-medicine)
# ---------------------------------------------------------------------------
def run_sarima(features_df: pd.DataFrame, split_map: dict, top_n=50) -> pd.DataFrame:
    # Select top-N medicines by total demand
    total_demand = (features_df.groupby("MEDICINE_ID")[DEMAND_COL]
                    .sum().sort_values(ascending=False))
    top_meds = total_demand.head(top_n).index.tolist()
    rows = []
    for med in top_meds:
        if med not in split_map:
            continue
        grp = features_df[features_df["MEDICINE_ID"]==med].sort_values("PERIOD").reset_index(drop=True)
        gran = grp["GRANULARITY"].iloc[0]
        n_tr, n_va, n_te = split_map[med]
        train_arr = grp[DEMAND_COL].values[:n_tr]
        y_test    = grp[DEMAND_COL].values[n_tr+n_va:n_tr+n_va+n_te]
        try:
            sm = SARIMAModel(granularity=gran)
            sm.fit(train_arr)
            res = sm.predict_one_step(train_arr, n_te)
            m = compute_all_metrics(y_test, res["forecast"],
                                    lower_90=res["lower_90"], upper_90=res["upper_90"])
            m.update({"MEDICINE_ID":med,"MODEL":"SARIMA","GRANULARITY":gran,
                      "N_TRAIN":n_tr,"N_VAL":n_va,"N_TEST":n_te})
            rows.append(m)
            logger.info(f"  SARIMA done: {med}")
        except Exception as e:
            logger.warning(f"  SARIMA failed for {med}: {e}")
    logger.info(f"SARIMA: evaluated {len(rows)}/{top_n} medicines")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Global XGBoost
# ---------------------------------------------------------------------------
def run_xgboost(train_df, val_df, test_df, split_map) -> pd.DataFrame:
    if not XGB_OK:
        return pd.DataFrame()
    feat_cols = get_feature_cols(train_df)
    X_tr = train_df[feat_cols].fillna(0);  y_tr = train_df[DEMAND_COL].values
    X_va = val_df[feat_cols].fillna(0);    y_va = val_df[DEMAND_COL].values
    X_te = test_df[feat_cols].fillna(0);   y_te = test_df[DEMAND_COL].values

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=50
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    logger.info(f"XGBoost global best iteration: {model.best_iteration}")

    preds_global = np.maximum(model.predict(X_te), 0)
    test_df = test_df.copy()
    test_df["XGB_PRED"] = preds_global

    rows = []
    for med, grp in test_df.groupby("MEDICINE_ID"):
        if med not in split_map: continue
        n_tr, n_va, n_te = split_map[med]
        gran = grp["GRANULARITY"].iloc[0]
        m = compute_all_metrics(grp[DEMAND_COL].values, grp["XGB_PRED"].values)
        m.update({"MEDICINE_ID":med,"MODEL":"XGBoost","GRANULARITY":gran,
                  "N_TRAIN":n_tr,"N_VAL":n_va,"N_TEST":n_te})
        rows.append(m)
    logger.info(f"XGBoost: evaluated {len(rows)} medicines")

    # Save model
    model_path = os.path.join(PROJECT_ROOT, "data","processed","models","xgboost_global.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f: pickle.dump(model, f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Global LightGBM
# ---------------------------------------------------------------------------
def run_lightgbm(train_df, val_df, test_df, split_map) -> pd.DataFrame:
    if not LGB_OK:
        return pd.DataFrame()
    feat_cols = get_feature_cols(train_df)
    X_tr = train_df[feat_cols].fillna(0);  y_tr = train_df[DEMAND_COL].values
    X_va = val_df[feat_cols].fillna(0);    y_va = val_df[DEMAND_COL].values
    X_te = test_df[feat_cols].fillna(0);   y_te = test_df[DEMAND_COL].values

    model = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        objective="regression", random_state=SEED, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    logger.info(f"LightGBM global best iteration: {model.best_iteration_}")

    preds_global = np.maximum(model.predict(X_te), 0)
    test_df = test_df.copy()
    test_df["LGB_PRED"] = preds_global

    rows = []
    for med, grp in test_df.groupby("MEDICINE_ID"):
        if med not in split_map: continue
        n_tr, n_va, n_te = split_map[med]
        gran = grp["GRANULARITY"].iloc[0]
        m = compute_all_metrics(grp[DEMAND_COL].values, grp["LGB_PRED"].values)
        m.update({"MEDICINE_ID":med,"MODEL":"LightGBM","GRANULARITY":gran,
                  "N_TRAIN":n_tr,"N_VAL":n_va,"N_TEST":n_te})
        rows.append(m)
    logger.info(f"LightGBM: evaluated {len(rows)} medicines")

    model_path = os.path.join(PROJECT_ROOT, "data","processed","models","lightgbm_global.pkl")
    with open(model_path, "wb") as f: pickle.dump(model, f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Global Quantile LightGBM (PRIMARY, frozen after this)
# ---------------------------------------------------------------------------
def run_quantile_lgbm(train_df, val_df, test_df, split_map) -> tuple:
    if not LGB_OK:
        return pd.DataFrame(), None
    feat_cols = get_feature_cols(train_df)
    X_tr = train_df[feat_cols].fillna(0);  y_tr = train_df[DEMAND_COL].values
    X_va = val_df[feat_cols].fillna(0);    y_va = val_df[DEMAND_COL].values
    X_te = test_df[feat_cols].fillna(0)

    quantile_models = {}
    for q in [0.10, 0.50, 0.90]:
        m = LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            objective="quantile", alpha=q,
            random_state=SEED, n_jobs=-1, verbose=-1
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        quantile_models[q] = m
        logger.info(f"  QuantileLGBM q={q:.2f} fitted. Best iter: {m.best_iteration_}")

    # Compute version hash
    h = hashlib.sha256(pd.util.hash_pandas_object(train_df[feat_cols].fillna(0), index=True).values.tobytes())
    version_hash = h.hexdigest()[:12]
    logger.info(f"QuantileLGBM version hash: {version_hash}")

    q10 = np.maximum(quantile_models[0.10].predict(X_te), 0)
    q50 = np.maximum(quantile_models[0.50].predict(X_te), 0)
    q90 = np.maximum(quantile_models[0.90].predict(X_te), 0)

    test_df = test_df.copy()
    test_df["Q10"] = q10; test_df["Q50"] = q50; test_df["Q90"] = q90

    rows = []
    for med, grp in test_df.groupby("MEDICINE_ID"):
        if med not in split_map: continue
        n_tr, n_va, n_te = split_map[med]
        gran = grp["GRANULARITY"].iloc[0]
        m = compute_all_metrics(
            grp[DEMAND_COL].values,
            y_pred=grp["Q50"].values,
            lower_90=grp["Q10"].values,
            upper_90=grp["Q90"].values,
            q10_pred=grp["Q10"].values,
            q50_pred=grp["Q50"].values,
            q90_pred=grp["Q90"].values
        )
        m.update({"MEDICINE_ID":med,"MODEL":"QuantileLGBM","GRANULARITY":gran,
                  "N_TRAIN":n_tr,"N_VAL":n_va,"N_TEST":n_te,
                  "VERSION_HASH": version_hash})
        rows.append(m)
    logger.info(f"QuantileLGBM: evaluated {len(rows)} medicines")

    # Save predictions and frozen model
    pred_path = os.path.join(PROJECT_ROOT,"data","processed","quantile_lgbm_test_predictions.csv")
    prediction_export = test_df[["PERIOD","MEDICINE_ID",DEMAND_COL,"Q10","Q50","Q90"]].copy()
    prediction_export["DATA_TYPE"] = "OBSERVED_DEMAND"
    prediction_export["DISCLAIMER"] = OBSERVED_DEMAND_DISCLAIMER
    prediction_export.to_csv(pred_path, index=False)
    test_df["DATA_TYPE"] = "OBSERVED_DEMAND"
    test_df["DISCLAIMER"] = OBSERVED_DEMAND_DISCLAIMER

    model_path = os.path.join(PROJECT_ROOT,"data","processed","models","quantile_lgbm_global.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"models": quantile_models, "feat_cols": feat_cols,
                     "version_hash": version_hash}, f)
    logger.info(f"Frozen QuantileLGBM saved to {model_path}")

    return pd.DataFrame(rows), test_df


# ---------------------------------------------------------------------------
# Gate 3 analysis
# ---------------------------------------------------------------------------
def gate3_analysis(metrics_df: pd.DataFrame) -> dict:
    sn    = metrics_df[metrics_df["MODEL"]=="SeasonalNaive"]
    qlgbm = metrics_df[metrics_df["MODEL"]=="QuantileLGBM"]

    median_wape = qlgbm["WAPE"].median()
    c1 = {"metric":"Median WAPE (QuantileLGBM)","value":round(float(median_wape),2),
          "threshold":50.0,"status":"PASS" if median_wape < 50 else "FAIL"}

    merged = pd.merge(qlgbm[["MEDICINE_ID","WAPE"]].rename(columns={"WAPE":"W_Q"}),
                      sn[["MEDICINE_ID","WAPE"]].rename(columns={"WAPE":"W_SN"}),
                      on="MEDICINE_ID").dropna()
    pct = (merged["W_Q"] < merged["W_SN"]).mean()*100 if len(merged)>0 else np.nan
    c2 = {"metric":"QuantileLGBM beats SeasonalNaive (% medicines)","value":round(float(pct),1),
          "threshold":60.0,"status":"PASS" if pct>=60 else "FLAG"}

    picp_med = qlgbm["PICP_90"].median() if "PICP_90" in qlgbm.columns else np.nan
    if np.isnan(picp_med):  st3="NO_DATA"
    elif 0.75<=picp_med<=0.98: st3="PASS"
    elif picp_med<0.75: st3="FAIL_UNDERCOVERAGE"
    else: st3="FLAG_OVERCOVERAGE"
    c3 = {"metric":"Median PICP @ 90% PI","value":round(float(picp_med),4),
          "threshold":"[0.75, 0.98]","status":st3}

    cov_err = qlgbm["COVERAGE_ERROR_90"].median() if "COVERAGE_ERROR_90" in qlgbm.columns else np.nan
    c4 = {"metric":"Median Coverage Error |PICP-0.90|","value":round(float(cov_err),4),
          "threshold":0.15,"status":"PASS" if (not np.isnan(cov_err) and cov_err<0.15) else "FLAG"}

    n_meds = qlgbm["MEDICINE_ID"].nunique()
    c5 = {"metric":"Medicines with valid forecasts","value":int(n_meds),
          "threshold":15,"status":"PASS" if n_meds>=15 else "STOP"}

    checks = [c1,c2,c3,c4,c5]
    if any(c["status"]=="STOP" for c in checks):         overall="STOP"
    elif any(c["status"].startswith("FAIL") for c in checks): overall="FAIL"
    elif any(c["status"]=="FLAG" for c in checks):        overall="FLAG"
    else:                                                  overall="PASS"
    return {"gate3_checks":checks,"overall":overall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    feat_path  = os.path.join(PROJECT_ROOT,"data","processed","features.parquet")
    tables_dir = os.path.join(PROJECT_ROOT,"output","tables")
    output_dir = os.path.join(PROJECT_ROOT,"output")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT,"data","processed","models"), exist_ok=True)

    logger.info(f"Loading features …")
    df = pd.read_parquet(feat_path)
    logger.info(f"Features: {df.shape}")

    # Rename DEMAND_CLEAN if needed (features.py outputs DEMAND column)
    if "DEMAND_CLEAN" not in df.columns and "DEMAND" in df.columns:
        df = df.rename(columns={"DEMAND":"DEMAND_CLEAN"})

    logger.info("Building chronological splits …")
    train_df, val_df, test_df, split_map = make_global_splits(df)
    logger.info(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    logger.info(f"  Medicines with valid test set: {len(split_map):,}")

    all_metrics = []

    # 1. Seasonal Naive
    logger.info("--- Seasonal Naive ---")
    sn_df = run_seasonal_naive(df, split_map)
    all_metrics.append(sn_df)

    # 2. SARIMA (top-50)
    logger.info("--- SARIMA (top-50 medicines) ---")
    sarima_df = run_sarima(df, split_map, top_n=50)
    all_metrics.append(sarima_df)

    # 3. XGBoost global
    logger.info("--- XGBoost (global) ---")
    xgb_df = run_xgboost(train_df, val_df, test_df, split_map)
    all_metrics.append(xgb_df)

    # 4. LightGBM global
    logger.info("--- LightGBM (global) ---")
    lgb_df = run_lightgbm(train_df, val_df, test_df, split_map)
    all_metrics.append(lgb_df)

    # 5. Quantile LightGBM global
    logger.info("--- Quantile LightGBM (global, PRIMARY) ---")
    qlgbm_df, q_test_df = run_quantile_lgbm(train_df, val_df, test_df, split_map)
    all_metrics.append(qlgbm_df)

    # Combine and save
    metrics_df = pd.concat([m for m in all_metrics if len(m)>0], ignore_index=True)
    metrics_df["DATA_TYPE"] = "OBSERVED_DEMAND"
    metrics_df["DISCLAIMER"] = OBSERVED_DEMAND_DISCLAIMER
    metrics_path = os.path.join(tables_dir,"phase3_forecast_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"All metrics saved → {metrics_path}")

    # Gate 3
    gate = gate3_analysis(metrics_df)
    gate["DATA_TYPE"] = "OBSERVED_DEMAND"
    gate["disclaimer"] = OBSERVED_DEMAND_DISCLAIMER
    gate_path = os.path.join(output_dir,"phase3_gate3_analysis.json")
    with open(gate_path,"w") as f: json.dump(gate, f, indent=2, default=str)

    # Print report
    print("\n" + "="*65)
    print("PHASE 3 COMPLETE — GATE 3 ANALYSIS")
    print("="*65)
    for c in gate["gate3_checks"]:
        icon = "✅" if c["status"]=="PASS" else ("⚠️ " if c["status"]=="FLAG" else "❌")
        print(f"{icon} {c['metric']}")
        print(f"   Value: {c['value']}  |  Threshold: {c['threshold']}  → {c['status']}")
    print(f"\nOVERALL GATE 3: {gate['overall']}")
    print("="*65)

    print("\n--- Median metrics by model (over all medicines) ---")
    cols = [c for c in ["MAE","RMSE","WAPE","PICP_90","COVERAGE_ERROR_90",
                         "PINBALL_10","PINBALL_50","PINBALL_90"] if c in metrics_df.columns]
    agg = metrics_df.groupby("MODEL")[cols].median().round(4)
    print(agg.to_string())

    print("\n--- Model comparison vs SeasonalNaive (median WAPE) ---")
    wape_comp = metrics_df.groupby("MODEL")["WAPE"].median().sort_values()
    for model, wape in wape_comp.items():
        print(f"  {model:<20} WAPE={wape:.2f}%")

    if gate["overall"] in ("STOP","FAIL"):
        print("\n⛔  Review required before proceeding to Phase 4.")
    elif gate["overall"] == "FLAG":
        print("\n⚠️   Some checks flagged — review before Phase 4.")
    else:
        print("\n✅  Gate 3 PASSED — ready for Phase 4.")

    return metrics_df, gate


if __name__ == "__main__":
    main()
