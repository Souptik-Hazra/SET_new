"""
preprocessing.py
================
Phase 1: Dataset Construction

Steps performed:
  1. Filter to Drug type only
  2. Parse dates
  3. Compute net demand per (DATE, MEDICINE_ID) with floor at 0
  4. Determine granularity per medicine (daily if >=60% non-zero, else weekly)
  5. Reindex to complete calendar, fill gaps with 0
  6. Apply inclusion filter (min periods, min non-zero)
  7. Outlier flagging, handling, and winsorisation
  8. Compute per-medicine diagnostics (ADI, CV2, classification)
  9. Save diagnostics JSON and processed series CSV
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Load + basic filter
# ---------------------------------------------------------------------------
def filter_drugs(df: pd.DataFrame, type_col: str = "type", type_val: str = "Drug") -> pd.DataFrame:
    """Keep only Drug rows; log count of dropped Supply rows."""
    n_before = len(df)
    df = df[df[type_col].str.strip().str.lower() == type_val.lower()].copy()
    n_after = len(df)
    logger.info(f"Type filter: {n_before:,} → {n_after:,} rows (dropped {n_before - n_after:,} non-Drug rows)")
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "addeddate") -> pd.DataFrame:
    """Parse addeddate to datetime; drop rows that cannot be parsed."""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")
    n_bad = df["DATE"].isna().sum()
    if n_bad > 0:
        logger.warning(f"{n_bad:,} rows dropped: unparseable date in '{date_col}'")
        df = df.dropna(subset=["DATE"])
    return df


# ---------------------------------------------------------------------------
# 2. Net demand aggregation
# ---------------------------------------------------------------------------
def compute_net_demand(df: pd.DataFrame,
                       qty_col: str = "Sheet",
                       sales_pack_col: str = "Sales_pack",
                       medicine_col: str = "name") -> pd.DataFrame:
    """
    Aggregate to (DATE, MEDICINE_ID) level using vectorized operations.
    Returns = rows where Sales_pack == 0 (returned items) → subtract from gross.
    Net demand = sum(qty) - sum(qty where Sales_pack==0).
    Floor at 0.
    """
    df = df.copy()
    df["MEDICINE_ID"] = df[medicine_col].str.strip().str.upper()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df[sales_pack_col] = pd.to_numeric(df[sales_pack_col], errors="coerce").fillna(1)

    # Mark returns: Sales_pack == 0 means no packs sold (return/refund)
    df["_is_return"] = (df[sales_pack_col] == 0).astype(int)
    df["_return_qty"] = df[qty_col] * df["_is_return"]

    key = ["DATE", "MEDICINE_ID"]

    gross = df.groupby(key)[qty_col].sum().rename("GROSS_QTY")
    returns = df.groupby(key)["_return_qty"].sum().rename("RETURN_QTY")
    n_trans = df.groupby(key).size().rename("NUM_TRANSACTIONS")
    n_ret = df.groupby(key)["_is_return"].sum().rename("NUM_RETURNS")

    grouped = pd.concat([gross, returns, n_trans, n_ret], axis=1).reset_index()
    grouped["NET_DEMAND"] = grouped["GROSS_QTY"] - grouped["RETURN_QTY"]
    grouped["MODEL_DEMAND"] = grouped["NET_DEMAND"].clip(lower=0)

    logger.info(f"Net demand aggregation: {grouped['MEDICINE_ID'].nunique():,} unique medicines, "
                f"{grouped['DATE'].nunique():,} unique dates")
    return grouped


# ---------------------------------------------------------------------------
# 3. Granularity decision
# ---------------------------------------------------------------------------
def decide_granularity(daily_series: pd.DataFrame,
                       med_col: str = "MEDICINE_ID",
                       date_col: str = "DATE",
                       demand_col: str = "MODEL_DEMAND") -> dict:
    """
    Returns dict: {medicine_id -> 'daily' | 'weekly'}
    Daily if >= 60% non-zero days, weekly otherwise.
    """
    result = {}
    for med, grp in daily_series.groupby(med_col):
        # Reindex to full daily calendar
        date_range = pd.date_range(grp[date_col].min(), grp[date_col].max(), freq="D")
        grp_reindexed = grp.set_index(date_col)[demand_col].reindex(date_range, fill_value=0)
        pct_nonzero = (grp_reindexed > 0).mean()
        result[med] = "daily" if pct_nonzero >= 0.60 else "weekly"
    
    n_daily = sum(1 for v in result.values() if v == "daily")
    n_weekly = sum(1 for v in result.values() if v == "weekly")
    logger.info(f"Granularity decision: {n_daily} daily, {n_weekly} weekly series")
    return result


# ---------------------------------------------------------------------------
# 4. Reindex to complete calendar
# ---------------------------------------------------------------------------
def reindex_series(daily_df: pd.DataFrame,
                   granularity_map: dict,
                   med_col: str = "MEDICINE_ID",
                   date_col: str = "DATE",
                   demand_col: str = "MODEL_DEMAND") -> pd.DataFrame:
    """
    Reindex each medicine to its own complete calendar grid.
    Returns long-format DataFrame with PERIOD (date), MEDICINE_ID, DEMAND, GRANULARITY.
    """
    pieces = []
    for med, grp in daily_df.groupby(med_col):
        gran = granularity_map.get(med, "weekly")
        freq = "D" if gran == "daily" else "W-MON"
        
        if gran == "weekly":
            # Aggregate daily to weekly first
            grp = grp.set_index(date_col)[[demand_col]].resample("W-MON").sum().reset_index()
            grp.columns = [date_col, demand_col]
        
        full_range = pd.date_range(grp[date_col].min(), grp[date_col].max(), freq=freq)
        grp_ri = grp.set_index(date_col)[demand_col].reindex(full_range, fill_value=0).reset_index()
        grp_ri.columns = ["PERIOD", "DEMAND"]
        grp_ri["MEDICINE_ID"] = med
        grp_ri["GRANULARITY"] = gran
        pieces.append(grp_ri)

    out = pd.concat(pieces, ignore_index=True)
    logger.info(f"Reindexed: {out.shape[0]:,} rows total")
    return out


# ---------------------------------------------------------------------------
# 5. Inclusion filter
# ---------------------------------------------------------------------------
def apply_inclusion_filter(df: pd.DataFrame,
                           min_periods: int = 18,
                           min_nonzero: int = 6) -> tuple:
    """
    Returns (kept_df, drop_log) where drop_log is a dict of dropped medicines and reasons.
    """
    drop_log = {}
    kept_meds = []

    for med, grp in df.groupby("MEDICINE_ID"):
        n_periods = len(grp)
        n_nonzero = (grp["DEMAND"] > 0).sum()
        reason = None

        if n_periods < min_periods:
            reason = f"insufficient_periods ({n_periods} < {min_periods})"
        elif n_nonzero < min_nonzero:
            reason = f"insufficient_nonzero ({n_nonzero} < {min_nonzero})"

        if reason:
            drop_log[med] = reason
        else:
            kept_meds.append(med)

    kept_df = df[df["MEDICINE_ID"].isin(kept_meds)].copy()
    logger.info(f"Inclusion filter: kept {len(kept_meds):,} medicines, dropped {len(drop_log):,}")
    return kept_df, drop_log


# ---------------------------------------------------------------------------
# 6. Outlier detection and handling
# ---------------------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, iqr_mult: float = 3.0, winsorize_q: float = 0.99) -> pd.DataFrame:
    """
    Flag extreme high values per medicine using Q3 + iqr_mult * IQR.
    Strategy: retain genuine surges; winsorise ambiguous at 99th percentile.
    Returns df with added columns: EXTREME_FLAG, DEMAND_CLEAN.
    """
    df = df.copy()
    df["EXTREME_FLAG"] = False
    df["DEMAND_CLEAN"] = df["DEMAND"]
    df["OUTLIER_ACTION"] = "none"

    outlier_stats = {}

    for med, grp in df.groupby("MEDICINE_ID"):
        idx = grp.index
        q1 = grp["DEMAND"].quantile(0.25)
        q3 = grp["DEMAND"].quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + iqr_mult * iqr
        win_val = grp["DEMAND"].quantile(winsorize_q)
        pct_99 = win_val

        extreme_mask = grp["DEMAND"] > threshold
        n_extreme = extreme_mask.sum()

        if n_extreme > 0:
            df.loc[idx[extreme_mask], "EXTREME_FLAG"] = True
            # Winsorise ambiguous cases at 99th percentile (conservative: retain but cap)
            df.loc[idx[extreme_mask], "DEMAND_CLEAN"] = pct_99
            df.loc[idx[extreme_mask], "OUTLIER_ACTION"] = "winsorised"
            
            n_total = len(grp)
            pct_extreme = n_extreme / n_total
            outlier_stats[med] = {
                "n_extreme": int(n_extreme),
                "threshold": float(threshold),
                "pct_99": float(pct_99),
                "pct_extreme": float(pct_extreme)
            }

    logger.info(f"Outlier handling: {sum(s['n_extreme'] for s in outlier_stats.values())} extreme values across "
                f"{len(outlier_stats)} medicines; all winsorised at 99th pct.")
    return df, outlier_stats


# ---------------------------------------------------------------------------
# 7. Per-medicine diagnostics and classification
# ---------------------------------------------------------------------------
def compute_diagnostics(df: pd.DataFrame, outlier_stats: dict, drop_log: dict) -> dict:
    """
    Compute per-medicine stats and classify as Smooth / Intermittent / Lumpy.
    ADI = average demand interval (avg periods between non-zero observations)
    CV2 = (std/mean)^2
    """
    per_med = []
    for med, grp in df.groupby("MEDICINE_ID"):
        d = grp["DEMAND_CLEAN"].values
        n = len(d)
        nonzero = d[d > 0]
        n_nonzero = len(nonzero)
        mean_d = float(np.mean(d))
        median_d = float(np.median(d))
        std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
        cv = std_d / mean_d if mean_d > 0 else 0.0
        cv2 = cv ** 2
        zero_pct = float((d == 0).mean() * 100)

        # ADI: average inter-demand interval
        if n_nonzero > 1:
            nonzero_positions = np.where(d > 0)[0]
            gaps = np.diff(nonzero_positions)
            adi = float(np.mean(gaps))
        elif n_nonzero == 1:
            adi = float(n)
        else:
            adi = float(n)

        # Classification
        if adi < 1.32 and cv2 < 0.49:
            demand_class = "Smooth"
        elif adi >= 1.32 and cv2 >= 0.49:
            demand_class = "Lumpy"
        else:
            demand_class = "Intermittent"

        per_med.append({
            "MEDICINE_ID": med,
            "GRANULARITY": grp["GRANULARITY"].iloc[0],
            "N_PERIODS": n,
            "N_NONZERO": n_nonzero,
            "DATE_MIN": str(grp["PERIOD"].min().date()),
            "DATE_MAX": str(grp["PERIOD"].max().date()),
            "MEAN_DEMAND": round(mean_d, 4),
            "MEDIAN_DEMAND": round(median_d, 4),
            "STD_DEMAND": round(std_d, 4),
            "CV": round(cv, 4),
            "CV2": round(cv2, 4),
            "ZERO_PCT": round(zero_pct, 2),
            "ADI": round(adi, 4),
            "INTERMITTENCY_RATIO": round(n_nonzero / n, 4),
            "DEMAND_CLASS": demand_class,
            "N_EXTREME_OUTLIERS": outlier_stats.get(med, {}).get("n_extreme", 0),
            "PCT_EXTREME": outlier_stats.get(med, {}).get("pct_extreme", 0.0),
        })

    per_med_df = pd.DataFrame(per_med)
    
    class_counts = per_med_df["DEMAND_CLASS"].value_counts().to_dict()
    total_extreme = per_med_df["N_EXTREME_OUTLIERS"].sum()

    diagnostics = {
        "medicines_retained": int(len(per_med_df)),
        "medicines_dropped": int(len(drop_log)),
        "drop_reasons": drop_log,
        "total_periods_all_series": int(df.shape[0]),
        "date_range_overall": {
            "min": str(df["PERIOD"].min().date()),
            "max": str(df["PERIOD"].max().date()),
        },
        "pct_zero_demand_overall": float(round((df["DEMAND_CLEAN"] == 0).mean() * 100, 2)),
        "total_extreme_outliers_flagged": int(total_extreme),
        "demand_class_distribution": class_counts,
        "per_medicine_summary": per_med_df.to_dict(orient="records"),
    }
    return diagnostics, per_med_df


# ---------------------------------------------------------------------------
# 8. Main Phase 1 pipeline
# ---------------------------------------------------------------------------
def run_phase1(raw_df: pd.DataFrame,
               output_dir: str = "output",
               processed_dir: str = "data/processed",
               min_periods: int = 18,
               min_nonzero: int = 6,
               iqr_mult: float = 3.0,
               winsorize_q: float = 0.99) -> tuple:
    """
    Full Phase 1 pipeline. Returns (clean_df, diagnostics_dict).
    """
    logger.info("=== PHASE 1: Dataset Construction ===")

    # Step 1: Filter drugs
    df = filter_drugs(raw_df)

    # Step 2: Parse dates
    df = parse_dates(df)

    # Step 3: Net demand aggregation
    daily_demand = compute_net_demand(df)

    # Step 4: Granularity decision
    gran_map = decide_granularity(daily_demand)

    # Step 5: Reindex to complete calendar
    series_df = reindex_series(daily_demand, gran_map)

    # Step 6: Inclusion filter
    series_df, drop_log = apply_inclusion_filter(series_df, min_periods, min_nonzero)

    # Step 7: Outlier handling
    series_df, outlier_stats = handle_outliers(series_df, iqr_mult, winsorize_q)

    # Step 8: Diagnostics
    diagnostics, per_med_df = compute_diagnostics(series_df, outlier_stats, drop_log)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    diag_path = os.path.join(output_dir, "phase1_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info(f"Diagnostics saved to {diag_path}")

    series_path = os.path.join(processed_dir, "demand_series.csv")
    series_df.to_csv(series_path, index=False)
    logger.info(f"Clean demand series saved to {series_path}")

    per_med_path = os.path.join(output_dir, "tables", "phase1_diagnostics.csv")
    os.makedirs(os.path.dirname(per_med_path), exist_ok=True)
    per_med_df.to_csv(per_med_path, index=False)
    logger.info(f"Per-medicine diagnostics saved to {per_med_path}")

    return series_df, diagnostics, per_med_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all_csvs

    base = os.path.join(os.path.dirname(__file__), "..", "PharmacyTransactionalDataset")
    project_root = os.path.join(os.path.dirname(__file__), "..")
    raw = load_all_csvs(base)
    series_df, diagnostics, per_med_df = run_phase1(
        raw,
        output_dir=os.path.join(project_root, "output"),
        processed_dir=os.path.join(project_root, "data", "processed")
    )
    print(f"\n=== Phase 1 Complete ===")
    print(f"Medicines retained: {diagnostics['medicines_retained']}")
    print(f"Medicines dropped:  {diagnostics['medicines_dropped']}")
    print(f"Overall zero demand %: {diagnostics['pct_zero_demand_overall']}")
    print(f"Demand classes: {diagnostics['demand_class_distribution']}")
