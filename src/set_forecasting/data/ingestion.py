"""Phase 1: streaming SCMD ingestion and monthly cohort construction."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .source import get_project_root, get_scmd_path, validate_scmd_header

logger = logging.getLogger(__name__)
KEY = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events."

def compute_adi_cv2(values):
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if len(positive) == 0:
        return float("inf"), 0.0
    adi = len(values) / len(positive)
    mean = float(positive.mean())
    cv2 = float((positive.std(ddof=1) / mean) ** 2) if len(positive) > 1 and mean else 0.0
    return float(adi), cv2

def classify_demand(adi, cv2):
    if adi < 1.32:
        return "Smooth" if cv2 < 0.49 else "Erratic"
    return "Intermittent" if cv2 < 0.49 else "Lumpy"

def aggregate_scmd(path, output_path, chunksize=200_000):
    """Aggregate raw rows with DuckDB; IDs are read as VARCHAR."""
    source = get_scmd_path(path)
    validate_scmd_header(source)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    escaped = str(source).replace("'", "''")
    query = f"""
      SELECT
        trim(YEAR_MONTH) AS YEAR_MONTH,
        trim(ODS_CODE) AS ODS_CODE,
        trim(VMP_SNOMED_CODE) AS VMP_SNOMED_CODE,
        any_value(trim(VMP_PRODUCT_NAME)) AS VMP_PRODUCT_NAME,
        trim(UNIT_OF_MEASURE_IDENTIFIER) AS UNIT_OF_MEASURE_IDENTIFIER,
        any_value(trim(UNIT_OF_MEASURE_NAME)) AS UNIT_OF_MEASURE_NAME,
        count(*)::BIGINT AS SOURCE_RECORDS,
        count(try_cast(nullif(trim(TOTAL_QUANITY_IN_VMP_UNIT), '') AS DOUBLE))::BIGINT AS VALID_QUANTITY_RECORDS,
        count(*) FILTER (WHERE try_cast(nullif(trim(TOTAL_QUANITY_IN_VMP_UNIT), '') AS DOUBLE) IS NULL)::BIGINT AS INVALID_QUANTITY_RECORDS,
        sum(CASE WHEN try_cast(nullif(trim(TOTAL_QUANITY_IN_VMP_UNIT), '') AS DOUBLE) > 0
                 THEN try_cast(trim(TOTAL_QUANITY_IN_VMP_UNIT) AS DOUBLE) ELSE 0 END) AS POSITIVE_QUANTITY,
        sum(CASE WHEN try_cast(nullif(trim(TOTAL_QUANITY_IN_VMP_UNIT), '') AS DOUBLE) < 0
                 THEN try_cast(trim(TOTAL_QUANITY_IN_VMP_UNIT) AS DOUBLE) ELSE 0 END) AS NEGATIVE_QUANTITY,
        sum(try_cast(nullif(trim(TOTAL_QUANITY_IN_VMP_UNIT), '') AS DOUBLE)) AS NET_QUANTITY,
        count(*) FILTER (WHERE try_cast(nullif(trim(INDICATIVE_COST), '') AS DOUBLE) IS NULL)::BIGINT AS MISSING_OR_INVALID_COST_RECORDS,
        sum(try_cast(nullif(trim(INDICATIVE_COST), '') AS DOUBLE)) AS TOTAL_INDICATIVE_COST,
        count(*) FILTER (WHERE try_cast(nullif(trim(INDICATIVE_COST), '') AS DOUBLE) < 0)::BIGINT AS NEGATIVE_COST_RECORDS
      FROM read_csv_auto('{escaped}', header=true, all_varchar=true, sample_size=-1)
      GROUP BY 1,2,3,5
    """
    try:
        con.execute(f"COPY ({query}) TO '{str(output_path).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()
    return output_path

def build_monthly_series(aggregated_path, series_path, cohort_path, diagnostics_path,
                         min_periods=18, min_nonzero=6, train_frac=.60, val_frac=.20):
    agg = pd.read_parquet(aggregated_path)
    agg["YEAR_MONTH"] = agg["YEAR_MONTH"].astype(str).str.strip()
    parsed = pd.to_datetime(agg["YEAR_MONTH"], format="%Y%m", errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Invalid YEAR_MONTH values: {agg.loc[parsed.isna(), 'YEAR_MONTH'].unique()[:10]}")
    agg["PERIOD"] = parsed
    for col in ["POSITIVE_QUANTITY", "NEGATIVE_QUANTITY", "NET_QUANTITY", "TOTAL_INDICATIVE_COST"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0)

    # Aggregated table is unique on month + organisation + VMP + unit.
    business_key = ["YEAR_MONTH"] + KEY
    duplicate_count = int(agg.duplicated(business_key).sum())
    if duplicate_count:
        raise ValueError(f"Aggregated business key is not unique: {duplicate_count:,} duplicates")

    # Build an explicit monthly calendar per series; absence is not asserted to be zero demand.
    from pyarrow import Table
    from pyarrow.parquet import ParquetWriter
    series_path = Path(series_path)
    series_path.parent.mkdir(parents=True, exist_ok=True)
    batch_frames = []
    writer = None
    batch_size = 250
    for key_values, group in agg.groupby(KEY, sort=False, dropna=False):
        group = group.sort_values("PERIOD").copy()
        months = pd.date_range(group["PERIOD"].min(), group["PERIOD"].max(), freq="MS")
        indexed = group.set_index("PERIOD").reindex(months)
        indexed.index.name = "PERIOD"
        for col, value in zip(KEY, key_values):
            indexed[col] = value
        indexed["OBSERVATION_MISSING"] = indexed["SOURCE_RECORDS"].isna()
        indexed["SOURCE_RECORDS"] = indexed["SOURCE_RECORDS"].fillna(0).astype("int64")
        for col in ["POSITIVE_QUANTITY", "NEGATIVE_QUANTITY", "NET_QUANTITY",
                    "TOTAL_INDICATIVE_COST", "VALID_QUANTITY_RECORDS",
                    "INVALID_QUANTITY_RECORDS", "MISSING_OR_INVALID_COST_RECORDS",
                    "NEGATIVE_COST_RECORDS"]:
            indexed[col] = indexed[col].fillna(0)
        indexed["MODEL_DEMAND"] = indexed["POSITIVE_QUANTITY"]
        indexed["DEMAND_NET"] = indexed["NET_QUANTITY"]
        indexed["VMP_PRODUCT_NAME"] = indexed["VMP_PRODUCT_NAME"].ffill().bfill()
        indexed["UNIT_OF_MEASURE_NAME"] = indexed["UNIT_OF_MEASURE_NAME"].ffill().bfill()
        indexed["YEAR_MONTH"] = indexed.index.strftime("%Y%m")
        batch_frames.append(indexed.reset_index())
        if len(batch_frames) >= batch_size:
            batch = pd.concat(batch_frames, ignore_index=True).sort_values(KEY + ["PERIOD"])
            table = Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = ParquetWriter(series_path, table.schema, compression="zstd")
            writer.write_table(table)
            batch_frames.clear()
    if batch_frames:
        batch = pd.concat(batch_frames, ignore_index=True).sort_values(KEY + ["PERIOD"])
        table = Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = ParquetWriter(series_path, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
        series = pd.read_parquet(series_path)
        series = series.sort_values(KEY + ["PERIOD"]).reset_index(drop=True)
    else:
        series = pd.DataFrame()

    # Cohort statistics and exclusion reasons; count missing periods separately.
    stats = []
    for key_values, group in series.groupby(KEY, sort=False):
        observed = group.loc[~group["OBSERVATION_MISSING"]]
        n_periods = len(group)
        n_nonzero = int((observed["MODEL_DEMAND"] > 0).sum())
        missing_periods = int(group["OBSERVATION_MISSING"].sum())
        reason = []
        if n_periods < min_periods: reason.append("insufficient_periods")
        if n_nonzero < min_nonzero: reason.append("insufficient_nonzero")
        stats.append(dict(zip(KEY, key_values), N_PERIODS=n_periods,
                          N_OBSERVED=int(len(observed)), N_MISSING_PERIODS=missing_periods,
                          N_NONZERO=n_nonzero, COHORT_INCLUDED=not reason,
                          EXCLUSION_REASON=";".join(reason)))
    cohort = pd.DataFrame(stats)
    monthly_summary = agg.groupby("YEAR_MONTH", as_index=False).agg(
        SOURCE_RECORDS=("SOURCE_RECORDS", "sum"),
        POSITIVE_QUANTITY=("POSITIVE_QUANTITY", "sum"),
        NEGATIVE_QUANTITY=("NEGATIVE_QUANTITY", "sum"),
        NET_QUANTITY=("NET_QUANTITY", "sum"),
        TOTAL_INDICATIVE_COST=("TOTAL_INDICATIVE_COST", "sum"),
        INVALID_QUANTITY_RECORDS=("INVALID_QUANTITY_RECORDS", "sum"),
        MISSING_OR_INVALID_COST_RECORDS=("MISSING_OR_INVALID_COST_RECORDS", "sum"),
        NEGATIVE_COST_RECORDS=("NEGATIVE_COST_RECORDS", "sum"),
    )
    monthly_summary["COST_ANOMALY_FLAG"] = (
        monthly_summary["TOTAL_INDICATIVE_COST"] >
        monthly_summary["TOTAL_INDICATIVE_COST"].median() * 2.0
    )
    included = cohort.loc[cohort["COHORT_INCLUDED"], KEY]
    series = series.merge(included.assign(_included=True), on=KEY, how="inner")
    # Keep absent calendar months missing; zero is only valid for an observed source month.
    series["DEMAND"] = series["MODEL_DEMAND"].where(~series["OBSERVATION_MISSING"])

    # Flag high values, do not automatically winsorise genuine demand events.
    q = series.groupby(KEY)["DEMAND"]
    q1 = q.transform(lambda x: x.quantile(.25))
    q3 = q.transform(lambda x: x.quantile(.75))
    series["EXTREME_FLAG"] = series["DEMAND"] > (q3 + 3.0 * (q3 - q1))
    series["DEMAND_CLEAN"] = series["DEMAND"]

    # Correct intermittent-demand diagnostics use positive sizes for CV².
    diagnostics = []
    for key_values, group in series.groupby(KEY, sort=False):
        observed = group.loc[~group["OBSERVATION_MISSING"]]
        values = observed["MODEL_DEMAND"].to_numpy(dtype=float)
        adi, cv2 = compute_adi_cv2(values)
        diagnostics.append(dict(zip(KEY, key_values), N_PERIODS=len(group),
            N_OBSERVED=len(observed), N_NONZERO=int((values > 0).sum()),
            ADI=adi, CV2=cv2, DEMAND_CLASS=classify_demand(adi, cv2),
            ZERO_PCT=float((values == 0).mean() * 100) if len(values) else None,
            EXTREME_COUNT=int(group["EXTREME_FLAG"].sum())))
    demand_diagnostics = pd.DataFrame(diagnostics)
    cohort = cohort.merge(demand_diagnostics[KEY + ["ADI","CV2","DEMAND_CLASS","ZERO_PCT","EXTREME_COUNT"]],
                          on=KEY, how="left")

    # Per-series chronological 60/20/20 split, preserving row-level period assignments.
    split_rows = []
    for key_values, group in series.groupby(KEY, sort=False):
        periods = group["PERIOD"].drop_duplicates().sort_values().tolist()
        n = len(periods)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_end = periods[max(0, n_train - 1)] if n_train else None
        val_end = periods[min(n - 1, n_train + n_val - 1)] if n_train + n_val else None
        split_rows.append(dict(zip(KEY, key_values), TRAIN_END=train_end,
                               VALIDATION_END=val_end, TEST_START=(periods[n_train+n_val] if n_train+n_val < n else pd.NaT)))
    split_map = pd.DataFrame(split_rows)
    series = series.merge(split_map, on=KEY, how="left")
    series["SPLIT"] = np.select(
        [series["PERIOD"] <= series["TRAIN_END"],
         series["PERIOD"] <= series["VALIDATION_END"]],
        ["train", "validation"], default="test")

    Path(series_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cohort_path).parent.mkdir(parents=True, exist_ok=True)
    Path(diagnostics_path).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(cohort_path, index=False)
    monthly_summary.to_csv(Path(cohort_path).parent / "monthly_data_summary.csv", index=False)
    report = {
      "source": "NHSBSA Secondary Care Medicines Indicative Price",
      "granularity": "monthly",
      "date_range": {"start": str(series["PERIOD"].min().date()) if len(series) else None,
                     "end": str(series["PERIOD"].max().date()) if len(series) else None},
      "aggregated_rows": int(len(agg)), "global_duplicate_business_keys": duplicate_count,
      "monthly_cost_anomaly_months": monthly_summary.loc[monthly_summary["COST_ANOMALY_FLAG"], "YEAR_MONTH"].tolist(),
      "series_count_before_filter": int(len(cohort)),
      "series_retained": int(cohort["COHORT_INCLUDED"].sum()),
      "series_excluded": int((~cohort["COHORT_INCLUDED"]).sum()),
      "missing_month_observations_in_retained_cohort": int(series["OBSERVATION_MISSING"].sum()),
      "negative_quantity_total": float(agg["NEGATIVE_QUANTITY"].sum()),
      "positive_quantity_total": float(agg["POSITIVE_QUANTITY"].sum()),
      "invalid_quantity_records": int(agg["INVALID_QUANTITY_RECORDS"].sum()),
      "missing_or_invalid_cost_records": int(agg["MISSING_OR_INVALID_COST_RECORDS"].sum()),
      "negative_cost_records": int(agg["NEGATIVE_COST_RECORDS"].sum()),
      "demand_class_distribution": cohort.loc[cohort["COHORT_INCLUDED"], "DEMAND_CLASS"].value_counts().to_dict(),
      "disclaimer": DISCLAIMER,
      "note": "Missing month rows are flagged, not silently interpreted as observed zero demand. Model demand currently uses positive quantity; net and negative movements remain available separately."
    }
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return series, cohort, report

def run_phase1_from_csv(source=None, output_root="output", processed_root="data/processed",
                        min_periods=18, min_nonzero=6, train_frac=.60, val_frac=.20):
    root = get_project_root()
    source = get_scmd_path(source)
    output_root = Path(output_root); processed_root = Path(processed_root)
    if not output_root.is_absolute(): output_root = root / output_root
    if not processed_root.is_absolute(): processed_root = root / processed_root
    aggregated = processed_root / "scmd_monthly_aggregated.parquet"
    series_path = processed_root / "scmd_monthly_series.parquet"
    cohort_path = output_root / "tables" / "cohort_inclusion_report.csv"
    diagnostics_path = output_root / "phase1_diagnostics.json"
    aggregate_scmd(source, aggregated)
    return build_monthly_series(aggregated, series_path, cohort_path, diagnostics_path,
                                min_periods=min_periods, min_nonzero=min_nonzero,
                                train_frac=train_frac, val_frac=val_frac)

def run_phase1(raw_df=None, **kwargs):
    """Compatibility wrapper: Phase 1 now reads the configured SCMD CSV on disk."""
    return run_phase1_from_csv(**kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    args = parser.parse_args()
    _, cohort, report = run_phase1_from_csv(args.source)
    print(json.dumps(report, indent=2, default=str))
    print("Cohort rows:", len(cohort))
    print(DISCLAIMER)
