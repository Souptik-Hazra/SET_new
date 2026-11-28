"""Phase 1: streaming SCMD ingestion and monthly cohort construction."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .source import get_project_root, get_scmd_path, validate_scmd_header

logger = logging.getLogger(__name__)
KEY = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events."
OBSERVED_DEMAND_DISCLAIMER = "Demand observations are observed; inventory outcomes and stockout labels are SIMULATED."

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


def _allocate_split_counts(n_periods, fractions):
    """Allocate all three chronological splits using largest-remainder rounding."""
    raw = np.asarray(fractions, dtype=float) * int(n_periods)
    counts = np.floor(raw).astype(int)
    remainder = int(n_periods) - int(counts.sum())
    order = np.argsort(-(raw-counts), kind="stable")
    counts[order[:remainder]] += 1
    return tuple(int(value) for value in counts)

def _configure_duckdb(con, work_dir):
    """Bound DuckDB memory and direct spill files to local processing storage."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        memory_gb = max(2, min(available_gb * 0.65, 12))
    except ImportError:
        memory_gb = 6
    free_gb = shutil.disk_usage(work_dir).free / (1024 ** 3)
    if free_gb < 3:
        raise RuntimeError(f"DuckDB needs at least 3 GB free for temporary processing; only {free_gb:.2f} GB available at {work_dir}")
    temp_cap_gb = max(1, free_gb * 0.70)
    con.execute(f"SET memory_limit = '{memory_gb:.1f}GB'")
    con.execute(f"SET temp_directory = '{str(work_dir.resolve()).replace(chr(39), chr(39)*2)}'")
    con.execute(f"SET max_temp_directory_size = '{temp_cap_gb:.1f}GB'")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 1000")
    con.execute(f"SET threads = {max(1, min(os.cpu_count() or 2, 4))}")
    logger.info("DuckDB resources: memory_limit=%.1f GB, temp_limit=%.1f GB, temp_dir=%s", memory_gb, temp_cap_gb, work_dir)


def aggregate_scmd(path, output_path, chunksize=200_000):
    """Aggregate raw rows with DuckDB; IDs are read as VARCHAR."""
    source = get_scmd_path(path)
    header = validate_scmd_header(source)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    temp_output.unlink(missing_ok=True)
    work_dir = output_path.parent / ".duckdb_tmp"
    con = duckdb.connect()
    _configure_duckdb(con, work_dir)
    escaped = str(source).replace("'", "''")
    escaped_output = str(temp_output.resolve()).replace("'", "''")
    column_spec = "{" + ", ".join("'" + name.replace("'", "''") + "': 'VARCHAR'" for name in header) + "}"
    started = time.monotonic()
    logger.info("[INGEST 1/2] CSV scan and business-key aggregation started: %s (%.2f GB)", source, source.stat().st_size / 1024**3)
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
        count(*) FILTER (WHERE try_cast(nullif(trim(INDICATIVE_COST), '') AS DOUBLE) < 0)::BIGINT AS NEGATIVE_COST_RECORDS,
        'OBSERVED_DEMAND' AS DATA_TYPE,
        'Demand observations are observed; inventory outcomes and stockout labels are SIMULATED.' AS DISCLAIMER
      FROM read_csv('{escaped}', header=true, columns={column_spec})
      GROUP BY 1,2,3,5
    """
    try:
        con.execute(f"COPY ({query}) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        temp_output.replace(output_path)
        logger.info("[INGEST 1/2] Aggregation complete in %.1f minutes; output=%.2f GB", (time.monotonic()-started)/60, output_path.stat().st_size/1024**3)
    finally:
        con.close()
    return output_path

def build_monthly_series(aggregated_path, series_path, cohort_path, diagnostics_path,
                         min_periods=18, min_nonzero=6, train_frac=.60, val_frac=.20,
                         test_frac=.20, load_series=True):
    """Construct the monthly cohort entirely in DuckDB SQL.

    The aggregated Parquet is never loaded into pandas. DuckDB performs cohort
    statistics, calendar expansion, split allocation, diagnostics and Parquet
    writing; only the compact cohort report is materialised in pandas.
    """
    fractions = (float(train_frac), float(val_frac), float(test_frac))
    if any(not np.isfinite(v) or v <= 0 or v >= 1 for v in fractions):
        raise ValueError("train_frac, val_frac and test_frac must each be between zero and one")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    aggregated_path, series_path = Path(aggregated_path), Path(series_path)
    cohort_path, diagnostics_path = Path(cohort_path), Path(diagnostics_path)
    for path in (series_path, cohort_path, diagnostics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_series = series_path.with_name(series_path.stem + ".tmp" + series_path.suffix)
    temp_series.unlink(missing_ok=True)

    work_dir = series_path.parent / ".duckdb_tmp"
    con = duckdb.connect()
    _configure_duckdb(con, work_dir)
    started = time.monotonic()
    logger.info("[INGEST 2/2] Monthly cohort construction started from %s", aggregated_path)
    agg_file = str(aggregated_path.resolve()).replace("'", "''")
    temp_file = str(temp_series.resolve()).replace("'", "''")
    try:
        con.execute(f"CREATE VIEW agg AS SELECT * FROM read_parquet('{agg_file}')")
        columns = {row[0] for row in con.execute("DESCRIBE agg").fetchall()}
        required = ["YEAR_MONTH"] + KEY + [
            "VMP_PRODUCT_NAME", "UNIT_OF_MEASURE_NAME", "SOURCE_RECORDS",
            "POSITIVE_QUANTITY", "NEGATIVE_QUANTITY", "NET_QUANTITY",
            "TOTAL_INDICATIVE_COST", "VALID_QUANTITY_RECORDS",
            "INVALID_QUANTITY_RECORDS", "MISSING_OR_INVALID_COST_RECORDS",
            "NEGATIVE_COST_RECORDS",
        ]
        missing = sorted(set(required) - columns)
        if missing:
            raise ValueError(f"Aggregated SCMD data is missing columns: {missing}")
        duplicate_count = int(con.execute("""
            SELECT count(*) FROM (
                SELECT YEAR_MONTH, ODS_CODE, VMP_SNOMED_CODE,
                       UNIT_OF_MEASURE_IDENTIFIER
                FROM agg GROUP BY ALL HAVING count(*) > 1
            )
        """).fetchone()[0])
        if duplicate_count:
            raise ValueError(f"Aggregated business key is not unique: {duplicate_count:,} duplicates")

        con.execute("""
            CREATE TEMP TABLE clean AS
            SELECT trim(CAST(YEAR_MONTH AS VARCHAR)) AS YEAR_MONTH,
                   try_strptime(trim(CAST(YEAR_MONTH AS VARCHAR)), '%Y%m')::DATE AS PERIOD,
                   CAST(ODS_CODE AS VARCHAR) AS ODS_CODE,
                   CAST(VMP_SNOMED_CODE AS VARCHAR) AS VMP_SNOMED_CODE,
                   CAST(UNIT_OF_MEASURE_IDENTIFIER AS VARCHAR) AS UNIT_OF_MEASURE_IDENTIFIER,
                   VMP_PRODUCT_NAME, UNIT_OF_MEASURE_NAME,
                   CAST(SOURCE_RECORDS AS BIGINT) AS SOURCE_RECORDS,
                   coalesce(CAST(POSITIVE_QUANTITY AS DOUBLE), 0) AS POSITIVE_QUANTITY,
                   coalesce(CAST(NEGATIVE_QUANTITY AS DOUBLE), 0) AS NEGATIVE_QUANTITY,
                   coalesce(CAST(NET_QUANTITY AS DOUBLE), 0) AS NET_QUANTITY,
                   coalesce(CAST(TOTAL_INDICATIVE_COST AS DOUBLE), 0) AS TOTAL_INDICATIVE_COST,
                   coalesce(CAST(VALID_QUANTITY_RECORDS AS BIGINT), 0) AS VALID_QUANTITY_RECORDS,
                   coalesce(CAST(INVALID_QUANTITY_RECORDS AS BIGINT), 0) AS INVALID_QUANTITY_RECORDS,
                   coalesce(CAST(MISSING_OR_INVALID_COST_RECORDS AS BIGINT), 0) AS MISSING_OR_INVALID_COST_RECORDS,
                   coalesce(CAST(NEGATIVE_COST_RECORDS AS BIGINT), 0) AS NEGATIVE_COST_RECORDS
            FROM agg
        """)
        invalid_dates = int(con.execute("SELECT count(*) FROM clean WHERE PERIOD IS NULL").fetchone()[0])
        if invalid_dates:
            raise ValueError(f"Aggregated SCMD table contains {invalid_dates:,} invalid YEAR_MONTH values")
        null_source = int(con.execute("SELECT count(*) FROM clean WHERE SOURCE_RECORDS IS NULL").fetchone()[0])
        if null_source:
            raise ValueError("Aggregated SCMD table contains null SOURCE_RECORDS")

        con.execute(f"""
            CREATE TEMP TABLE cohort_stats AS
            WITH stats AS (
                SELECT ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER,
                       min(PERIOD) AS FIRST_PERIOD, max(PERIOD) AS LAST_PERIOD,
                       date_diff('month', min(PERIOD), max(PERIOD)) + 1 AS N_PERIODS,
                       count(*) AS N_OBSERVED,
                       sum(CASE WHEN POSITIVE_QUANTITY > 0 THEN 1 ELSE 0 END)::BIGINT AS N_NONZERO,
                       sum(CASE WHEN POSITIVE_QUANTITY = 0 THEN 1 ELSE 0 END)::BIGINT AS N_ZERO,
                       quantile_cont(POSITIVE_QUANTITY, .25) AS Q1,
                       quantile_cont(POSITIVE_QUANTITY, .75) AS Q3,
                       avg(POSITIVE_QUANTITY) FILTER (WHERE POSITIVE_QUANTITY > 0) AS POS_MEAN,
                       stddev_samp(POSITIVE_QUANTITY) FILTER (WHERE POSITIVE_QUANTITY > 0) AS POS_STD,
                       count(*) FILTER (WHERE POSITIVE_QUANTITY > 0) AS POS_COUNT
                FROM clean GROUP BY ALL
            ), split_raw AS (
                SELECT *, N_PERIODS * {fractions[0]:.17g} AS RAW_TRAIN,
                          N_PERIODS * {fractions[1]:.17g} AS RAW_VAL,
                          N_PERIODS * {fractions[2]:.17g} AS RAW_TEST
                FROM stats
            ), split_floor AS (
                SELECT *, floor(RAW_TRAIN)::BIGINT AS BASE_TRAIN,
                          floor(RAW_VAL)::BIGINT AS BASE_VAL,
                          floor(RAW_TEST)::BIGINT AS BASE_TEST,
                          N_PERIODS - floor(RAW_TRAIN) - floor(RAW_VAL) - floor(RAW_TEST) AS REMAINDER
                FROM split_raw
            ), split_rank AS (
                SELECT *,
                    1 + CASE WHEN RAW_VAL-BASE_VAL > RAW_TRAIN-BASE_TRAIN THEN 1 ELSE 0 END
                      + CASE WHEN RAW_TEST-BASE_TEST > RAW_TRAIN-BASE_TRAIN THEN 1 ELSE 0 END AS RANK_TRAIN,
                    1 + CASE WHEN RAW_TRAIN-BASE_TRAIN >= RAW_VAL-BASE_VAL THEN 1 ELSE 0 END
                      + CASE WHEN RAW_TEST-BASE_TEST > RAW_VAL-BASE_VAL THEN 1 ELSE 0 END AS RANK_VAL,
                    1 + CASE WHEN RAW_TRAIN-BASE_TRAIN >= RAW_TEST-BASE_TEST THEN 1 ELSE 0 END
                      + CASE WHEN RAW_VAL-BASE_VAL >= RAW_TEST-BASE_TEST THEN 1 ELSE 0 END AS RANK_TEST
                FROM split_floor
            )
            SELECT *,
                BASE_TRAIN + CASE WHEN RANK_TRAIN <= REMAINDER THEN 1 ELSE 0 END AS N_TRAIN,
                BASE_VAL + CASE WHEN RANK_VAL <= REMAINDER THEN 1 ELSE 0 END AS N_VAL,
                BASE_TEST + CASE WHEN RANK_TEST <= REMAINDER THEN 1 ELSE 0 END AS N_TEST,
                CASE WHEN N_PERIODS < {int(min_periods)} THEN 'insufficient_periods'
                     WHEN N_NONZERO < {int(min_nonzero)} THEN 'insufficient_nonzero'
                     WHEN least(
                         BASE_TRAIN + CASE WHEN RANK_TRAIN <= REMAINDER THEN 1 ELSE 0 END,
                         BASE_VAL + CASE WHEN RANK_VAL <= REMAINDER THEN 1 ELSE 0 END,
                         BASE_TEST + CASE WHEN RANK_TEST <= REMAINDER THEN 1 ELSE 0 END
                     ) < 1 THEN 'insufficient_rows_for_three_way_split'
                     ELSE '' END AS EXCLUSION_REASON,
                CASE WHEN N_PERIODS > 0 AND N_NONZERO > 0
                     THEN N_OBSERVED::DOUBLE / N_NONZERO ELSE 'Infinity'::DOUBLE END AS ADI,
                CASE WHEN POS_COUNT > 1 AND POS_MEAN <> 0
                     THEN pow(POS_STD / POS_MEAN, 2) ELSE 0 END AS CV2,
                CASE WHEN (N_OBSERVED::DOUBLE / N_NONZERO) < 1.32 AND
                          CASE WHEN POS_COUNT > 1 AND POS_MEAN <> 0 THEN pow(POS_STD/POS_MEAN,2) ELSE 0 END < .49 THEN 'Smooth'
                     WHEN (N_OBSERVED::DOUBLE / N_NONZERO) < 1.32 THEN 'Erratic'
                     WHEN CASE WHEN POS_COUNT > 1 AND POS_MEAN <> 0 THEN pow(POS_STD/POS_MEAN,2) ELSE 0 END < .49 THEN 'Intermittent'
                     ELSE 'Lumpy' END AS DEMAND_CLASS
            FROM split_rank
        """)

        # Exact monthly calendar expansion and carry-forward labels are executed
        # in DuckDB. Missing months remain missing observations, never zero demand.
        con.execute(f"""
            CREATE TEMP TABLE monthly AS
            WITH calendar AS (
                SELECT s.ODS_CODE, s.VMP_SNOMED_CODE, s.UNIT_OF_MEASURE_IDENTIFIER,
                       gs.month_date::DATE AS PERIOD
                FROM cohort_stats s,
                LATERAL generate_series(s.FIRST_PERIOD, s.LAST_PERIOD, INTERVAL 1 MONTH) gs(month_date)
                WHERE s.EXCLUSION_REASON = ''
            ), joined AS (
                SELECT c.*, a.YEAR_MONTH, a.VMP_PRODUCT_NAME, a.UNIT_OF_MEASURE_NAME,
                       a.SOURCE_RECORDS, a.POSITIVE_QUANTITY, a.NEGATIVE_QUANTITY,
                       a.NET_QUANTITY, a.TOTAL_INDICATIVE_COST, a.VALID_QUANTITY_RECORDS,
                       a.INVALID_QUANTITY_RECORDS, a.MISSING_OR_INVALID_COST_RECORDS,
                       a.NEGATIVE_COST_RECORDS
                FROM calendar c LEFT JOIN clean a USING
                    (ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER, PERIOD)
            ), labelled AS (
                SELECT *,
                    SOURCE_RECORDS IS NULL AS OBSERVATION_MISSING,
                    coalesce(SOURCE_RECORDS, 0)::BIGINT AS SOURCE_RECORDS_FILLED,
                    coalesce(POSITIVE_QUANTITY, 0) AS POSITIVE_FILLED,
                    coalesce(NEGATIVE_QUANTITY, 0) AS NEGATIVE_FILLED,
                    coalesce(NET_QUANTITY, 0) AS NET_FILLED,
                    coalesce(TOTAL_INDICATIVE_COST, 0) AS COST_FILLED,
                    coalesce(VALID_QUANTITY_RECORDS, 0) AS VALID_FILLED,
                    coalesce(INVALID_QUANTITY_RECORDS, 0) AS INVALID_FILLED,
                    coalesce(MISSING_OR_INVALID_COST_RECORDS, 0) AS MISSING_COST_FILLED,
                    coalesce(NEGATIVE_COST_RECORDS, 0) AS NEGATIVE_COST_FILLED,
                    last_value(VMP_PRODUCT_NAME IGNORE NULLS) OVER w AS PRODUCT_FF,
                    last_value(UNIT_OF_MEASURE_NAME IGNORE NULLS) OVER w AS UNIT_FF,
                    first_value(VMP_PRODUCT_NAME IGNORE NULLS) OVER (PARTITION BY ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER ORDER BY PERIOD ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS PRODUCT_BF,
                    first_value(UNIT_OF_MEASURE_NAME IGNORE NULLS) OVER (PARTITION BY ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER ORDER BY PERIOD ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS UNIT_BF,
                    row_number() OVER w AS PERIOD_INDEX
                FROM joined
                WINDOW w AS (PARTITION BY ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER ORDER BY PERIOD
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            ), bounds AS (
                SELECT *, coalesce(PRODUCT_FF, PRODUCT_BF) AS PRODUCT_NAME_FILLED,
                          coalesce(UNIT_FF, UNIT_BF) AS UNIT_NAME_FILLED
                FROM labelled
            ), demand AS (
                SELECT *, CASE WHEN OBSERVATION_MISSING THEN NULL ELSE POSITIVE_FILLED END AS MODEL_DEMAND,
                          CASE WHEN OBSERVATION_MISSING THEN NULL ELSE NET_FILLED END AS DEMAND_NET,
                          CASE WHEN OBSERVATION_MISSING THEN NULL ELSE POSITIVE_FILLED END AS DEMAND
                FROM bounds
            ), split AS (
                SELECT m.*, s.N_PERIODS, s.N_OBSERVED, s.N_NONZERO, s.ADI, s.CV2, s.DEMAND_CLASS,
                       s.N_TRAIN, s.N_VAL, s.N_TEST, s.FIRST_PERIOD,
                       (s.Q3 + 3.0*(s.Q3-s.Q1)) AS COHORT_UPPER_FENCE,
                       row_number() OVER (PARTITION BY m.ODS_CODE,m.VMP_SNOMED_CODE,m.UNIT_OF_MEASURE_IDENTIFIER ORDER BY m.PERIOD) AS SPLIT_INDEX
                FROM demand m JOIN cohort_stats s USING (ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER)
            )
            SELECT strftime(PERIOD, '%Y%m') AS YEAR_MONTH, ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER,
                   coalesce(PRODUCT_NAME_FILLED, '') AS VMP_PRODUCT_NAME,
                   coalesce(UNIT_NAME_FILLED, '') AS UNIT_OF_MEASURE_NAME,
                   PERIOD, SOURCE_RECORDS_FILLED AS SOURCE_RECORDS,
                   POSITIVE_FILLED AS POSITIVE_QUANTITY, NEGATIVE_FILLED AS NEGATIVE_QUANTITY,
                   NET_FILLED AS NET_QUANTITY, COST_FILLED AS TOTAL_INDICATIVE_COST,
                   VALID_FILLED AS VALID_QUANTITY_RECORDS, INVALID_FILLED AS INVALID_QUANTITY_RECORDS,
                   MISSING_COST_FILLED AS MISSING_OR_INVALID_COST_RECORDS,
                   NEGATIVE_COST_FILLED AS NEGATIVE_COST_RECORDS,
                   OBSERVATION_MISSING,
                   MODEL_DEMAND, DEMAND_NET, DEMAND,
                   DEMAND AS DEMAND_CLEAN,
                   coalesce(MODEL_DEMAND > COHORT_UPPER_FENCE, false) AS EXTREME_FLAG,
                   CASE WHEN SPLIT_INDEX <= N_TRAIN THEN 'train'
                        WHEN SPLIT_INDEX <= N_TRAIN + N_VAL THEN 'validation'
                        ELSE 'test' END AS SPLIT,
                   (FIRST_PERIOD + (N_TRAIN-1) * INTERVAL 1 MONTH)::DATE AS TRAIN_END,
                   (FIRST_PERIOD + (N_TRAIN+N_VAL-1) * INTERVAL 1 MONTH)::DATE AS VALIDATION_END,
                   (FIRST_PERIOD + (N_TRAIN+N_VAL) * INTERVAL 1 MONTH)::DATE AS TEST_START,
                   'OBSERVED_DEMAND' AS DATA_TYPE,
                   '{OBSERVED_DEMAND_DISCLAIMER}' AS DISCLAIMER
            FROM split
        """)

        con.execute(f"COPY (SELECT * FROM monthly) TO '{temp_file}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        temp_series.replace(series_path)
        logger.info("[INGEST 2/2] Monthly Parquet written in %.1f minutes (%.2f GB)", (time.monotonic()-started)/60, series_path.stat().st_size/1024**3)

        cohort = con.execute("""
            SELECT s.ODS_CODE, s.VMP_SNOMED_CODE, s.UNIT_OF_MEASURE_IDENTIFIER,
                   s.N_PERIODS, s.N_OBSERVED,
                   s.N_PERIODS-s.N_OBSERVED AS N_MISSING_PERIODS,
                   s.N_NONZERO, (s.EXCLUSION_REASON = '') AS COHORT_INCLUDED,
                   s.EXCLUSION_REASON, s.ADI, s.CV2, s.DEMAND_CLASS,
                   CASE WHEN s.N_OBSERVED > 0 THEN s.N_ZERO::DOUBLE/s.N_OBSERVED*100 ELSE NULL END AS ZERO_PCT,
                   coalesce(x.EXTREME_COUNT,0) AS EXTREME_COUNT,
                   'OBSERVED_DEMAND' AS DATA_TYPE, ? AS DISCLAIMER
            FROM cohort_stats s LEFT JOIN (
                SELECT c.ODS_CODE,c.VMP_SNOMED_CODE,c.UNIT_OF_MEASURE_IDENTIFIER,
                       count(*) FILTER (WHERE a.POSITIVE_QUANTITY > c.Q3 + 3.0*(c.Q3-c.Q1)) AS EXTREME_COUNT
                FROM cohort_stats c JOIN clean a USING
                    (ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER)
                GROUP BY ALL
            ) x USING (ODS_CODE,VMP_SNOMED_CODE,UNIT_OF_MEASURE_IDENTIFIER)
        """, [OBSERVED_DEMAND_DISCLAIMER]).df()

        monthly_summary = con.execute("""
            SELECT YEAR_MONTH,
                   sum(SOURCE_RECORDS)::BIGINT AS SOURCE_RECORDS,
                   sum(POSITIVE_QUANTITY) AS POSITIVE_QUANTITY,
                   sum(NEGATIVE_QUANTITY) AS NEGATIVE_QUANTITY,
                   sum(NET_QUANTITY) AS NET_QUANTITY,
                   sum(TOTAL_INDICATIVE_COST) AS TOTAL_INDICATIVE_COST,
                   sum(INVALID_QUANTITY_RECORDS)::BIGINT AS INVALID_QUANTITY_RECORDS,
                   sum(MISSING_OR_INVALID_COST_RECORDS)::BIGINT AS MISSING_OR_INVALID_COST_RECORDS,
                   sum(NEGATIVE_COST_RECORDS)::BIGINT AS NEGATIVE_COST_RECORDS
            FROM clean GROUP BY YEAR_MONTH ORDER BY YEAR_MONTH
        """).df()
        if cohort.empty:
            raise ValueError("Aggregated SCMD dataset is empty")
        if int(cohort.COHORT_INCLUDED.sum()) == 0:
            raise ValueError(
                "No SCMD series met cohort and chronological split requirements; "
                "review min_periods, min_nonzero and split fractions."
            )

        monthly_summary["COST_ANOMALY_FLAG"] = (
            monthly_summary["TOTAL_INDICATIVE_COST"] > monthly_summary["TOTAL_INDICATIVE_COST"].median() * 2.0
        )
        monthly_summary["DATA_TYPE"] = "OBSERVED_DEMAND"
        monthly_summary["DISCLAIMER"] = OBSERVED_DEMAND_DISCLAIMER
        cohort.to_csv(cohort_path, index=False)
        monthly_summary.to_csv(cohort_path.parent / "monthly_data_summary.csv", index=False)

        totals = con.execute("""
            SELECT min(PERIOD), max(PERIOD), count(*),
                   sum(NEGATIVE_QUANTITY), sum(POSITIVE_QUANTITY),
                   sum(INVALID_QUANTITY_RECORDS), sum(MISSING_OR_INVALID_COST_RECORDS),
                   sum(NEGATIVE_COST_RECORDS)
            FROM clean
        """).fetchone()
        retained = int(cohort.COHORT_INCLUDED.sum())
        report = {
            "source": "NHSBSA Secondary Care Medicines Indicative Price",
            "granularity": "monthly",
            "date_range": {"start": str(totals[0]), "end": str(totals[1])},
            "aggregated_rows": int(totals[2]),
            "global_duplicate_business_keys": duplicate_count,
            "monthly_cost_anomaly_months": monthly_summary.loc[monthly_summary.COST_ANOMALY_FLAG, "YEAR_MONTH"].tolist(),
            "series_count_before_filter": int(len(cohort)),
            "series_retained": retained,
            "series_excluded": int(len(cohort)-retained),
            "missing_month_observations_in_retained_cohort": int(cohort.loc[cohort.COHORT_INCLUDED,"N_MISSING_PERIODS"].sum()),
            "negative_quantity_total": float(totals[3] or 0),
            "positive_quantity_total": float(totals[4] or 0),
            "invalid_quantity_records": int(totals[5] or 0),
            "missing_or_invalid_cost_records": int(totals[6] or 0),
            "negative_cost_records": int(totals[7] or 0),
            "demand_class_distribution": cohort.loc[cohort.COHORT_INCLUDED,"DEMAND_CLASS"].value_counts().to_dict(),
            "memory_strategy": "DuckDB SQL performs cohort statistics, monthly calendar expansion, chronological split assignment and direct Parquet writing; DuckDB memory is bounded and excess work spills to a configured local temp directory; pandas materialises only compact reports.",
            "duckdb_memory_limit": con.execute("SELECT current_setting('memory_limit')").fetchone()[0],
            "duckdb_temp_directory": con.execute("SELECT current_setting('temp_directory')").fetchone()[0],
            "monthly_parquet_bytes": series_path.stat().st_size,
            "DATA_TYPE": "OBSERVED_DEMAND",
            "disclaimer": OBSERVED_DEMAND_DISCLAIMER,
            "note": "Missing month rows are flagged, not interpreted as observed zero demand. MODEL_DEMAND is NULL for unobserved months; negative and net movements remain separately available."
        }
        with open(diagnostics_path,"w",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2,default=str)
        returned = con.execute(f"SELECT * FROM read_parquet('{str(series_path.resolve()).replace(chr(39), chr(39)*2)}')").df() if load_series else None
        logger.info("[INGEST 2/2] Cohort construction and diagnostics complete in %.1f minutes", (time.monotonic()-started)/60)
        return returned, cohort, report
    finally:
        con.close()

def run_phase1_from_csv(source=None, output_root="output", processed_root="data/processed",
                        min_periods=18, min_nonzero=6, train_frac=.60, val_frac=.20,
                        test_frac=.20, load_series=True):
    root = get_project_root()
    source = get_scmd_path(source)
    output_root = Path(output_root); processed_root = Path(processed_root)
    if not output_root.is_absolute(): output_root = root / output_root
    if not processed_root.is_absolute(): processed_root = root / processed_root
    aggregated = processed_root / "scmd_monthly_aggregated.parquet"
    series_path = processed_root / "scmd_monthly_series.parquet"
    cohort_path = output_root / "tables" / "cohort_inclusion_report.csv"
    diagnostics_path = output_root / "phase1_diagnostics.json"
    logger.info("Phase 1/2: DuckDB is scanning and aggregating the source CSV.")
    aggregate_scmd(source, aggregated)
    logger.info("Phase 2/2: constructing the eligible monthly cohort.")
    return build_monthly_series(aggregated, series_path, cohort_path, diagnostics_path,
                                min_periods=min_periods, min_nonzero=min_nonzero,
                                train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                                load_series=load_series)

def run_phase1(raw_df=None, **kwargs):
    """Compatibility wrapper: Phase 1 now reads the configured SCMD CSV on disk."""
    return run_phase1_from_csv(**kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    args = parser.parse_args()
    _, cohort, report = run_phase1_from_csv(args.source, load_series=False)
    print(json.dumps(report, indent=2, default=str))
    print("Cohort rows:", len(cohort))
    print(OBSERVED_DEMAND_DISCLAIMER)
    print(DISCLAIMER)
