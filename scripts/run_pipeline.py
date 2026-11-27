"""Canonical end-to-end SCMD ingestion and RADCSR model pipeline."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.set_forecasting.data.ingestion import run_phase1_from_csv
from src.set_forecasting.pipeline import run_model_pipeline


def _run_stage(name, function, records, log_path):
    started = time.monotonic()
    record = {
        "stage": name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
    }
    records.append(record)
    _write_stage_log(log_path, records)
    try:
        result = function()
        record["status"] = "PASS"
        return result
    except Exception as exc:
        record["status"] = "FAIL"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write_stage_log(log_path, records)
        print("PIPELINE_STAGE=" + json.dumps(record, sort_keys=True), flush=True)


def _write_stage_log(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/combined_secondary_care_medicines.csv")
    parser.add_argument("--min-periods", type=int, default=18)
    parser.add_argument("--min-nonzero", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"PIPELINE_ENTRYPOINT={run_model_pipeline.__module__}.run_model_pipeline", flush=True)
    stage_log = Path("output") / "pipeline_stage_log.json"
    records = []
    _run_stage(
        "phase1_ingestion_and_cohort",
        lambda: run_phase1_from_csv(
            source=args.source,
            min_periods=args.min_periods,
            min_nonzero=args.min_nonzero,
            load_series=False,
        ),
        records,
        stage_log,
    )
    report = _run_stage(
        "forecast_reliability_scenarios_and_risk",
        lambda: run_model_pipeline(n_boot=args.bootstrap, seed=args.seed),
        records,
        stage_log,
    )
    print(report)


if __name__ == "__main__":
    main()
