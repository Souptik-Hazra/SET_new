"""Unified command-line entry point for the SETReliable SCMD pipeline.

Usage:
    python scripts/run_pipeline.py [run] --source <CSV>
    python scripts/run_pipeline.py phase1 [options]
    python scripts/run_pipeline.py phase2 [options]
    python scripts/run_pipeline.py phase3 [options]
    python scripts/run_pipeline.py phase4 [options]
    python scripts/run_pipeline.py model [options]

The default command, `run`, performs Phase 1 ingestion followed by the
canonical SCMD/RADCSR model pipeline. Standalone phase commands are retained
for controlled debugging and reproducible stage-level execution.

The former pharmacy-schema full-cohort runner is intentionally unavailable:
that dataset contract is incompatible with the current SCMD pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from src.set_forecasting.config import get_config, validate_config

from src.set_forecasting.data.ingestion import run_phase1_from_csv
from src.set_forecasting.features.engineering import create_features, leakage_check
from src.set_forecasting.pipeline import run_model_pipeline


DEFAULT_SOURCE = "data/raw/combined_secondary_care_medicines.csv"
DEFAULT_SERIES = "data/processed/scmd_monthly_series.parquet"
DEFAULT_PREDICTIONS = "data/processed/scmd_test_predictions.parquet"


def _load_config():
    return validate_config(get_config())

def _phase1(args, config):
    data = config.get("data", {})
    _, cohort, report = run_phase1_from_csv(
        source=args.source,
        output_root=args.output,
        processed_root=args.processed,
        min_periods=args.min_periods if args.min_periods is not None else data.get("min_periods", 18),
        min_nonzero=args.min_nonzero if args.min_nonzero is not None else data.get("min_nonzero", 6),
        train_frac=data.get("train_frac", 0.60),
        val_frac=data.get("validation_frac", data.get("val_frac", 0.20)),
        test_frac=data.get("test_frac", 0.20),
        load_series=False,
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"\nCohort rows: {len(cohort):,}")


def _phase2(args):
    source = Path(args.series)
    target = Path(args.features)
    if not source.is_file():
        raise FileNotFoundError(f"Run Phase 1 ingestion first: {source}")
    features = create_features(pd.read_parquet(source))
    if not leakage_check(features):
        raise RuntimeError("Feature leakage check failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    print(f"Phase 2 complete: {len(features):,} rows, {len(features.columns)} columns")
    print(f"Features saved to: {target}")


def _phase3(args):
    from src.scmd_forecasting import run_forecasting

    _, _, report = run_forecasting(
        series_path=args.series,
        output_dir=args.output,
        processed_dir=args.processed,
        random_state=args.seed if args.seed is not None else 42,
        winsor_quantile=args.winsor_quantile,
    )
    print(json.dumps(report, indent=2, default=str))


def _phase4(args):
    from src.scmd_reliability import run_reliability

    _, _, report = run_reliability(
        prediction_path=args.predictions,
        output_dir=args.output,
        processed_dir=args.processed,
        target_coverage=args.target_coverage,
        window=args.window,
    )
    print(json.dumps(report, indent=2, default=str))


def _model(args):
    print(f"PIPELINE_ENTRYPOINT={run_model_pipeline.__module__}.run_model_pipeline")
    report = run_model_pipeline(
        series_path=args.series,
        output_dir=args.output,
        processed_dir=args.processed,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    print(report)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the unified SETReliable SCMD ingestion, forecasting and RADCSR pipeline."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "phase1", "phase2", "phase3", "phase4", "model", "legacy-full-cohort"),
        default="run",
        help="run all stages (default), run one stage, or explicitly reject the retired legacy pipeline",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Raw SCMD CSV path")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Processed monthly-series Parquet path")
    parser.add_argument("--features", default="data/processed/scmd_features.parquet", help="Phase 2 feature output path")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Phase 4 input prediction Parquet path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--processed", default="data/processed", help="Processed-data directory")
    parser.add_argument("--min-periods", type=int, default=None)
    parser.add_argument("--min-nonzero", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config = _load_config()

    if args.command == "legacy-full-cohort":
        raise SystemExit(
            "The legacy full-cohort pipeline is disabled: it expects the historical "
            "MEDICINE_ID/DEMAND_CLEAN pharmacy schema, not the current SCMD schema. "
            "Use 'python scripts/run_pipeline.py' for the supported SCMD workflow."
        )
    if args.command == "phase1":
        _phase1(args, config)
        return
    if args.command == "phase2":
        _phase2(args)
        return
    if args.command == "phase3":
        _phase3(args)
        return
    if args.command == "phase4":
        _phase4(args)
        return
    if args.command == "model":
        _model(args)
        return

    # Default "run": ingestion followed by the one canonical model orchestrator.
    _phase1(args, config)
    _model(args)


if __name__ == "__main__":
    main()
