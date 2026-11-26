"""Run the canonical SCMD/RADCSR model pipeline on processed monthly data.

Phase 1 ingestion is performed by scripts/run_pipeline.py. This entry point is
for rerunning the canonical model pipeline after processed data already exists.
"""
import argparse

from src.set_forecasting.pipeline import run_model_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="data/processed/scmd_monthly_series.parquet")
    parser.add_argument("--output", default="output")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"PIPELINE_ENTRYPOINT={run_model_pipeline.__module__}.run_model_pipeline")
    report = run_model_pipeline(
        series_path=args.series,
        output_dir=args.output,
        processed_dir=args.processed,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    print(report)


if __name__ == "__main__":
    main()
