"""Canonical end-to-end SCMD ingestion and RADCSR model pipeline."""
import argparse

from src.set_forecasting.data.ingestion import run_phase1_from_csv
from src.set_forecasting.pipeline import run_model_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/combined_secondary_care_medicines.csv")
    parser.add_argument("--min-periods", type=int, default=18)
    parser.add_argument("--min-nonzero", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"PIPELINE_ENTRYPOINT={run_model_pipeline.__module__}.run_model_pipeline")
    run_phase1_from_csv(
        source=args.source,
        min_periods=args.min_periods,
        min_nonzero=args.min_nonzero,
        load_series=False,
    )
    report = run_model_pipeline(n_boot=args.bootstrap, seed=args.seed)
    print(report)


if __name__ == "__main__":
    main()
