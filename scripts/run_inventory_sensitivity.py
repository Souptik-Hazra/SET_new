"""Run an inventory-assumption sensitivity grid on saved synthetic scenarios."""
import argparse
from pathlib import Path

import pandas as pd
from src.set_forecasting.config import get_config, validate_config

from src.set_forecasting.data.source import get_project_root
from src.set_forecasting.risk.simulation import inventory_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/processed/scenario_streams.parquet")
    parser.add_argument("--output", default="output/tables/inventory_sensitivity.csv")
    parser.add_argument("--service-levels", default="0.90,0.95,0.99")
    parser.add_argument("--initial-covers", default="1,2")
    parser.add_argument("--target-covers", default="2,3")
    parser.add_argument("--lead-times", default=None, help="Comma-separated base lead-time assumptions in months")
    args = parser.parse_args()
    root = get_project_root()
    source = Path(args.scenarios)
    destination = Path(args.output)
    if not source.is_absolute():
        source = root / source
    if not destination.is_absolute():
        destination = root / destination
    scenarios = pd.read_parquet(source)
    config = validate_config(get_config())
    configured_leads = config["inventory"]["lead_time_months"]
    lead_times = tuple(float(x) for x in args.lead_times.split(",")) if args.lead_times else tuple(configured_leads)
    result = inventory_sensitivity(
        scenarios,
        service_levels=tuple(float(x) for x in args.service_levels.split(",")),
        initial_covers=tuple(float(x) for x in args.initial_covers.split(",")),
        target_covers=tuple(float(x) for x in args.target_covers.split(",")),
        lead_times=lead_times,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote {len(result):,} sensitivity rows to {destination}")
    print("All outcomes in this sensitivity analysis are SIMULATED.")


if __name__ == "__main__":
    main()
