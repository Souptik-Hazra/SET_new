"""
data_loader.py
==============
Loads and merges all pharmacy transaction CSVs into a single raw DataFrame.
Adds source metadata columns (city, zone, pharmacy).
Never modifies raw files.
"""

import os
import glob
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_all_csvs(base_dir: str) -> pd.DataFrame:
    """
    Discover and load all pharmacy CSVs from the given base directory.
    Returns a single merged DataFrame with source metadata columns added.
    """
    pattern = os.path.join(base_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)

    # Separate the global test set
    global_test_files = [f for f in files if "global_test_set" in os.path.basename(f).lower()]
    city_files = [f for f in files if f not in global_test_files]

    logger.info(f"Found {len(city_files)} city/zone/pharmacy CSVs and {len(global_test_files)} global test files.")

    dfs = []

    # Load city pharmacy files
    for filepath in city_files:
        parts = filepath.replace("\\", "/").split("/")
        # Extract city/zone/pharmacy from path structure
        city = next((p for p in parts if p.startswith("City")), "Unknown")
        zone = next((p for p in parts if p.startswith("Zone")), "Unknown")
        pharmacy = os.path.splitext(os.path.basename(filepath))[0]

        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["_city"] = city
            df["_zone"] = zone
            df["_pharmacy"] = pharmacy
            df["_source_file"] = os.path.basename(filepath)
            df["_is_test"] = False
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    # Load global test set
    for filepath in global_test_files:
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["_city"] = "Global"
            df["_zone"] = "Global"
            df["_pharmacy"] = "global_test_set"
            df["_source_file"] = os.path.basename(filepath)
            df["_is_test"] = True
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    if not dfs:
        raise ValueError("No CSV files were loaded.")

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged raw DataFrame: {merged.shape[0]:,} rows, {merged.shape[1]} columns.")
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.path.join(os.path.dirname(__file__), "..", "PharmacyTransactionalDataset")
    df = load_all_csvs(base)
    print(df.head())
    print(df.shape)
