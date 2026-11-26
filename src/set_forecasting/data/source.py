"""SCMD source loader. The source CSV is streamed and never loaded wholesale."""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SCMD_COLUMNS = [
    "YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_IDENTIFIER", "UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANITY_IN_VMP_UNIT", "INDICATIVE_COST",
]

def get_project_root():
    """Return the repository root for paths supplied relative to the project."""
    return Path(__file__).resolve().parents[3]


def get_scmd_path(path=None):
    root = get_project_root()
    source = Path(path) if path else root / "data" / "raw" / "combined_secondary_care_medicines.csv"
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(
            f"SCMD CSV not found: {source}. Place combined_secondary_care_medicines.csv in data/raw/."
        )
    return source

def validate_scmd_header(path):
    import csv
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    missing = [column for column in SCMD_COLUMNS if column not in header]
    unexpected = [column for column in header if column not in SCMD_COLUMNS]
    if missing or unexpected:
        raise ValueError(f"SCMD schema mismatch. Missing={missing}; unexpected={unexpected}")
    return header

def load_scmd_csv(path=None, chunksize=200_000):
    """Yield string-typed chunks; identifiers and leading zeros remain intact."""
    import pandas as pd
    source = get_scmd_path(path)
    validate_scmd_header(source)
    for chunk in pd.read_csv(source, dtype=str, chunksize=chunksize,
                             keep_default_na=False, encoding="utf-8-sig"):
        yield chunk

# Backward-compatible explicit name; old pharmacy directory discovery is removed.
def load_all_csvs(base_dir=None):
    return load_scmd_csv(base_dir)
