"""Backward-compatible import path; use src.set_forecasting.data.source."""
from src.set_forecasting.data.source import get_scmd_path,validate_scmd_header,load_scmd_csv,load_all_csvs
__all__=["get_scmd_path","validate_scmd_header","load_scmd_csv","load_all_csvs"]
