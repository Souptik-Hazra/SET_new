"""Deprecated Phase 3 compatibility command.

Forecasting logic lives in src.set_forecasting.forecasting.model.
Prefer the canonical end-to-end runner: python scripts/run_pipeline.py.
"""
from src.scmd_forecasting import run_forecasting


if __name__ == "__main__":
    _, _, report = run_forecasting()
    print(report)
