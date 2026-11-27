"""Deprecated Phase 4 compatibility command.

Reliability logic lives in src.set_forecasting.reliability.
Prefer the canonical end-to-end runner: python scripts/run_pipeline.py.
"""
from src.scmd_reliability import run_reliability


if __name__ == "__main__":
    _, _, report = run_reliability()
    print(report)
