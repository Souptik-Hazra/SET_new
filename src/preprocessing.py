"""SCMD Phase 1 entry points.

The old pharmacy transaction preprocessing assumptions have been retired.
Use src.scmd_phase1 for monthly organisation-medicine-unit cohort construction.
"""
from .scmd_phase1 import run_phase1, run_phase1_from_csv, compute_adi_cv2

__all__ = ["run_phase1", "run_phase1_from_csv", "compute_adi_cv2"]

if __name__ == "__main__":
    run_phase1_from_csv()
