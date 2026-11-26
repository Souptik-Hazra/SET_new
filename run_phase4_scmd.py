"""Run SCMD Phase 4 prequential calibration."""
from src.scmd_reliability import run_reliability
if __name__=="__main__":
 calibrated,summary,report=run_reliability()
 print(report)
