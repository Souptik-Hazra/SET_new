"""Run SCMD Phase 3 global probabilistic forecasting."""
from src.scmd_forecasting import run_forecasting
if __name__=="__main__":
 metrics,predictions,report=run_forecasting()
 print(report)
