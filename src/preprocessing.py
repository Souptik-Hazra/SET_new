"""Backward-compatible preprocessing imports for the SCMD data package."""
from src.set_forecasting.data.ingestion import run_phase1,run_phase1_from_csv,compute_adi_cv2
__all__=["run_phase1","run_phase1_from_csv","compute_adi_cv2"]
if __name__=="__main__":run_phase1_from_csv()
