"""Backward-compatible imports; use src.set_forecasting.data.ingestion."""
from src.set_forecasting.data.ingestion import (
    KEY, DISCLAIMER, compute_adi_cv2, classify_demand, aggregate_scmd,
    build_monthly_series, run_phase1_from_csv, run_phase1,
)
__all__=["KEY","DISCLAIMER","compute_adi_cv2","classify_demand","aggregate_scmd",
         "build_monthly_series","run_phase1_from_csv","run_phase1"]
if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument("--source",default=None);args=parser.parse_args()
    _,cohort,report=run_phase1_from_csv(args.source, load_series=False)
    print(json.dumps(report,indent=2,default=str));print("Cohort rows:",len(cohort));print(DISCLAIMER)
