"""Run the full SETReliable SCMD and RADCSR pipeline."""
import argparse
from pathlib import Path
from src.scmd_phase1 import run_phase1_from_csv
from src.set_forecasting.pipeline import run_model_pipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="data/raw/combined_secondary_care_medicines.csv")
    parser.add_argument("--min-periods",type=int,default=18)
    parser.add_argument("--min-nonzero",type=int,default=6)
    parser.add_argument("--bootstrap",type=int,default=2000)
    args=parser.parse_args()
    run_phase1_from_csv(source=args.source,min_periods=args.min_periods,min_nonzero=args.min_nonzero)
    report=run_model_pipeline(n_boot=args.bootstrap)
    print(report)
if __name__=="__main__":main()
