"""Run SCMD Phase 1 using config/config.yaml defaults."""
import argparse,json,yaml
from pathlib import Path
from src.scmd_phase1 import run_phase1_from_csv
if __name__=="__main__":
 p=argparse.ArgumentParser(description="Build SCMD monthly cohort")
 p.add_argument("--source",default=None);p.add_argument("--min-periods",type=int,default=None);p.add_argument("--min-nonzero",type=int,default=None)
 a=p.parse_args();cfg=yaml.safe_load(Path("config/config.yaml").read_text());d=cfg.get("data",{})
 _,cohort,report=run_phase1_from_csv(source=a.source,min_periods=a.min_periods or d.get("min_periods",18),
   min_nonzero=a.min_nonzero or d.get("min_nonzero",6),train_frac=d.get("train_frac",.6),val_frac=d.get("val_frac",.2),load_series=False)
 print(json.dumps(report,indent=2,default=str));print(f"\nCohort rows: {len(cohort):,}")
