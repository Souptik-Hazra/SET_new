"""Phase B mathematical validation. Run with python -m src.set_forecasting.validation.mathematical."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from ..risk.radcsr import radcsr_probability, infer_lead_time_distribution, block_bootstrap_lead_time_samples
from ..reliability.index import build_reliability_index, KEYS

DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events."
FORMULAS = {
 "RADCSR_DISTRIBUTION_PROBABILITY": "1-Phi((A-mu_adj*L_eff/L)/(sigma_adj*sqrt(L_eff/L))); mu_adj=mu+I(A<mu)*mu*kappa_mu*R_rel; sigma_adj=sigma*(1+kappa_sigma*R_rel); L_eff=L*(1+kappa_l*R_rel), capped at horizon_cap*L.",
 "RADCSR_PROBABILITY": "p0+(1-p0)*(1-exp(-S)); p0=1-Phi((A-mu)/sigma); S=log(1+kappa_sigma*R_rel)+log(1+kappa_mu*R_rel*I(A<mu))+log(max(L_eff/L,1)). Risk-uplift score, not a distribution tail."
}
ROOT = Path(__file__).resolve().parents[3]

def load(rel):
 p=ROOT/rel
 if not p.is_file(): raise FileNotFoundError(f"Required Phase B input is missing: {p}")
 return pd.read_parquet(p)

def monotonicity():
 r=np.round(np.arange(0,1.0001,.05),2); q10,q50,q90,L=80.,100.,130.,2.
 mu,sigma=infer_lead_time_distribution(q10,q50,q90,L)
 cases=[]
 for regime,A in (("A > mu_L",mu+sigma),("A < mu_L",max(0.,mu-sigma))):
  z=radcsr_probability(np.full(len(r),q10),np.full(len(r),q50),np.full(len(r),q90),np.full(len(r),A),np.full(len(r),L),r)
  for col in ("RADCSR_DISTRIBUTION_PROBABILITY","RADCSR_PROBABILITY"):
   seq=z[col].to_numpy(float); inc=np.diff(seq)
   cases.append({"quantity":col,"regime":regime,"availability":float(A),"pass":bool(np.all(inc>=-1e-10)),"sequence":seq.tolist(),"increments":inc.tolist(),"minimum_increment":float(inc.min())})
 return cases

def demand_class(y):
 y=np.asarray(y,float); y=y[np.isfinite(y)&(y>=0)]; pos=y[y>0]
 if not len(pos): return "intermittent",float("inf"),float("inf")
 adi=len(y)/len(pos); cv2=float((np.std(pos)/max(np.mean(pos),1e-12))**2)
 label=("smooth" if cv2<.49 else "erratic") if adi<1.32 else ("intermittent" if cv2<.49 else "lumpy")
 return label,float(adi),cv2

def stats(a,lo,md,hi):
 if not a:return {"n":0,"coverage_80":None,"median_abs_error":None,"mean_interval_width":None,"adequate":False}
 a,lo,md,hi=map(np.asarray,(a,lo,md,hi));cov=float(np.mean((a>=lo)&(a<=hi)))
 return {"n":len(a),"coverage_80":cov,"median_abs_error":float(np.mean(abs(a-md))),"mean_interval_width":float(np.mean(hi-lo)),"adequate":bool(abs(cov-.8)<=.1)}

def lead_time_validation():
 f=load("data/processed/forecasts.parquet"); s=load("data/processed/scmd_monthly_series.parquet")
 classes={}
 for key,g in s.groupby(KEYS,sort=False,dropna=False):
  classes[tuple(key)]=demand_class(g.loc[g.SPLIT=="train","MODEL_DEMAND"])[0]
 val=f[f.SPLIT=="validation"].copy(); val.PERIOD=pd.to_datetime(val.PERIOD)
 agg={c:{m:[[],[],[],[]] for m in ("normal","block_bootstrap")} for c in ("smooth","erratic","intermittent","lumpy")}
 counts={c:0 for c in agg}
 for key,g in val.groupby(KEYS,sort=False,dropna=False):
  g=g.sort_values("PERIOD").reset_index(drop=True); c=classes.get(tuple(key),"intermittent")
  y=g.MODEL_DEMAND.to_numpy(float); q10=g.Q10.to_numpy(float); q50=g.Q50.to_numpy(float); q90=g.Q90.to_numpy(float); err=y-q50
  for t in range(6,len(g)):
   for L in (1,2,3):
    if t+L>len(g): continue
    target=float(y[t:t+L].sum()); mu,sigma=infer_lead_time_distribution(q10[t],q50[t],q90[t],L)
    n=agg[c]["normal"]; n[0].append(target); n[1].append(max(0.,float(mu-1.2815515655*sigma))); n[2].append(float(mu)); n[3].append(float(mu+1.2815515655*sigma))
    samples=block_bootstrap_lead_time_samples(q50[t:t+L],err[:t],L,n_boot=300,seed=19+t+L)
    b=agg[c]["block_bootstrap"]; b[0].append(target); b[1].append(float(np.quantile(samples,.1))); b[2].append(float(np.quantile(samples,.5))); b[3].append(float(np.quantile(samples,.9)))
    counts[c]+=1
 out={}
 for c in agg:
  n=stats(*agg[c]["normal"]); b=stats(*agg[c]["block_bootstrap"])
  out[c]={"normal_approximation_baseline":n,"dependence_aware_block_bootstrap":b,"normal_adequate":n["adequate"],"block_bootstrap_adequate":b["adequate"],"evaluated_origin_horizons":counts[c]}
 return {"class_definition":"ADI/CV2 from train-only demand; ADI threshold 1.32, CV2 threshold 0.49.","bootstrap_method":"Circular moving blocks of forecast errors strictly before each validation origin; block length round(n_history^(1/3)); 300 simulations per origin/lead.","limitation":"Prequential validation diagnostic; six prior validation residuals are required. Not independent external validation.","by_demand_class":out}

def probability_calibration():
 raw=load("data/processed/raw_risk_probabilities.parquet"); inv=load("data/processed/simulated_inventory.parquet")
 x=raw[raw.METHOD=="RADCSR"].merge(inv[KEYS+["PERIOD","SCENARIO_ID","STOCKOUT_LABEL"]],on=KEYS+["PERIOD","SCENARIO_ID"],how="inner",validate="many_to_one")
 x=x[x.STOCKOUT_LABEL.notna() & x.SHORTAGE_PROBABILITY.notna()]
 def metric(g,pcol):
  y=g.STOCKOUT_LABEL.to_numpy(float); p=np.clip(g[pcol].to_numpy(float),0,1)
  if not len(y): return {"n":0,"brier":None,"ece_10":None,"calibration_in_large":None}
  ece=0.
  for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
   m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
   if m.any(): ece+=float(m.mean()*abs(y[m].mean()-p[m].mean()))
  return {"n":len(y),"brier":float(np.mean((p-y)**2)),"ece_10":ece,"calibration_in_large":float(y.mean()-p.mean())}
 v=x[x.SPLIT=="validation"].copy(); t=x[x.SPLIT=="test"].copy()
 if len(v)<20 or v.STOCKOUT_LABEL.nunique()<2: raise ValueError(f"Insufficient validation labels for isotonic calibration: n={len(v)}, classes={v.STOCKOUT_LABEL.nunique()}")
 iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip").fit(v.SHORTAGE_PROBABILITY,v.STOCKOUT_LABEL)
 v["CAL"]=iso.predict(v.SHORTAGE_PROBABILITY); t["CAL"]=iso.predict(t.SHORTAGE_PROBABILITY)
 vm=metric(v,"SHORTAGE_PROBABILITY"); vf=metric(v,"CAL"); tm=metric(t,"CAL")
 passed=bool(tm["ece_10"] is not None and tm["ece_10"]<=.10 and abs(tm["calibration_in_large"])<=.05)
 return {"validation_raw":vm,"validation_isotonic_fit_in_sample":vf,"held_out_test_after_validation_isotonic":tm,"held_out_gate":{"ECE_10_max":.10,"absolute_calibration_in_large_max":.05,"pass":passed},"validation_note":"Isotonic fit is trained on validation labels; its fit-sample metrics are not an independent calibration pass.","conclusion":"Held-out descriptive calibration gate passed on simulated labels." if passed else "Calibration gate failed/unavailable. RADCSR_PROBABILITY must not be interpreted as a calibrated probability."}

def signal_perturbations():
 n=40
 base=pd.DataFrame({KEYS[0]:["ORG"]*n,KEYS[1]:["VMP"]*n,KEYS[2]:["U"]*n,"PERIOD":pd.date_range("2020-01-01",periods=n,freq="MS"),"MODEL_DEMAND":100.,"Q10":90.,"Q50":100.,"Q90":110.})
 def last(x): return build_reliability_index(x,window=8).iloc[-1]
 b=last(base); under=base.copy(); under.loc[32:,"MODEL_DEMAND"]+=30
 over=base.copy(); over.loc[32:,"MODEL_DEMAND"]-=30
 noisy=base.copy(); noisy.loc[32:,"MODEL_DEMAND"]+=np.array([30,-30,30,-30,30,-30,30,-30])
 wide=base.copy(); wide["Q10"]=70.; wide["Q90"]=130.
 isolated=base.copy(); isolated.loc[35,"MODEL_DEMAND"]+=3
 u,o,v,w,d=map(last,(under,over,noisy,wide,isolated))
 return {
 "R_E":{"responds_to_absolute_error":bool(u.R_E>b.R_E),"baseline":float(b.R_E),"perturbed":float(u.R_E)},
 "R_B":{"underforecast_penalty_exceeds_overforecast":bool(u.R_B>o.R_B),"symmetric_ablation_equal":bool(np.isclose(u.R_B_SYMMETRIC,o.R_B_SYMMETRIC)),"overforecast_separate":bool(o.R_B_OVER>0),"under_R_B":float(u.R_B),"over_R_B":float(o.R_B),"under_symmetric":float(u.R_B_SYMMETRIC),"over_symmetric":float(o.R_B_SYMMETRIC),"formula":"R_B=clip(2*(sigmoid(2*max(mean(actual-Q50),0)/scale)-0.5),0,1); R_B_OVER applies same transform to max(-mean(error),0)."},
 "R_C":{"interval_width_only_flat":bool(np.isclose(w.R_C,b.R_C)),"responds_to_coverage_error":bool(v.R_C>b.R_C),"baseline":float(b.R_C),"perturbed":float(v.R_C)},
 "R_D":{"isolated_small_error_no_alarm":bool(d.R_D==b.R_D),"baseline":float(b.R_D),"perturbed":float(d.R_D)}
 }

def drift_metrics():
 x=load("data/processed/scenario_streams.parquet"); stable=x[x.SCENARIO_ID=="S1_Stable"]; drift=x[x.SCENARIO_ID=="S3_Drift"]
 flags=stable.R_D.to_numpy(float); delays=[]
 for _,g in drift.groupby(KEYS,sort=False,dropna=False):
  g=g.sort_values("PERIOD"); start=int(g.SHIFT_START_INDEX.iloc[0]) if "SHIFT_START_INDEX" in g else 0
  hit=np.flatnonzero(g.R_D.to_numpy(float)[start:]>0)
  if len(hit): delays.append(int(hit[0]))
 return {"stable_false_alarm_rate":float(np.mean(flags>0)) if len(flags) else None,"stable_periods":len(flags),"stable_false_alarm_count":int(np.sum(flags>0)),"drift_detection_delay_months":{"n_detected_series":len(delays),"median":float(np.median(delays)) if delays else None,"mean":float(np.mean(delays)) if delays else None,"per_series":delays}}

def correlations():
 x=load("data/processed/reliability.parquet"); cols=["R_E","R_B","R_C","R_D"]; c=x[cols].corr(); flags=[]
 for i,a in enumerate(cols):
  for b in cols[i+1:]:
   v=float(c.loc[a,b])
   if np.isfinite(v) and abs(v)>.70: flags.append({"pair":[a,b],"correlation":v,"flag":"potentially redundant"})
 return {"matrix":c.round(6).to_dict(),"threshold":.70,"redundancy_flags":flags,"assessment":"Unflagged signals are provisionally complementary; correlation alone does not prove redundancy."}

def run():
 sweeps=monotonicity(); failed=[x for x in sweeps if not x["pass"]]
 result={"disclaimer":DISCLAIMER,"probability_formulas":FORMULAS,"monotonicity_sweeps":sweeps}
 # A monotonicity failure is a hard stop: print the exact sequences and do not run later checks.
 print(DISCLAIMER); print("1. MONOTONICITY SWEEPS")
 for x in sweeps: print(x["quantity"],"|",x["regime"],"|","PASS" if x["pass"] else "FAIL","| sequence=",x["sequence"])
 if failed:
  print("MONOTONICITY FAILURE: stopping. Failing sequences above are verbatim.")
  return result
 result["lead_time_distribution_validation"]=lead_time_validation()
 result["probability_calibration"]=probability_calibration()
 result["reliability_signal_perturbations"]=signal_perturbations()
 result["drift_detection"]=drift_metrics()
 result["signal_correlations"]=correlations()
 out=ROOT/"output"; out.mkdir(parents=True,exist_ok=True)
 (out/"mathematical_validation.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
 lines=["# Phase B Mathematical Validation","",DISCLAIMER,"","## Dual-probability formulas",json.dumps(FORMULAS,indent=2),"","## Monotonicity sweeps"]
 for x in sweeps: lines.append(f"{x['quantity']} / {x['regime']}: {'PASS' if x['pass'] else 'FAIL'}; sequence={x['sequence']}")
 for key in ("lead_time_distribution_validation","probability_calibration","reliability_signal_perturbations","drift_detection","signal_correlations"):
  lines += ["",f"## {key}",json.dumps(result[key],indent=2,default=str)]
 (out/"mathematical_validation.md").write_text("\n".join(lines),encoding="utf-8")
 print("2. CALIBRATION PER DEMAND CLASS"); print(json.dumps(result["lead_time_distribution_validation"],indent=2))
 print("3. ASYMMETRIC-BIAS SENSITIVITY"); print(json.dumps(result["reliability_signal_perturbations"]["R_B"],indent=2))
 print("4. DRIFT DETECTION"); print(json.dumps(result["drift_detection"],indent=2))
 print("5. SIGNAL CORRELATION MATRIX / REDUNDANCY FLAGS"); print(json.dumps(result["signal_correlations"],indent=2))
 print("RADCSR_PROBABILITY CALIBRATION"); print(json.dumps(result["probability_calibration"],indent=2))
 print("Reports: output/mathematical_validation.md and output/mathematical_validation.json")
 return result

if __name__=="__main__": run()
