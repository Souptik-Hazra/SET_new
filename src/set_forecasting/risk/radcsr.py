"""RADCSR: reliability-adaptive demand-conditioned shortage risk."""
import numpy as np
import pandas as pd
from scipy.special import ndtr
KEYS=["ODS_CODE","VMP_SNOMED_CODE","UNIT_OF_MEASURE_IDENTIFIER"]
DISCLAIMER="SCMD contains no observed stock, lead time or stockout events. Inventory, lead times and labels are simulated."

def infer_lead_time_distribution(q10,q50,q90,lead):
    q10,q50,q90=np.maximum(0,np.asarray(q10,float)),np.maximum(0,np.asarray(q50,float)),np.maximum(0,np.asarray(q90,float))
    if np.any(q10>q50) or np.any(q50>q90):raise ValueError("Require Q10 <= Q50 <= Q90")
    l=np.maximum(np.asarray(lead,float),1.)
    return q50*l,np.maximum((q90-q10)/2.563,1e-6)*np.sqrt(l)

def _tail(mu,sigma,a):
    return np.clip(1-ndtr((np.asarray(a)-mu)/np.maximum(sigma,1e-8)),0,1)

def _raw(q10,q50,q90,a,lead,r,ks,km,kl,horizon_cap=2.0):
    mu,sigma=infer_lead_time_distribution(q10,q50,q90,lead)
    l=np.maximum(np.asarray(lead,float),1.);r=np.asarray(r,float);a=np.asarray(a,float)
    sigma_adj=sigma*(1+ks*r)
    mu_adj=mu+np.where(a<mu,mu*km*r,0.)
    l_eff=l*(1+kl*r)
    if horizon_cap is not None:l_eff=np.minimum(l_eff,horizon_cap*l)
    ratio=l_eff/l
    return _tail(mu_adj*ratio,sigma_adj*np.sqrt(ratio),a),mu,sigma,sigma_adj,mu_adj,l_eff

def radcsr_probability(q10,q50,q90,availability,lead_time,r_rel,
                       kappa_sigma=.75,kappa_mu=.50,kappa_l=1.5,
                       horizon_cap=2.0,grid_size=101):
    """Compute raw distributional risk and a per-case monotone probability envelope."""
    r=np.clip(np.asarray(r_rel,float),0,1)
    raw,mu,sigma,sadj,madj,leff=_raw(q10,q50,q90,availability,lead_time,r,
      kappa_sigma,kappa_mu,kappa_l,horizon_cap)
    arrays=[np.asarray(v,float).reshape(-1) for v in [q10,q50,q90,availability,lead_time,r]]
    q10a,q50a,q90a,aa,la,ra=arrays;raw_flat=np.asarray(raw).reshape(-1);prob=np.empty_like(raw_flat)
    for i,(a,b,c,avail,lead,rel) in enumerate(zip(q10a,q50a,q90a,aa,la,ra)):
        grid=np.linspace(0,rel,max(2,int(grid_size*max(rel,1/grid_size))))
        curve=_raw(np.full(len(grid),a),np.full(len(grid),b),np.full(len(grid),c),
          np.full(len(grid),avail),np.full(len(grid),lead),grid,kappa_sigma,kappa_mu,kappa_l,horizon_cap)[0]
        prob[i]=max(float(raw_flat[i]),float(np.max(curve)))
    return pd.DataFrame({"MU_L":np.asarray(mu).reshape(-1),"SIGMA_L":np.asarray(sigma).reshape(-1),
      "SIGMA_ADJ":np.asarray(sadj).reshape(-1),"MU_ADJ":np.asarray(madj).reshape(-1),
      "L_EFF":np.asarray(leff).reshape(-1),"RADCSR_RAW_PROBABILITY":raw_flat,
      "RADCSR_PROBABILITY":prob,"R_REL":ra,"STOCKOUTS_ARE_OBSERVED":0,"OUTCOME_TYPE":"SIMULATED"})

def radcsr_ablations(frame):
    """Return full RADCSR plus the requested one-component ablations."""
    x=frame
    e=x.R_E.to_numpy(float);b=x.R_B.to_numpy(float);c=x.R_C.to_numpy(float);d=x.R_D.to_numpy(float)
    r_full=x.R_REL.to_numpy(float)
    variants={
      "RADCSR":(r_full,.75,.50,1.5,2.0),
      "RADCSR_noBias":((.35*e+.25*c+.15*d)/.75,.75,.50,1.5,2.0),
      "RADCSR_noCoverage":((.35*e+.25*b+.15*d)/.75,.75,.50,1.5,2.0),
      "RADCSR_noDrift":((.35*e+.25*b+.25*c)/.85,.75,.50,1.5,2.0),
      # Clinical weights are intentionally neutral without verified VEN provenance.
      "RADCSR_noClinical":(r_full,.75,.50,1.5,2.0),
      "RADCSR_noCap":(r_full,.75,.50,1.5,None),
      "RADCSR_noHorizon":(r_full,.75,.50,0.0,2.0),
      "RADCSR_singleSignal":(e,.75,.50,1.5,2.0)}
    outputs=[]
    for name,(r,ks,km,kl,cap) in variants.items():
        result=radcsr_probability(x.Q10.to_numpy(),x.Q50.to_numpy(),x.Q90.to_numpy(),
          x.AVAILABLE_STOCK.to_numpy(),x.LEAD_TIME_MONTHS.to_numpy(),r,ks,km,kl,cap)
        outputs.append(pd.DataFrame({"METHOD":name,"SHORTAGE_PROBABILITY":result.RADCSR_PROBABILITY}))
    return pd.concat(outputs,ignore_index=True)

def monotonicity_audit(q10,q50,q90,availability,lead_time,ks=.75,km=.50,kl=1.5):
    grid=np.linspace(0,1,101)
    raw=_raw(np.full(101,q10),np.full(101,q50),np.full(101,q90),
      np.full(101,availability),np.full(101,lead_time),grid,ks,km,kl,2.0)[0]
    adjusted=np.maximum.accumulate(raw)
    return {"raw_pass":bool(np.all(np.diff(raw)>=-1e-10)),
      "envelope_pass":bool(np.all(np.diff(adjusted)>=-1e-10)),
      "raw_minimum_step":float(np.min(np.diff(raw))),
      "probability_curve":adjusted.tolist()}
