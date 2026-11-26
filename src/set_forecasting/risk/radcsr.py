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
    mu=q50*l;sigma=np.maximum((q90-q10)/2.563,1e-6)*np.sqrt(l)
    return mu,sigma

def _tail(mu,sigma,availability):
    return np.clip(1-ndtr((np.asarray(availability)-mu)/np.maximum(sigma,1e-8)),0,1)

def _raw_at_r(q10,q50,q90,availability,lead,r,ks,km,kl):
    mu,sigma=infer_lead_time_distribution(q10,q50,q90,lead)
    a=np.asarray(availability,float);l=np.maximum(np.asarray(lead,float),1.)
    r=np.asarray(r,float)
    sigma_adj=sigma*(1+ks*r)
    mu_adj=mu+np.where(a<mu,mu*km*r,0.)
    l_eff=l*(1+kl*r)
    ratio=l_eff/l
    mu_eff=mu_adj*ratio
    sigma_eff=sigma_adj*np.sqrt(ratio)
    return _tail(mu_eff,sigma_eff,a),mu,sigma,sigma_adj,mu_adj,l_eff

def radcsr_probability(q10,q50,q90,availability,lead_time,r_rel,
                       kappa_sigma=.75,kappa_mu=.50,kappa_l=.50,grid_size=101):
    """Probability plus monotone envelope for each fixed case across R_rel in [0,1]."""
    r=np.clip(np.asarray(r_rel,float),0,1)
    raw,mu,sigma,sadj,madj,leff=_raw_at_r(q10,q50,q90,availability,lead_time,r,kappa_sigma,kappa_mu,kappa_l)
    arrays=[np.asarray(v,float).reshape(-1) for v in [q10,q50,q90,availability,lead_time,r]]
    q10a,q50a,q90a,aa,la,ra=arrays
    raw_flat=np.asarray(raw).reshape(-1);prob=np.empty_like(raw_flat)
    for i,(a,b,c,avail,lead,rel) in enumerate(zip(q10a,q50a,q90a,aa,la,ra)):
        grid=np.linspace(0,rel,max(2,int(grid_size*max(rel,1/grid_size))))
        curve=_raw_at_r(np.full(len(grid),a),np.full(len(grid),b),np.full(len(grid),c),
                        np.full(len(grid),avail),np.full(len(grid),lead),grid,
                        kappa_sigma,kappa_mu,kappa_l)[0]
        prob[i]=max(float(raw_flat[i]),float(np.max(curve)))
    return pd.DataFrame({"MU_L":np.asarray(mu).reshape(-1),"SIGMA_L":np.asarray(sigma).reshape(-1),
      "SIGMA_ADJ":np.asarray(sadj).reshape(-1),"MU_ADJ":np.asarray(madj).reshape(-1),
      "L_EFF":np.asarray(leff).reshape(-1),"RADCSR_RAW_PROBABILITY":raw_flat,
      "RADCSR_PROBABILITY":prob,"R_REL":ra,"STOCKOUTS_ARE_OBSERVED":0,"OUTCOME_TYPE":"SIMULATED"})

def monotonicity_audit(q10,q50,q90,availability,lead_time,ks=.75,km=.50,kl=.50):
    grid=np.linspace(0,1,101)
    raw=_raw_at_r(np.full(101,q10),np.full(101,q50),np.full(101,q90),
      np.full(101,availability),np.full(101,lead_time),grid,ks,km,kl)[0]
    adjusted=np.maximum.accumulate(raw)
    return {"raw_pass":bool(np.all(np.diff(raw)>=-1e-10)),
      "envelope_pass":bool(np.all(np.diff(adjusted)>=-1e-10)),
      "raw_minimum_step":float(np.min(np.diff(raw))),
      "probability_curve":adjusted.tolist()}
