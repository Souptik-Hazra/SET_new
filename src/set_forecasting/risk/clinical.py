"""Optional evidence-backed VEN weighting. No clinical class is inferred from SCMD demand."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

REQUIRED=["VMP_SNOMED_CODE","VEN_CLASS","SUBSTITUTABLE","SOURCE_NAME","SOURCE_VERSION","SOURCE_URL"]

def load_verified_ven_mapping(path):
    mapping=pd.read_csv(path,dtype={"VMP_SNOMED_CODE":str,"VEN_CLASS":str})
    missing=set(REQUIRED)-set(mapping.columns)
    if missing:raise ValueError(f"VEN mapping missing provenance fields: {sorted(missing)}")
    mapping["VEN_CLASS"]=mapping.VEN_CLASS.str.upper().str.strip()
    if not set(mapping.VEN_CLASS.dropna()).issubset({"V","E","N"}):raise ValueError("VEN_CLASS must be V, E or N")
    mapping["SUBSTITUTABLE"]=pd.to_numeric(mapping.SUBSTITUTABLE,errors="raise")
    if not mapping.SUBSTITUTABLE.isin([0,1]).all():raise ValueError("SUBSTITUTABLE must be 0 or 1")
    if mapping.VMP_SNOMED_CODE.duplicated().any():raise ValueError("VEN mapping must have one row per VMP SNOMED code")
    if mapping[["SOURCE_NAME","SOURCE_VERSION","SOURCE_URL"]].isna().any().any():raise ValueError("VEN source provenance cannot be blank")
    return mapping

def compare_clinical_priority(risk_frame,mapping):
    x=risk_frame.merge(mapping,on="VMP_SNOMED_CODE",how="inner",validate="many_to_one")
    ven=x.VEN_CLASS.map({"V":3.,"E":2.,"N":1.})
    x["CLINICAL_WEIGHT"]=.6*ven+.4*(1-x.SUBSTITUTABLE)
    x["PRIORITY_UNWEIGHTED"]=x.RADCSR_PROBABILITY
    x["PRIORITY_CLINICAL"]=x.RADCSR_PROBABILITY*x.CLINICAL_WEIGHT
    rows=[]
    for scenario,g in x.groupby("SCENARIO_ID"):
        base=g.groupby("VMP_SNOMED_CODE").PRIORITY_UNWEIGHTED.mean().rank(ascending=False,method="average")
        weighted=g.groupby("VMP_SNOMED_CODE").PRIORITY_CLINICAL.mean().rank(ascending=False,method="average")
        rho=float(spearmanr(base,weighted).statistic) if len(base)>1 else float("nan")
        rows.append({"SCENARIO_ID":scenario,"MAPPED_MEDICINES":len(base),"SPEARMAN_RANK_CORRELATION":rho,
            "RANKING_CHANGED":bool(not base.equals(weighted))})
    return x,pd.DataFrame(rows)
