"""Evaluate frozen pilot predictions outside blinded annotation folders."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
source=ROOT/'sol_label_generation'; out=Path(__file__).resolve().parent
manifest=json.loads((source/'protocol_v2_manifest.json').read_text())
for file,h in manifest['files'].items():
    assert hashlib.sha256((source/file).read_bytes()).hexdigest()==h, file
provenance=json.loads((source/'output_v2/provenance.json').read_text())
for file,h in provenance['hashes'].items():
    assert hashlib.sha256((source/file).read_bytes()).hexdigest()==h, file
schema=json.loads((source/'label_schema_v2.json').read_text())
keys=[k for k in schema['required'] if k!='StudyInstanceUID']
columns=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
path=source/'output_v2/validate_labels.jsonl'
records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
ids=pd.read_csv(source/'validation_reports.csv')['StudyInstanceUID'].tolist()
assert len(records)==58 and [r['StudyInstanceUID'] for r in records]==ids and len(set(ids))==58
for r in records:
    assert set(r)==set(schema['required'])
    for k in keys: assert r[k] in schema['properties'][k]['enum']
freeze=out/'predictions_evaluated.jsonl'
if freeze.exists(): assert freeze.read_bytes()==path.read_bytes(), 'Existing audit snapshot differs'
else: freeze.write_bytes(path.read_bytes())
gold=pd.read_csv(ROOT/'sonnet_label_completion/train.csv').set_index('StudyInstanceUID').loc[ids,columns]
assert gold.notna().all().all()
pred=pd.DataFrame(records).set_index('StudyInstanceUID').loc[ids]
gemini=pd.DataFrame([json.loads(l) for l in (ROOT/'audits/gemini_v2_pilot/predictions_evaluated.jsonl').read_text().splitlines()]).set_index('StudyInstanceUID').loc[ids]
rows=[]
for k,c in zip(keys,columns):
    y=gold[c].to_numpy(float); s=pred[k].map({'absent':0.,'definite':1.,'unsure':np.nan}).to_numpy(float)
    g=gemini[k].map({'absent':0.,'definite':1.,'unsure':np.nan}).to_numpy(float)
    assert set(y)<={0,1}
    ok=np.isfinite(s);a=y[ok];b=s[ok]
    tp=int(((a==1)&(b==1)).sum());tn=int(((a==0)&(b==0)).sum());fp=int(((a==0)&(b==1)).sum());fn=int(((a==1)&(b==0)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    shared=ok&np.isfinite(g)
    auc=lambda y,s:float(roc_auc_score(y,s)) if len(set(y))==2 else np.nan
    rows.append(dict(finding=c,covered=int(ok.sum()),abstentions=int((~ok).sum()),tp=tp,tn=tn,fp=fp,fn=fn,sensitivity=sens,specificity=spec,auc_covered=auc(a,b),auc_unsure_neutral=auc(y,np.nan_to_num(s,nan=.5)),gemini_original_auc_unsure_neutral=auc(y,np.nan_to_num(g,nan=.5)),shared_covered=int(shared.sum()),sol_auc_shared=auc(y[shared],s[shared]),gemini_original_auc_shared=auc(y[shared],g[shared])))
df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
summary=dict(model=provenance['model'],prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=len(records),covered_cells=int(df.covered.sum()),total_cells=58*12,abstentions=int(df.abstentions.sum()),macro_auc_covered=float(df.auc_covered.mean()),macro_auc_unsure_neutral=float(df.auc_unsure_neutral.mean()),gemini_original_macro_auc_unsure_neutral=float(df.gemini_original_auc_unsure_neutral.mean()),sol_macro_auc_shared=float(df.sol_auc_shared.mean()),gemini_original_macro_auc_shared=float(df.gemini_original_auc_shared.mean()),usage_attributable=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2)); print(df[['finding','covered','sensitivity','specificity','auc_covered']].round(3).to_string(index=False))
