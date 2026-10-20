from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
root=Path('/Users/shreyanshgoyal/Documents/compi');p=root/'gemini_3_1_pro_label_generation';out=root/'audits/gemini_v2_pilot'
m=json.loads((p/'pilot_manifest.json').read_text());schema=json.loads((p/'label_schema_v2.json').read_text())
for f,h in m['files'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
keys=[k for k in schema['required'] if k!='StudyInstanceUID']
records=[];provenance=[]
for b in m['batches']:
 assert hashlib.sha256((p/b['input']).read_bytes()).hexdigest()==b['sha256']
 rs=[json.loads(l) for l in (p/b['output']).read_text().splitlines() if l.strip()]
 assert len(rs)==b['count'] and {r['StudyInstanceUID'] for r in rs}==set(b['ids'])
 for r in rs:
  assert set(r)==set(schema['required'])
  for k in keys:assert r[k] in schema['properties'][k]['enum']
 records.extend(rs)
 provenance.append(json.loads((p/b['output']).with_suffix('.provenance.json').read_text()))
merged=p/'output_v2/validate_gemini_3_1_pro_v2.jsonl'
actual=[json.loads(l) for l in merged.read_text().splitlines() if l.strip()]
assert actual==records and len({r['StudyInstanceUID'] for r in records})==58
# Freeze the evaluated predictions outside the workers' annotation directory.
(out/'predictions_evaluated.jsonl').write_bytes(merged.read_bytes())
cols=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold=pd.read_csv(root/'sonnet_label_completion/train.csv').set_index('StudyInstanceUID')
pred=pd.DataFrame(records).set_index('StudyInstanceUID');gold=gold.loc[pred.index]
assert gold[cols].notna().all().all()
rows=[]
for k,c in zip(keys,cols):
 y=gold[c].to_numpy(float);score=pred[k].map({'absent':0.,'definite':1.,'unsure':np.nan}).to_numpy(float)
 assert set(y)<= {0.,1.}
 ok=np.isfinite(score);a=y[ok];b=score[ok];tp=int(((a==1)&(b==1)).sum());tn=int(((a==0)&(b==0)).sum());fp=int(((a==0)&(b==1)).sum());fn=int(((a==1)&(b==0)).sum())
 sens=tp/(tp+fn) if tp+fn else np.nan;spec=tn/(tn+fp) if tn+fp else np.nan
 rows.append(dict(finding=c,covered=int(ok.sum()),total=len(y),abstentions=int((~ok).sum()),coverage=float(ok.mean()),tp=tp,tn=tn,fp=fp,fn=fn,sensitivity=sens,specificity=spec,balanced_accuracy=(sens+spec)/2,auc_covered=roc_auc_score(a,b) if len(set(a))==2 else np.nan,auc_unsure_neutral=roc_auc_score(y,np.nan_to_num(score,nan=.5)) if len(set(y))==2 else np.nan))
df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
summary={'prediction_sha256':hashlib.sha256(merged.read_bytes()).hexdigest(),'records':58,'total_label_cells':696,'covered_cells':int(df.covered.sum()),'abstentions':int(df.abstentions.sum()),'macro_auc_covered':float(df.auc_covered.mean()),'macro_auc_unsure_neutral':float(df.auc_unsure_neutral.mean()),'provenance_status':'unverified; batch_02 simulated_direct_output; batch_04 mock/direct inference; other identities self-reported','usage':'unavailable'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));print(df[['finding','covered','sensitivity','specificity','auc_covered']].round(3).to_string(index=False))
