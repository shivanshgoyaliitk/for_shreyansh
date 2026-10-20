"""Teacher targets for noisy-student training, from our own models' out-of-fold predictions.

Run on Kaggle (CPU) with session outputs attached, or locally: python scripts/teacher_targets.py DIR
Writes teacher_targets.csv (StudyInstanceUID, twelve probabilities, n_members) and teacher_meta.json.
The logic is label_builder.build_teacher (embedded above this cell in 07_teacher_targets.ipynb):
every value is out-of-fold, seeds of a family are averaged first, then families by
TEACHER_FAMILIES. Gold studies have no OOF rows and never receive teacher values. Only our own
models are teachers, so students stay entirely our own work.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if '__file__' in globals():                    # local run: import the shared implementation
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from label_builder import build_teacher

ROOT = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else '/kaggle/input'
OUT = os.environ.get('TEACHER_OUT', '.' if ROOT != '/kaggle/input' else '/kaggle/working')
LABEL_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

targets = None
for path, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
    if 'targets_used.csv' in files:
        targets = pd.read_csv(os.path.join(path, 'targets_used.csv'), dtype={'StudyInstanceUID': str})
        break
assert targets is not None, 'no targets_used.csv found'
gold = set(targets.loc[targets['source'].eq('gold'), 'StudyInstanceUID'])
teacher, meta = build_teacher(ROOT, LABEL_COLS, gold_ids=gold)

# Diagnostics against the report consensus (the targets the teachers learned from).
pseudo = targets[targets['source'].eq('pseudo')].set_index('StudyInstanceUID')
t = teacher.set_index('StudyInstanceUID')
common = t.index.intersection(pseudo.index)
rows = []
for c in LABEL_COLS:
    y, p = pseudo.loc[common, c].to_numpy(float), t.loc[common, c].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    b = (y[ok] >= .5).astype(int)
    rows.append(dict(finding=c, auc_vs_report=roc_auc_score(b, p[ok]) if len(set(b)) == 2 else np.nan,
                     report_mean=float(np.nanmean(y)), teacher_mean=float(np.nanmean(p)),
                     report_missing=float(np.isnan(y).mean())))
diag = pd.DataFrame(rows).set_index('finding')
print(json.dumps(meta, indent=1))
print(diag.round(3).to_string())
print(f'teacher studies: {len(teacher)} of {len(pseudo)} pseudo; members per study:',
      teacher['n_members'].value_counts().sort_index().to_dict())
meta['diagnostics'] = diag.round(4).reset_index().to_dict('records')
os.makedirs(OUT, exist_ok=True)
teacher.to_csv(os.path.join(OUT, 'teacher_targets.csv'), index=False)
with open(os.path.join(OUT, 'teacher_meta.json'), 'w') as handle:
    json.dump(meta, handle, indent=1, default=float)
print('TEACHER TARGETS WRITTEN', os.path.join(OUT, 'teacher_targets.csv'))
