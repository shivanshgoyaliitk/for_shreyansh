"""Out-of-fold check of family blend weights against the pseudo (report-label) targets.

Run on Kaggle (CPU) with the session outputs attached, or locally with a folder of them:
    python scripts/oof_blend_check.py /path/to/outputs
Each family folder (reader_convnext/, reader_convnext_s7/, slot_dinov2/, radimagenet/) holds
oof_fold_N.csv and targets_used.csv. Gold studies never enter OOF (they train in every fold),
so this check never consults gold. Pseudo AUC is a noisy-label proxy: use it to compare blends,
not as a test-score estimate.
"""
import os, sys, itertools, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Inside Jupyter, argv holds the kernel's own flags; only a directory argument names the root.
ROOT = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else '/kaggle/input'
LABEL_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def collect(root):
    members, targets = {}, None
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        name = os.path.basename(path)
        oofs = sorted(f for f in files if f.startswith('oof_fold_') and f.endswith('.csv'))
        if oofs:
            frame = pd.concat([pd.read_csv(os.path.join(path, f), dtype={'StudyInstanceUID': str})
                               for f in oofs], ignore_index=True)
            members.setdefault(name, []).append(frame)
        if 'targets_used.csv' in files:
            t = pd.read_csv(os.path.join(path, 'targets_used.csv'), dtype={'StudyInstanceUID': str})
            if targets is None:
                targets = t
            else:
                same = t.set_index('StudyInstanceUID')[LABEL_COLS].reindex(targets['StudyInstanceUID'])
                assert np.allclose(same.to_numpy(float), targets[LABEL_COLS].to_numpy(float),
                                   equal_nan=True), f'targets differ in {path}'
    assert targets is not None, 'no targets_used.csv found'
    frames = {}
    for name, parts in members.items():
        frame = pd.concat(parts, ignore_index=True)
        assert frame['StudyInstanceUID'].is_unique, f'{name}: a study is OOF in two attached folds'
        frames[name] = frame.set_index('StudyInstanceUID')
    return frames, targets


def aucs(truth, score):
    out = []
    for k in range(truth.shape[1]):
        ok = np.isfinite(truth[:, k]) & np.isfinite(score[:, k])
        y = (truth[ok, k] >= .5).astype(int)
        out.append(roc_auc_score(y, score[ok, k]) if len(np.unique(y)) == 2 else np.nan)
    return np.array(out)


def pct_rank(frame):
    return frame[LABEL_COLS].rank(pct=True, method='average')


frames, targets = collect(ROOT)
pseudo = targets[targets['source'].eq('pseudo')].set_index('StudyInstanceUID')
print('Members found:', {k: (len(v), sorted(v['fold'].unique().tolist())) for k, v in frames.items()})

# 1. Every member on the studies it covers, and seed members against seed 42 on the same studies.
for name, frame in sorted(frames.items()):
    ids = frame.index.intersection(pseudo.index)
    a = aucs(pseudo.loc[ids, LABEL_COLS].to_numpy(float), frame.loc[ids, LABEL_COLS].to_numpy(float))
    print(f'{name:24s} n={len(ids):5d} macro={np.nanmean(a):.4f}')
for name in sorted(frames):
    base = name.split('_s')[0] if '_s' in name and name.rsplit('_s', 1)[1].isdigit() else None
    if base and base in frames:
        ids = frames[name].index.intersection(frames[base].index).intersection(pseudo.index)
        truth = pseudo.loc[ids, LABEL_COLS].to_numpy(float)
        one = frames[base].loc[ids]; two = frames[name].loc[ids]
        both = (pct_rank(one) + pct_rank(two)) / 2
        print(f'  same {len(ids)} studies: {base}={np.nanmean(aucs(truth, one[LABEL_COLS].to_numpy(float))):.4f} '
              f'{name}={np.nanmean(aucs(truth, two[LABEL_COLS].to_numpy(float))):.4f} '
              f'rank-mean={np.nanmean(aucs(truth, both.to_numpy())):.4f}')

# 2. Family weights on the studies every complete seed-42 family covers.
complete = [n for n, f in frames.items() if f.index.intersection(pseudo.index).size == pseudo.index.size]
print('Complete families:', complete)
ids = pseudo.index
for n in complete: ids = ids.intersection(frames[n].index)
truth = pseudo.loc[ids, LABEL_COLS].to_numpy(float)
ranks = {n: pct_rank(frames[n].loc[ids]).to_numpy() for n in complete}
grid = [w for w in itertools.product(np.arange(0, 1.01, .1), repeat=len(complete)) if abs(sum(w) - 1) < 1e-9]
rows = []
for w in grid:
    blend = sum(wi * ranks[n] for wi, n in zip(w, complete))
    a = aucs(truth, blend)
    rows.append(dict(zip(complete, np.round(w, 2)), macro=np.nanmean(a), **dict(zip(LABEL_COLS, a))))
table = pd.DataFrame(rows).sort_values('macro', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\nTop family weights (macro pseudo AUC):')
print(table[complete + ['macro']].head(12).to_string(index=False))
equal = sum(ranks[n] for n in complete) / len(complete)
print(f'\nEqual family weights (current submission): {np.nanmean(aucs(truth, equal)):.4f}')
print('\nEach family alone:')
for n in complete:
    print(f'  {n}: {table.loc[np.isclose(table[n], 1.0), "macro"].iloc[0]:.4f}')
# Per-label best weights (optimistic: chosen on the same OOF they are scored on).
best = {c: table.sort_values(c, ascending=False).iloc[0] for c in LABEL_COLS}
print('\nPer-label best weights (optimistic, in-sample on OOF):')
for c, r in best.items():
    print(f'  {c:18s}', {n: float(r[n]) for n in complete}, f'{r[c]:.4f}')
print('Per-label best macro (optimistic):', np.mean([r[c] for c, r in best.items()]))
json.dump(dict(top=table.head(30).to_dict('records')), open('oof_blend_check.json', 'w'), indent=1, default=float)
print('OOF BLEND CHECK DONE')
