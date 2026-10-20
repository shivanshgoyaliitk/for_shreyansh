"""Fold creation and metrics with explicit gold exclusion."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def assign_folds(df, label_cols, n_splits=5, seed=42, group_col=None):
    folds = np.full(len(df), -1, dtype=int)
    pseudo = np.flatnonzero(df['source'].eq('pseudo').to_numpy())
    assert len(pseudo) >= n_splits
    y = df.iloc[pseudo][label_cols].to_numpy(float)
    # Balance missingness as well as positive findings. No gold target enters the split.
    strata = np.concatenate([np.nan_to_num(y, nan=0) >= .5, np.isfinite(y)], axis=1).astype(int)
    groups = df.iloc[pseudo][group_col].astype(str).to_numpy() if group_col else df.iloc[pseudo]['StudyInstanceUID'].to_numpy()
    if len(np.unique(groups)) < len(groups):
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(pseudo, groups=groups)
    else:
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(pseudo, strata)
    for fold, (_, val) in enumerate(splits): folds[pseudo[val]] = fold
    assert (folds[pseudo] >= 0).all()
    assert (folds[df['source'].eq('gold')] == -1).all()
    return folds


def macro_auc(y, scores, label_cols):
    aucs = {}
    for k, name in enumerate(label_cols):
        ok = np.isfinite(y[:, k]) & np.isfinite(scores[:, k])
        binary = (y[ok, k] >= .5).astype(int)
        aucs[name] = roc_auc_score(binary, scores[ok, k]) if len(np.unique(binary)) == 2 else np.nan
    valid = [v for v in aucs.values() if np.isfinite(v)]
    return (float(np.mean(valid)) if valid else float('nan')), aucs


def rank_blend(predictions):
    arrays = [np.asarray(p, dtype=float) for p in predictions]
    assert arrays and all(a.shape == arrays[0].shape for a in arrays)
    assert all(np.isfinite(a).all() for a in arrays)
    return np.mean([pd.DataFrame(a).rank(pct=True, method='average').to_numpy() for a in arrays], axis=0)
