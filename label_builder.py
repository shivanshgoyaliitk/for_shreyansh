"""Soft training targets from report-label readers; embedded in training.ipynb."""
import os, re, json, fnmatch, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Public report-label tables (rules allow public external data). File name -> suffix of
# the twelve target columns. The mean of these four reached 0.896 macro AUC on the gold
# studies in the public 2.5D reader's own measurement.
PUBLIC_LABEL_TABLES = {
    'report_labels_v2.csv': '',            # pilkwang/rsna-knee-llm-labels
    'llm_labels_v4_blend.csv': '',         # stevenleehans/rsna-knee-llm-report-labels
    'labels_llm_gpt56sol.csv': '',         # lixin73/rsna-knee-llm-report-labels-sol56
    'qwen_knee_weak_labels.csv': '__label',  # laymond/rsna-knee-abnormality-qwen3-8b-weak-labels
}
# Our own readers:
#  - extra_labels_*.csv from scripts/export_reader_labels.py (GPT Sol v2, Sonnet legacy);
#  - the Flash member file written by our 01_labels notebook (its labels/ folder), read
#    directly so Flash counts once. That notebook's consensus train_targets.csv is not read:
#    it already mixes Flash with Sonnet.
OWN_PREFIX = 'extra_labels'
OWN_JSONL_READERS = {'pseudo_google_gemini*flash*.jsonl': 'legacy'}

# State -> probability that the finding is positive. Legacy is the 01_labels notebook's own
# HARD map; v2 states follow the knee-report-labels-v2 protocol. Unsure abstains (NaN)
# under legacy; under v2 it marks an affirmative finding of undecidable eligibility.
STATE_P = {
    'legacy': {'definite': 1.0, 'probable': .75, 'possible': .35, 'absent': 0.0},
    'v2': {'definite': 1.0, 'unsure': .5, 'absent': 0.0},
}
STATE_KEYS = dict(zip(
    ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"],
    ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa",
     "pf_oa", "effusion", "synovitis", "bakers", "contusion", "fracture"]))


def iter_jsonl(path):
    # One historical shard was written with literal "\n" separators; split on both.
    with open(path) as handle:
        for chunk in re.split(r'\n|\\n', handle.read()):
            if chunk.strip():
                yield json.loads(chunk)


def states_frame(records, label_cols, protocol):
    mapping = STATE_P[protocol]
    rows = [{'StudyInstanceUID': str(r['StudyInstanceUID']),
             **{c: mapping.get(str(r.get(STATE_KEYS.get(c, c), '')).lower()) for c in label_cols}}
            for r in records]
    frame = pd.DataFrame(rows, columns=['StudyInstanceUID', *label_cols])
    return frame.drop_duplicates('StudyInstanceUID', keep='last')


def find_label_tables(root='/kaggle/input'):
    found = {}
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        for name in files:
            if (name in PUBLIC_LABEL_TABLES or (name.startswith(OWN_PREFIX) and name.endswith('.csv'))
                    or any(fnmatch.fnmatch(name, pattern) for pattern in OWN_JSONL_READERS)):
                found.setdefault(name, []).append(os.path.join(path, name))
    duplicates = {k: v for k, v in found.items() if len(v) > 1}
    assert not duplicates, f'Attach each label table once: {duplicates}'
    flash = [k for k in found if any(fnmatch.fnmatch(k, p) for p in OWN_JSONL_READERS)]
    assert len(flash) <= 1, f'More than one Flash member file attached: {flash}'
    return {k: v[0] for k, v in sorted(found.items())}


def read_reader(path, label_cols):
    name = os.path.basename(path)
    if name.endswith('.jsonl'):
        protocol = next(v for p, v in OWN_JSONL_READERS.items() if fnmatch.fnmatch(name, p))
        frame = states_frame(iter_jsonl(path), label_cols, protocol)
        columns = list(label_cols)
    else:
        suffix = PUBLIC_LABEL_TABLES.get(name, '')
        frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        columns = [c + suffix for c in label_cols]
    missing = [c for c in columns if c not in frame.columns]
    assert not missing, f'{name} lacks {missing[:3]}'
    assert not frame['StudyInstanceUID'].duplicated().any(), f'{name} repeats studies'
    values = frame.set_index('StudyInstanceUID')[columns].apply(pd.to_numeric, errors='coerce')
    values.columns = label_cols
    bad = (values < 0) | (values > 1)
    if bad.any().any():
        print(f'{name}: {int(bad.sum().sum())} values outside 0..1 treated as unobserved')
        values = values.mask(bad)
    return values


# Noisy-student targets (2026-10-08). teacher_targets.csv holds out-of-fold predictions of our own
# image models (scripts/teacher_targets.py); image models average out report noise, so their OOF
# predictions are cleaner targets than any single report reader. Gold rows never take teacher values.
TEACHER_FILE = 'teacher_targets.csv'


def find_teacher(root='/kaggle/input'):
    hits = []
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if TEACHER_FILE in files:
            hits.append(os.path.join(path, TEACHER_FILE))
    assert len(hits) <= 1, f'Attach one teacher table, found {hits}'
    return hits[0] if hits else None


# Family -> weight for teachers. Readers only: on the public LB the slot and RadImageNet families
# lowered the blend (17 models at equal weights 0.917 < two reader folds 0.923). Students and
# third-party checkpoints are never teachers.
TEACHER_FAMILIES = {'reader_convnext': 1.0, 'reader_caformer': 1.0}


def build_teacher(root, label_cols, gold_ids=(), families=None, quantile_map=()):
    """Teacher probabilities from the out-of-fold predictions (oof_fold_N.csv) of our own models
    found under root: seeds of a family are averaged, then families by weight. Every value is
    out-of-fold. Families in quantile_map (an outside model trained on other label definitions)
    keep their per-finding ranking but take the value distribution of the weighted mean of the
    other families before the average. Returns (frame with StudyInstanceUID, label_cols, n_members; meta)."""
    families = TEACHER_FAMILIES if families is None else families
    family_of = lambda folder: re.sub(r'_s\d+$', '', folder)
    members = {}
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        folder = os.path.basename(path)
        oofs = sorted(f for f in files if re.fullmatch(r'oof_fold_\d+\.csv', f))
        if oofs and family_of(folder) in families:
            members.setdefault(folder, []).extend(
                pd.read_csv(os.path.join(path, f), dtype={'StudyInstanceUID': str}) for f in oofs)
    assert members, f'no teacher OOF predictions under {root}'
    gold_ids, per_family = set(gold_ids), {}
    meta = dict(members={}, families={}, weights=dict(families))
    for folder, parts in sorted(members.items()):
        frame = pd.concat(parts, ignore_index=True)
        assert frame['StudyInstanceUID'].is_unique, f'{folder}: a study is out-of-fold twice'
        assert not gold_ids & set(frame['StudyInstanceUID']), f'{folder}: gold study in OOF'
        meta['members'][folder] = dict(studies=len(frame), folds=sorted(int(f) for f in frame['fold'].unique()))
        values = frame.set_index('StudyInstanceUID')[label_cols].astype(float)
        assert ((values >= 0) & (values <= 1)).all().all(), f'{folder}: OOF outside 0..1'
        per_family.setdefault(family_of(folder), []).append(values)
    ids = sorted(set().union(*[f.index for frames in per_family.values() for f in frames]))
    num = np.zeros((len(ids), len(label_cols))); den = np.zeros_like(num)
    n_members = np.zeros(len(ids), int)
    means = {}
    for family, frames in per_family.items():
        stack = np.stack([f.reindex(ids).to_numpy(float) for f in frames])
        counts = np.isfinite(stack).sum(axis=0)
        means[family] = (np.where(counts > 0, np.nansum(stack, axis=0) / np.maximum(counts, 1), np.nan), stack)
    mapped = [f for f in quantile_map if f in means]
    if mapped:
        ref_num = sum(np.nan_to_num(m) * families[f] for f, (m, _) in means.items() if f not in quantile_map)
        ref_den = sum(np.isfinite(m) * families[f] for f, (m, _) in means.items() if f not in quantile_map)
        assert np.max(ref_den) > 0, 'quantile_map needs at least one other teacher family'
        ref = np.where(ref_den > 0, ref_num / np.maximum(ref_den, 1e-9), np.nan)
        for family in mapped:
            m = means[family][0].copy()
            for j in range(m.shape[1]):
                ok, r = np.isfinite(m[:, j]), np.isfinite(ref[:, j])
                if ok.any() and r.any():
                    pct = (pd.Series(m[ok, j]).rank(method='average').to_numpy() - .5) / ok.sum()
                    m[ok, j] = np.quantile(ref[r, j], pct)
            means[family] = (m, means[family][1])
        meta['quantile_mapped'] = mapped
    for family, (mean, stack) in means.items():
        ok = np.isfinite(mean)
        num += np.where(ok, mean, 0) * families[family]
        den += ok * families[family]
        n_members += np.isfinite(stack).all(axis=2).sum(axis=0)
        meta['families'][family] = dict(members=len(frames), studies=int(ok.all(axis=1).sum()))
    teacher = pd.DataFrame(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), columns=label_cols)
    teacher.insert(0, 'StudyInstanceUID', ids)
    teacher['n_members'] = n_members
    return teacher, meta


def apply_teacher(frame, teacher_path, label_cols, weight):
    """Mix teacher probabilities into the pseudo rows: y = (1 - a) * report + a * teacher.
    weight is a constant a, or a gate (lo, hi): a = lo + (hi - lo) * u with u = 1 - |2 * report - 1|,
    so cells where the report readers agree confidently (consensus near 0 or 1) keep more of the
    report and split or uncertain cells lean on the image teacher. A pseudo cell without a report
    label takes the teacher value at cell weight hi. Gold rows are returned unchanged.
    Returns (frame, summary)."""
    lo, hi = (weight, weight) if np.isscalar(weight) else tuple(weight)
    assert 0 <= lo <= hi <= 1 and hi > 0
    teacher = pd.read_csv(teacher_path, dtype={'StudyInstanceUID': str})
    assert teacher['StudyInstanceUID'].is_unique, 'teacher repeats studies'
    t = teacher.set_index('StudyInstanceUID')[label_cols].apply(pd.to_numeric, errors='coerce')
    assert (t.isna() | ((t >= 0) & (t <= 1))).all().all(), 'teacher values outside 0..1'
    out = frame.copy()
    rows = np.flatnonzero(out['source'].eq('pseudo').to_numpy())
    wcols = [f'w_{c}' for c in label_cols]
    y = np.array(out.loc[out.index[rows], label_cols], dtype=float)
    w = np.array(out.loc[out.index[rows], wcols], dtype=float)
    tv = np.array(t.reindex(out['StudyInstanceUID'].iloc[rows]), dtype=float)
    has_t = np.isfinite(tv)
    both, only_t = has_t & np.isfinite(y), has_t & ~np.isfinite(y)
    a = lo + (hi - lo) * (1 - np.abs(2 * np.where(np.isfinite(y), y, .5) - 1))
    y[both] = (1 - a[both]) * y[both] + a[both] * tv[both]
    y[only_t] = tv[only_t]
    w[only_t] = hi
    out.loc[out.index[rows], label_cols] = np.where(np.isfinite(y), np.clip(y, .01, .99), np.nan).astype(np.float32)
    out.loc[out.index[rows], wcols] = w.astype(np.float32)
    summary = dict(teacher=teacher_path, weight=[lo, hi], mean_teacher_share=float(a[both].mean()) if both.any() else 0.,
                   studies=int(has_t.any(axis=1).sum()),
                   pseudo_studies=len(rows), mixed_cells=int(both.sum()), filled_cells=int(only_t.sum()),
                   sha256=hashlib.sha256(open(teacher_path, 'rb').read()).hexdigest())
    return out, summary


def report_groups(reports):
    # Byte-identical (up to case/punctuation) reports share targets: keep them in one fold.
    norm = reports.fillna('').astype(str).str.lower().map(lambda t: re.sub(r'\W+', '', t))
    return norm.map(lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]).to_numpy()


def build_targets(train, label_cols, tables, reader_weights=None, gold_weight=3.0):
    """train: competition train.csv. Returns (frame, report) where frame holds soft
    targets, per-cell weights (w_<col>), source and report group per study."""
    ids = train['StudyInstanceUID'].astype(str)
    readers = {name: read_reader(path, label_cols).reindex(ids) for name, path in tables.items()}
    assert readers, 'Attach at least one report-label table'
    weights = {name: (reader_weights or {}).get(name, 1.0) for name in readers}
    num = sum(r[label_cols].fillna(0).to_numpy() * weights[n] for n, r in readers.items())
    den = sum(r[label_cols].notna().to_numpy() * weights[n] for n, r in readers.items())
    with np.errstate(invalid='ignore', divide='ignore'):
        soft = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    soft = np.clip(soft, .01, .99)
    gold = train[label_cols].apply(pd.to_numeric, errors='coerce')
    is_gold = gold.notna().all(axis=1).to_numpy()
    y = soft.astype(np.float32)
    w = np.where(np.isfinite(soft), 1.0, 0.0).astype(np.float32)
    y[is_gold] = gold.to_numpy(np.float32)[is_gold]
    w[is_gold] = gold_weight
    frame = pd.DataFrame(y, columns=label_cols)
    frame.insert(0, 'StudyInstanceUID', ids.to_numpy())
    for k, c in enumerate(label_cols): frame[f'w_{c}'] = w[:, k]
    frame['source'] = np.where(is_gold, 'gold', 'pseudo')
    frame['report_group'] = report_groups(train['Report']) if 'Report' in train else ids.to_numpy()
    # Diagnostic only: each reader against the image-read gold labels. Not used for any choice.
    report = {}
    for name, r in {**readers, 'consensus': pd.DataFrame(soft, index=ids, columns=label_cols)}.items():
        aucs = []
        # A reader that never labelled the gold studies has no gold AUC (not 0.5).
        gold_rows = int(r.loc[ids[is_gold], label_cols].notna().any(axis=1).sum())
        for c in label_cols:
            truth, score = gold.loc[is_gold, c].to_numpy(), r.loc[ids[is_gold], c].fillna(.5).to_numpy()
            aucs.append(roc_auc_score(truth, score) if len(np.unique(truth)) == 2 and gold_rows else np.nan)
        finite = [a for a in aucs if np.isfinite(a)]
        report[name] = dict(gold_macro_auc=float(np.mean(finite)) if finite else float('nan'),
                            gold_rows=gold_rows, weight=weights.get(name, 0),
                            coverage=float(np.isfinite(r.to_numpy()).mean()))
    return frame, report
