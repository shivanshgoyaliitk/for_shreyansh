"""How reliable is the coordinate laterality fallback (CPU, competition attached)?

series_side() falls back to the sign of the field centre's x (LPS, +x = patient left) of the FIRST file of a series
when no laterality tag exists; about half the training studies take their side from that fallback. For a sample of
training studies this computes, per series, the tag side, the current first-file coordinate side and the side from
the stack centre (mean over all slices, robust to sagittal edge slices), votes each per study as build_manifest does,
and reports how often the coordinate methods agree with the tags (where tags exist) and with each other. Reads no labels.
Expects the preprocessing source (read_header, annotate, series_side, stack_centre, ...) defined above.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

N_STUDIES = int(os.environ.get('LATERALITY_N', 600))
DATA_DIR = next(d for d in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                            '/kaggle/input/rsna-knee-abnormality-detection') if os.path.isdir(d))
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
series = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))
studies = train[['StudyInstanceUID']].drop_duplicates().sample(N_STUDIES, random_state=5)
series = series[series['StudyInstanceUID'].isin(studies['StudyInstanceUID'])].reset_index(drop=True)
series['series_dir'] = [os.path.join(DATA_DIR, 'train_series', str(st), str(se))
                        for st, se in zip(series['StudyInstanceUID'], series['SeriesInstanceUID'])]
with ThreadPoolExecutor(max_workers=8) as pool:
    headers = pd.DataFrame(list(pool.map(read_header, series['series_dir'])))
    centres = list(pool.map(stack_centre, series['series_dir']))
for col in headers:
    series[col] = headers[col]
series = annotate(series)


def tag_side(row):
    for tag in ('ImageLaterality', 'Laterality'):
        side = str(row.get(tag, '')).strip().upper()
        if side in ('L', 'R'):
            return side
    return None


untagged = series.copy()
untagged['ImageLaterality'] = untagged['Laterality'] = ''
series['tag'] = [tag_side(r) for _, r in series.iterrows()]
series['first_file'] = [series_side(r)[0] for _, r in untagged.iterrows()]
series['stack'] = [None if c is None or abs(c[0]) <= 5 else ('L' if c[0] > 0 else 'R') for c in centres]
series['stack_x'] = [np.nan if c is None else c[0] for c in centres]


def majority(values):
    counts = pd.Series([v for v in values if v]).value_counts()
    if not len(counts) or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
        return None
    return counts.index[0]


rows = []
for uid, g in series.groupby('StudyInstanceUID'):
    sag = g[g['Anatomical_Plane'].astype(str).eq('Sagittal')]
    rows.append(dict(StudyInstanceUID=uid, n_series=len(g), tag=majority(g['tag']),
                     first_file=majority(g['first_file']), stack=majority(g['stack']),
                     first_file_nonsag=majority(g.loc[~g.index.isin(sag.index), 'first_file']),
                     sag_x=float(np.nanmedian(sag['stack_x'])) if len(sag) else np.nan))
study = pd.DataFrame(rows)
tagged = study[study['tag'].notna()]
report = {}
for method in ('first_file', 'stack', 'first_file_nonsag'):
    t = tagged[tagged[method].notna()]
    report[method] = dict(tagged_studies=len(tagged), decided=len(t), agree_with_tag=float((t[method] == t['tag']).mean()),
                          undecided=int(tagged[method].isna().sum()))
untag = study[study['tag'].isna()]
both = untag[untag['first_file'].notna() & untag['stack'].notna()]
report['untagged'] = dict(studies=len(untag), first_file_vs_stack_disagree=int((both['first_file'] != both['stack']).sum()),
                          compared=len(both), first_file_undecided=int(untag['first_file'].isna().sum()),
                          stack_undecided=int(untag['stack'].isna().sum()))
print(pd.DataFrame(report).T.to_string())
print('|sagittal stack x| quantiles (mm):', np.nanquantile(study['sag_x'].abs(), [.05, .25, .5, .75, .95]).round(1).tolist())
study.to_csv('/kaggle/working/laterality_check.csv', index=False)
