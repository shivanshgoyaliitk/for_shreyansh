"""Check the knee-2c-v4 joint anchor on real studies before building the cache (CPU, competition attached).

For a sample of training studies: builds the v4 manifest (anchors from header geometry), reports how
often each local slot's crop is centred on the anchor (versus the centred fallback) and how far the
anchor projects from the image centre (mm), and saves montages of the middle window of the full and
local sagittal / coronal slots (anchor_montage_*.png) for visual inspection. Reads no labels.
Expects the preprocessing source (load_series, build_manifest, slot_kwargs, ...) defined above.
"""
import os, glob, json
import numpy as np
import pandas as pd
import cv2
import pydicom

N_STUDIES = int(os.environ.get('ANCHOR_N', 60))
DATA_DIR = next(d for d in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                            '/kaggle/input/rsna-knee-abnormality-detection') if os.path.isdir(d))
WORK = '/kaggle/working'
spec = recipe('knee-2c-v4')
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
series = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))
series['series_dir'] = [os.path.join(DATA_DIR, 'train_series', str(st), str(se))
                        for st, se in zip(series['StudyInstanceUID'], series['SeriesInstanceUID'])]
studies = train[['StudyInstanceUID']].sample(N_STUDIES, random_state=11).reset_index(drop=True)
manifest = build_manifest(studies, series[series['StudyInstanceUID'].isin(studies['StudyInstanceUID'])], spec)
names = [s[0] for s in spec['slots']]
print('anchors found:', int(manifest['anchor_x'].notna().sum()), 'of', len(manifest))

rows = []
for _, row in manifest.iterrows():
    anchor = row_anchor(row)
    for name in ('SAG_LOCAL', 'COR_LOCAL'):
        d = row[f'dir_{names.index(name)}']
        if not d or anchor is None:
            rows.append(dict(study=row['StudyInstanceUID'], slot=name, has_series=bool(d), used=False)); continue
        files = sorted(glob.glob(os.path.join(d, '*.dcm')))
        ds = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
        shape = (int(ds.Rows), int(ds.Columns))
        flip = name == 'COR_LOCAL' and not _row_points_left(getattr(ds, 'ImageOrientationPatient', None))
        at = project_anchor(ds, anchor, shape, flip)
        py, px = (float(v) for v in ds.PixelSpacing)
        off = None if at is None else float(np.hypot((at[0] - shape[0] / 2) * py, (at[1] - shape[1] / 2) * px))
        rows.append(dict(study=row['StudyInstanceUID'], slot=name, has_series=True, used=at is not None,
                         offset_mm=off, fov_mm=shape[0] * py))
report = pd.DataFrame(rows)
print(report.groupby('slot')[['has_series', 'used']].mean().round(3).to_string())
print('anchor offset from the image centre (mm):')
print(report.dropna(subset=['offset_mm']).groupby('slot')['offset_mm'].describe().round(1).to_string())
report.to_csv(os.path.join(WORK, 'anchor_report.csv'), index=False)

# Montages: middle window (centre slice) of SAG_FLUID_FS | SAG_LOCAL | COR_FLUID_FS | COR_LOCAL.
show = [names.index(n) for n in ('SAG_FLUID_FS', 'SAG_LOCAL', 'COR_FLUID_FS', 'COR_LOCAL')]
tiles = []
for _, row in manifest.head(16).iterrows():
    anchor, strip = row_anchor(row), []
    for slot in show:
        d = row[f'dir_{slot}']
        img = np.zeros((spec['img'], spec['img']), np.uint8)
        if d:
            try:
                vol = load_series(d, spec['slots'][slot][1], row['study_side'], **slot_kwargs(spec, slot, anchor))
                img = vol[len(vol) // 2]
            except Exception as exc:
                print('decode failed', row['StudyInstanceUID'], slot, type(exc).__name__)
        strip.append(img)
    tiles.append(np.concatenate(strip, 1))
for k in range(0, len(tiles), 8):
    cv2.imwrite(os.path.join(WORK, f'anchor_montage_{k // 8}.png'), np.concatenate(tiles[k:k + 8], 0))
print('ANCHOR CHECK DONE')
