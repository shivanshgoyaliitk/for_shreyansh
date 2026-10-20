"""Shared step-2c preprocessing. Embedded verbatim in cache and inference notebooks."""
import os, glob, re, json, hashlib
import numpy as np
import pandas as pd
import pydicom
import cv2
from concurrent.futures import ThreadPoolExecutor

# Each recipe fully determines the cached pixels. v2 stays decodable so checkpoints
# trained on it keep their exact inputs; new caches use the current version.
RECIPES = {
    "knee-2c-v2": dict(
        img=336, n_slices=12, crop_mm=130.0, band=(0.2, 0.8), min_slices=8, canon="L",
        slot_rule="fluid_fallback",
        slots=[("SAG_FLUID", "Sagittal", True), ("COR_FLUID", "Coronal", True),
               ("AX_FLUID", "Axial", True)]),
    # Six header-typed slots, as in the public DINOv2 members: three fat-suppressed
    # fluid-sensitive planes plus non-suppressed sagittal fluid and two T1 slots.
    # Typing is strict; an absent sequence stays absent instead of borrowing another.
    # Each slot stores 8 windows of 3 ADJACENT slices whose centres span 6-94% of the
    # stack (the public 2.5D reader layout), at 0.54 mm/px. 37 GB: built as two shards.
    "knee-2c-v3": dict(
        img=240, n_slices=24, windows=8, crop_mm=130.0, band=(0.06, 0.94), min_slices=8,
        canon="L", slot_rule="strict", n_shards=2,
        slots=[("SAG_FLUID_FS", "Sagittal", True, True),
               ("COR_FLUID_FS", "Coronal", True, True),
               ("AX_FLUID_FS", "Axial", True, True),
               ("SAG_FLUID_NOFS", "Sagittal", True, False),
               ("COR_T1", "Coronal", False, False),
               ("SAG_T1", "Sagittal", False, False)]),
    # v3 plus two LOCAL slots (2026-10-11): the sagittal and coronal fluid-sensitive series again,
    # cropped to 90 mm (0.375 mm/px instead of 0.54) around the knee-joint centre. The centre is
    # header geometry only: left-right from the middle of the sagittal stack, anterior-posterior
    # from the middle of the coronal stack, superior-inferior from both field centres. The six
    # v3 slots are pixel-identical to v3. 49 GB: built as three shards.
    "knee-2c-v4": dict(
        img=240, n_slices=24, windows=8, crop_mm=130.0, band=(0.06, 0.94), min_slices=8,
        canon="L", slot_rule="strict", n_shards=3, local_mm=90.0,
        slots=[("SAG_FLUID_FS", "Sagittal", True, True),
               ("COR_FLUID_FS", "Coronal", True, True),
               ("AX_FLUID_FS", "Axial", True, True),
               ("SAG_FLUID_NOFS", "Sagittal", True, False),
               ("COR_T1", "Coronal", False, False),
               ("SAG_T1", "Sagittal", False, False),
               ("SAG_LOCAL", "Sagittal", True, True),
               ("COR_LOCAL", "Coronal", True, True)],
        local={"SAG_LOCAL": ["SAG_FLUID_FS", "SAG_FLUID_NOFS"], "COR_LOCAL": ["COR_FLUID_FS", "COR_T1"]}),
    # v4 at higher resolution and slice density (2026-10-12): 288 px (0.45 mm/px full field, 0.31 mm/px
    # local) and 10 windows of 3 adjacent slices per series (30 slices, so less of a 30-40 slice stack is
    # skipped). Same slots, crops and anchor. 88 GB: built as five shards.
    "knee-2c-v5": dict(
        img=288, n_slices=30, windows=10, crop_mm=130.0, band=(0.06, 0.94), min_slices=8,
        canon="L", slot_rule="strict", n_shards=5, local_mm=90.0,
        slots=[("SAG_FLUID_FS", "Sagittal", True, True),
               ("COR_FLUID_FS", "Coronal", True, True),
               ("AX_FLUID_FS", "Axial", True, True),
               ("SAG_FLUID_NOFS", "Sagittal", True, False),
               ("COR_T1", "Coronal", False, False),
               ("SAG_T1", "Sagittal", False, False),
               ("SAG_LOCAL", "Sagittal", True, True),
               ("COR_LOCAL", "Coronal", True, True)],
        local={"SAG_LOCAL": ["SAG_FLUID_FS", "SAG_FLUID_NOFS"], "COR_LOCAL": ["COR_FLUID_FS", "COR_T1"]}),
}
PREPROCESS_VERSION = "knee-2c-v5"
_R = RECIPES[PREPROCESS_VERSION]
IMG, N_SLICES, CROP_MM = _R["img"], _R["n_slices"], _R["crop_mm"]
SLICE_BAND, MIN_SLICES, CANON_SIDE = _R["band"], _R["min_slices"], _R["canon"]
SLOTS = _R["slots"]
HDR = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
       "RepetitionTime", "EchoTime", "Laterality", "ImageLaterality",
       "PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient", "Rows", "Columns"]


def read_header(series_dir):
    rec = {t: None for t in HDR}
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    rec["n_slices"] = len(files)
    for f in files[:3]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        for t in HDR:
            v = getattr(ds, t, None)
            if v is None:
                continue
            if t == "PixelSpacing":
                try: v = "|".join(str(float(x)) for x in v)
                except Exception: v = None
            elif t in ("ImageOrientationPatient", "ImagePositionPatient"):
                try: v = "|".join(str(float(x)) for x in v)
                except Exception: v = None
            elif t == "ScanOptions":
                v = "|".join(map(str, v)) if not isinstance(v, str) else v
            elif not isinstance(v, (int, float, str)):
                v = str(v)
            rec[t] = v
        break
    return rec


_SEP    = re.compile(r"[^a-z0-9]+")
_FATSAT = re.compile(r"\b(?:fs|fatsat|fatsuppressed|spair|spir|stir|dixon|spectral)\b")
_T1_RX, _T2_RX = re.compile(r"\bt1\b"), re.compile(r"\bt2\b")
_PD_RX  = re.compile(r"\b(?:pd|pdw|proton)\b")
FATSAT_OPTS = {"FS", "FSAT", "SFS", "SP", "SPIR", "SPAIR", "FATSAT"}
# v3 slot typing. Siemens "SP" is spatial presaturation, not fat suppression, so the
# option set is exact; descriptions add water excitation and TIRM.
_FATSAT_V3 = re.compile(r"\b(?:fs|fatsat|fatsuppressed|fatsup|spair|spir|stir|tirm|dixon|"
                        r"spectral|we)\b|fat sat|fat sup|water excit")
FATSAT_OPTS_V3 = {"FS", "FSAT", "FAT_SAT", "FATSAT", "SPIR", "SPAIR"}


def annotate(df):
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    df["fatsat_hdr"] = desc.str.contains(_FATSAT) | opts.apply(
        lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_V3) | opts.apply(
        lambda ts: any(t.strip() in FATSAT_OPTS_V3 for t in ts))
    tr, te = pd.to_numeric(df["RepetitionTime"], errors="coerce"), pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)
    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                    np.where(t2 & ~pdw, "T2", np.where(pdw, "PD", np.where(gre, "GRE",
                     np.where(tr < 800, "T1", np.where(te > 60, "T2",
                      np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid_hdr"] = np.isin(df["weight"], ["PD", "T2"])
    return df


def _sort_key(ds):
    # Canonical through-plane direction: the normal's sign depends on how the scanner
    # wrote ImageOrientationPatient, so force its dominant component positive. For a
    # sagittal stack that makes slices always run in +x = toward the patient's left.
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        try:
            n = np.cross(np.array(iop[:3], float), np.array(iop[3:6], float))
            n = n * np.sign(n[np.argmax(np.abs(n))] or 1.0)
            return float(np.dot(np.array(ipp, float), n))
        except Exception:
            pass
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _row_points_left(iop):
    # +column direction in patient coords. True if increasing column goes toward
    # the patient's LEFT (+x). Used to make image handedness consistent first.
    if iop is None or len(iop) < 3:
        return True
    return float(iop[0]) >= 0


def slice_indices(n, n_slices, band, windows=None):
    # Positions read from an ordered stack of n files.
    if windows:
        # Window centres spread over the band; each window is the centre and its two
        # neighbours, so every 3-slice input keeps true through-plane adjacency.
        a, b = max(1.0, band[0] * (n - 1)), min(n - 2.0, band[1] * (n - 1))
        if b < a:
            a = b = (n - 1) / 2
        centres = np.clip(np.round(np.linspace(a, b, windows)).astype(int), 1, max(1, n - 2))
        return np.clip(np.stack([centres - 1, centres, centres + 1], 1).reshape(-1), 0, n - 1)
    lo_i, hi_i = int(band[0] * (n - 1)), int(band[1] * (n - 1))
    return (np.linspace(lo_i, hi_i, n_slices).round().astype(int) if hi_i > lo_i
            else np.full(n_slices, n // 2))


ANCHOR_SOURCES = {'Sagittal': ('SAG_FLUID_FS', 'SAG_FLUID_NOFS', 'SAG_T1'), 'Coronal': ('COR_FLUID_FS', 'COR_T1')}
ANCHOR_WINDOW = (0.15, 0.85)   # a projected centre outside this share of rows/columns is not trusted


def stack_centre(series_dir):
    """Patient-coordinate (LPS) centre of a stack: the mean of its slices' field centres."""
    centres = []
    for f in sorted(glob.glob(os.path.join(series_dir, "*.dcm"))):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            ipp = np.asarray([float(x) for x in ds.ImagePositionPatient])
            iop = np.asarray([float(x) for x in ds.ImageOrientationPatient])
            py, px = (float(x) for x in ds.PixelSpacing)
            rows, cols = int(ds.Rows), int(ds.Columns)
        except Exception:
            continue
        centres.append(ipp + iop[:3] * px * (cols - 1) / 2 + iop[3:] * py * (rows - 1) / 2)
    return np.mean(centres, 0) if centres else None


def joint_anchor(sag_centre, cor_centre):
    """Knee-joint centre (LPS) from header geometry: left-right from the sagittal stack, anterior-
    posterior from the coronal stack, superior-inferior from both field centres. None if neither."""
    if sag_centre is not None and cor_centre is not None:
        return np.array([sag_centre[0], cor_centre[1], (sag_centre[2] + cor_centre[2]) / 2])
    return sag_centre if sag_centre is not None else cor_centre


def project_anchor(ds, anchor, shape, flip_cols):
    """(row, col) of an LPS point in this slice's pixel grid (after the column flip), or None when
    the geometry is missing or the point falls outside ANCHOR_WINDOW of the image."""
    try:
        ipp = np.asarray([float(x) for x in ds.ImagePositionPatient])
        iop = np.asarray([float(x) for x in ds.ImageOrientationPatient])
        py, px = (float(x) for x in ds.PixelSpacing)
    except Exception:
        return None
    d = np.asarray(anchor, float) - ipp
    row, col = float(d @ iop[3:]) / py, float(d @ iop[:3]) / px
    if flip_cols:
        col = shape[1] - 1 - col
    lo, hi = ANCHOR_WINDOW
    if not (lo * shape[0] <= row <= hi * shape[0] and lo * shape[1] <= col <= hi * shape[1]):
        return None
    return int(round(row)), int(round(col))


def square_crop(a, cy, cx, hy, hx):
    """Crop [cy-hy, cy+hy) x [cx-hx, cx+hx), zero-padded where it leaves the image (no distortion)."""
    out = np.zeros((2 * hy, 2 * hx), a.dtype)
    y0, x0 = cy - hy, cx - hx
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(a.shape[0], cy + hy), min(a.shape[1], cx + hx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def load_series(series_dir, plane, side, n_slices=N_SLICES, img=IMG,
                band=SLICE_BAND, crop_mm=CROP_MM, canon=CANON_SIDE, windows=None, center_lps=None):
    # Callers pass slot_kwargs(recipe, slot, anchor) so a cache or inference run reproduces its recipe.
    # center_lps (local slots): crop a zero-padded square around that patient-coordinate point when it
    # projects inside the image; otherwise, and for every regular slot, the centred crop below.
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(series_dir)
    keyed = []
    for f in files:
        try:
            keyed.append((_sort_key(pydicom.dcmread(f, stop_before_pixels=True)), f))
        except Exception:
            continue
    if not keyed:
        raise RuntimeError("no readable headers")
    keyed.sort(key=lambda t: t[0])
    ordered = [f for _, f in keyed]

    # Sagittal: L-R is the through-plane axis, so a right knee is the stack reversed.
    # Reverse the FILE ORDER here, before the slice band is applied -- reversing the
    # finished volume instead would sample anatomical positions shifted by one slice
    # relative to a left knee.
    if side != canon and plane == "Sagittal":
        ordered = ordered[::-1]

    n = len(ordered)
    idx = slice_indices(n, n_slices, band, windows)
    if windows: assert len(idx) == n_slices == 3 * windows

    out = np.zeros((n_slices, img, img), dtype=np.float32)
    ok, flip_cols = 0, False
    decoded = np.zeros(n_slices, dtype=bool)
    for j, i in enumerate(idx):
        try:
            ds = pydicom.dcmread(ordered[int(i)])
            a = ds.pixel_array.astype(np.float32)
            m = float(getattr(ds, "RescaleSlope", 1) or 1)
            b = float(getattr(ds, "RescaleIntercept", 0) or 0)
            if (m, b) != (1.0, 0.0):
                a = a * m + b
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = a.max() - a
            if a.ndim == 3:
                a = a.mean(axis=-1)

            iop = getattr(ds, "ImageOrientationPatient", None)
            flip_cols = plane in ("Coronal", "Axial") and not _row_points_left(iop)
            if flip_cols:
                a = a[:, ::-1]

            # physical crop, centred, before the resize
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None and len(ps) == 2:
                py, px = map(float, ps)
                if np.isfinite([py, px]).all() and min(py, px) > 0:
                    hy, hx = max(1, round(crop_mm / py / 2)), max(1, round(crop_mm / px / 2))
                    cy, cx = a.shape[0] // 2, a.shape[1] // 2
                    at = project_anchor(ds, center_lps, a.shape, flip_cols) if center_lps is not None else None
                    if at is not None:
                        a = square_crop(a, at[0], at[1], hy, hx)
                    else:
                        a = a[max(0, cy-hy):min(a.shape[0], cy+hy),
                              max(0, cx-hx):min(a.shape[1], cx+hx)]

            out[j] = cv2.resize(np.ascontiguousarray(a), (img, img),
                                interpolation=cv2.INTER_AREA)
            ok += 1
            decoded[j] = True
        except Exception:
            continue
    if ok == 0:
        raise RuntimeError("no decodable slices")

    # Replace failed slices by nearest decoded neighbour before normalization.
    good = np.flatnonzero(decoded)
    for j in np.flatnonzero(~decoded):
        out[j] = out[good[np.argmin(abs(good-j))]]
    lo_v, hi_v = np.percentile(out, [1, 99])
    if hi_v <= lo_v:
        lo_v, hi_v = float(out.min()), float(out.max())
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    out = np.clip((out - lo_v) / (hi_v - lo_v) * 255.0, 0, 255).astype(np.uint8)

    # ---- laterality normalisation, in-plane half ----------------------------
    # +x is the patient's left and columns now point left, so for coronal/axial a
    # right knee is the column-mirror of a left knee. (The sagittal half was done
    # above, on the file order.)
    if side != canon and plane != "Sagittal":
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)

def _vec(value):
    try:
        return np.asarray([float(x) for x in value.split('|')], dtype=float)
    except (AttributeError, ValueError, TypeError):
        return None


def series_side(row):
    for tag in ('ImageLaterality', 'Laterality'):
        side = str(row.get(tag, '')).strip().upper()
        if side in ('L', 'R'):
            return side, 'tag'
    ipp, iop, ps = (_vec(row.get(k)) for k in
                    ('ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'))
    try:
        rows, cols = float(row['Rows']), float(row['Columns'])
        if ipp is not None and iop is not None and ps is not None:
            center = ipp + iop[:3] * ps[1] * (cols-1)/2 + iop[3:] * ps[0] * (rows-1)/2
            if np.isfinite(center).all() and abs(center[0]) > 5:
                return ('L' if center[0] > 0 else 'R'), 'coords'
    except (KeyError, ValueError, TypeError):
        pass
    return None, 'unknown'


def vote(group):
    # Explicit tags outrank any number of coordinate guesses. Ties stay unknown.
    tagged = group.loc[group['side_src'].eq('tag'), 'side'].dropna()
    values = tagged if len(tagged) else group['side'].dropna()
    counts = values.value_counts()
    if not len(counts) or (len(counts)>1 and counts.iloc[0] == counts.iloc[1]):
        return CANON_SIDE, 'default'
    return counts.index[0], 'tag' if len(tagged) else 'coords'


def recipe(version=PREPROCESS_VERSION):
    if version not in RECIPES:
        raise ValueError(f'Unknown preprocessing version {version!r}')
    return RECIPES[version]


def series_kwargs(spec):
    # load_series arguments that reproduce one recipe's pixels exactly.
    return dict(n_slices=spec['n_slices'], img=spec['img'], band=tuple(spec['band']),
                crop_mm=spec['crop_mm'], canon=spec['canon'], windows=spec.get('windows'))


def slot_kwargs(spec, slot, anchor=None):
    # Per-slot load_series arguments: local slots crop local_mm around the study's joint anchor.
    kw = series_kwargs(spec)
    if spec['slots'][slot][0] in spec.get('local', {}):
        kw['crop_mm'] = spec['local_mm']
        if anchor is not None and np.isfinite(np.asarray(anchor, float)).all():
            kw['center_lps'] = np.asarray(anchor, float)
    return kw


def row_anchor(row):
    # The manifest's joint anchor for one study, or None (absent columns or unknown geometry).
    try:
        anchor = np.array([float(row['anchor_x']), float(row['anchor_y']), float(row['anchor_z'])])
    except (KeyError, TypeError, ValueError):
        return None
    return anchor if np.isfinite(anchor).all() else None


def choose(group, spec=None):
    spec = recipe() if spec is None else spec
    picked = {}
    local = spec.get('local', {})
    for slot in spec['slots']:
        name, plane, fluid = slot[:3]
        if name in local:
            continue
        sel = group['plane'].eq(plane) & group['fluid'].eq(fluid)
        if spec['slot_rule'] == 'strict':
            sel &= group['fatsat'].eq(slot[3])
            if not fluid:
                # T1-typed series before any other non-fluid (GRE/unknown) series.
                sel &= group['weight'].isin(['T1', 'GRE', 'UNK'])
            cand = group[sel & (group['n_slices'] > 0)]
            if len(cand):
                rank = cand.assign(_t1=cand['weight'].eq('T1'),
                                   _enough=cand['n_slices'] >= spec['min_slices'])
                picked[name] = rank.sort_values(['_enough', '_t1', 'n_slices', 'series_dir'],
                                                ascending=[False, False, False, True]).iloc[0]
            continue
        cand = group[sel & (group['n_slices'] >= spec['min_slices'])]
        if not len(cand): cand = group[sel & (group['n_slices'] > 0)]
        if not len(cand): cand = group[group['plane'].eq(plane) & (group['n_slices'] >= spec['min_slices'])]
        if len(cand):
            picked[name] = cand.sort_values(['n_slices', 'series_dir'], ascending=[False, True]).iloc[0]
    # Local slots re-read the first available source slot's series with the local crop.
    for name, sources in local.items():
        source = next((picked[s] for s in sources if s in picked), None)
        if source is not None:
            picked[name] = source
    return picked


def build_manifest(studies, series, spec=None):
    spec = recipe() if spec is None else spec
    slots = spec['slots']
    if series.empty:
        result = studies[['StudyInstanceUID']].copy()
        result['study_side'], result['side_src'] = spec['canon'], 'default'
        for i in range(len(slots)): result[f'dir_{i}'] = ''
        return result
    series = series.copy().reset_index(drop=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        headers = pd.DataFrame(list(pool.map(read_header, series['series_dir'])))
    # Header columns may already exist in an input CSV.
    for col in headers: series[col] = headers[col]
    series = annotate(series)
    provided = pd.to_numeric(series['Fluid_Sensitive'], errors='coerce').fillna(0).astype(bool)
    series['fluid'] = np.where(series['weight'].eq('UNK'), provided, series['fluid_hdr'])
    series['plane'] = series['Anatomical_Plane'].astype(str)
    sides = [series_side(row) for _, row in series.iterrows()]
    series['side'] = [s[0] for s in sides]
    series['side_src'] = [s[1] for s in sides]
    records = []
    for uid, group in series.groupby('StudyInstanceUID', sort=False):
        side, src = vote(group)
        rec = dict(StudyInstanceUID=uid, study_side=side, side_src=src)
        picked = choose(group, spec)
        for i, slot in enumerate(slots):
            row = picked.get(slot[0])
            rec[f'dir_{i}'] = row['series_dir'] if row is not None else ''
        if spec.get('local'):
            rec['_anchor_dirs'] = [next((picked[s]['series_dir'] for s in ANCHOR_SOURCES[plane] if s in picked), None)
                                   for plane in ('Sagittal', 'Coronal')]
        records.append(rec)
    anchor_cols = ['anchor_x', 'anchor_y', 'anchor_z'] if spec.get('local') else []
    if anchor_cols:
        # Joint anchor per study from header geometry (every slice header of one sagittal and one coronal stack).
        dirs = sorted({d for rec in records for d in rec['_anchor_dirs'] if d})
        with ThreadPoolExecutor(max_workers=8) as pool:
            centre_of = dict(zip(dirs, pool.map(stack_centre, dirs)))
        for rec in records:
            sag, cor = (centre_of.get(d) if d else None for d in rec.pop('_anchor_dirs'))
            anchor = joint_anchor(sag, cor)
            rec.update(zip(anchor_cols, anchor if anchor is not None else [np.nan] * 3))
    columns = ['StudyInstanceUID', 'study_side', 'side_src', *[f'dir_{i}' for i in range(len(slots))], *anchor_cols]
    result = studies[['StudyInstanceUID']].merge(pd.DataFrame(records, columns=columns),
                                               on='StudyInstanceUID', how='left', validate='one_to_one')
    result['study_side'] = result['study_side'].fillna(spec['canon'])
    result['side_src'] = result['side_src'].fillna('default')
    for i in range(len(slots)): result[f'dir_{i}'] = result[f'dir_{i}'].fillna('')
    print('Laterality sources:', result['side_src'].value_counts().to_dict())
    return result


def preprocessing_meta(version=PREPROCESS_VERSION):
    spec = recipe(version)
    meta = dict(version=version, slots=[s[0] for s in spec['slots']],
                planes=[s[1] for s in spec['slots']], n_slices=spec['n_slices'], img=spec['img'],
                crop_mm=spec['crop_mm'], slice_band=list(spec['band']), canon_side=spec['canon'],
                min_slices=spec['min_slices'])
    if spec['slot_rule'] != 'fluid_fallback':
        # v2 metadata is kept byte-identical so its checkpoints still validate.
        meta.update(slot_rule=spec['slot_rule'],
                    slot_types=[[bool(s[2]), bool(s[3])] for s in spec['slots']])
    if spec.get('windows'):
        meta.update(windows=spec['windows'], window_size=3)
    if spec.get('local'):
        meta.update(local={k: list(v) for k, v in spec['local'].items()}, local_mm=spec['local_mm'],
                    anchor='header geometry: LR sagittal stack, AP coronal stack, SI both field centres')
    return meta


def window_positions(meta):
    # Relative stack position (0..1) of each window centre, for position embeddings.
    if meta.get('windows'):
        return np.linspace(meta['slice_band'][0], meta['slice_band'][1], meta['windows']).tolist()
    return np.linspace(meta['slice_band'][0], meta['slice_band'][1], meta['n_slices']).tolist()


def cache_fingerprint(manifest, meta):
    content = manifest.to_json(orient='split') + json.dumps(meta, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
