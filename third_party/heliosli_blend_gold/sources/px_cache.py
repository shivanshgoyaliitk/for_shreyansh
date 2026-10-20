"""Per-slice pixel cache on the fast disk, for training compact image models without re-reading F:.

Data preparation only: no model, no optimizer.  Reads F:/rsnaknee/cache_native (read-only) and writes
ONLY under --out:
  px.u8.npy    [N, S, S] uint8   one row per (series, view, slice), slices in physical order
  meta.f32.npy [N, 12]
  index.json   study -> [start, n_rows, n_series], per-series blocks, config, progress
Each row is exactly one channel of the mainline's validation-mode window (same crop geometry, same
INTER_AREA resize, same series_companded intensity mapping), at S pixels instead of 384.
Views: full field + 100 mm local; coronal uses the mainline's bilateral views.
"""
import sys, os, json, time, hashlib, argparse
sys.dont_write_bytecode = True
CODE_ROOT = os.environ.get("RSNA_CODE_ROOT", "D:/rsnaknee")      # mainline checkout (Linux box: /root/autodl-tmp/rsnaknee)
sys.path.insert(0, CODE_ROOT)
import numpy as np
import torch

LOCAL_MM = 100.
THIN_MM, SLAB_MM = 1.75, 3.5       # stacks thinner than THIN_MM are averaged into SLAB_MM slabs (see StudySlices.build)
POLICY = "series_companded"
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid",
             "fatsat", "slice_order", "n_slices", "real_series_idx"]
UIDK = "StudyInstanceUID"


def study_json(cache, uid):
    return os.path.join(cache, "studies", hashlib.sha256(uid.encode()).hexdigest()[:24] + ".json")


# View sets.  v1: full field + 100 mm local at the field centre (coronal: bilateral locals).  v2 adds, for sagittal and axial
# series, a 100 mm "patellofemoral" local (view id 4; ids 2/3 are the coronal bilateral locals) centred anterior-superior: in the canonical cache anterior is +column
# for sagittal and -row for axial, and a centre-of-field local cuts the patella at its edge in every study inspected
# (10-06; PF OA is the class with the largest report-vs-image gap on the 58 official studies, 0.92 vs 0.82).
# Centre tuples are (row fraction, column fraction) as view_geometry expects.
PF_CENTER = {"Sagittal": (.38, .60), "Axial": (.38, .50)}
# v3 (high-resolution set, meant for a 384 px cache): NO full-field view; sagittal/axial = centre local (1) + patellofemoral
# local (4) + posterior local (5: popliteal fossa / Baker's cyst / PCL; posterior = -column for sagittal, +row for axial);
# coronal = the two bilateral locals only.  Three overlapping 100 mm locals cover what the 1.8 px/mm full view showed.
POST_CENTER = {"Sagittal": (.55, .35), "Axial": (.62, .50)}


def knee_anchor(vol, plane):
    """(row_frac, col_frac) of the knee in the series' in-plane frame from the foreground of the 3 middle slices.
    Falls back to the field centre when the leg fills the field or nothing is found."""
    import cv2
    m = len(vol) // 2
    mid = np.asarray(vol[max(0, m - 1):m + 2]).astype(np.float32).mean(0)
    H, W = mid.shape
    hi = float(np.percentile(mid, 99.5))
    if hi <= 0:
        return (.5, .5), "empty"
    fg = (mid > 0.08 * hi).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return (.5, .5), "no_fg"
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = [int(v) for v in stats[k, :4]]
    if w >= 0.97 * W and h >= 0.97 * H:
        return (.5, .5), "fills_field"
    comp = (lab == k)
    if plane in ("Sagittal", "Coronal"):
        width = comp[y:y + h].sum(1).astype(np.float32)
        if len(width) >= 15:
            kern = np.ones(15, np.float32) / 15
            width = np.convolve(width, kern, mode="same")
        row = (y + int(np.argmax(width))) / max(1, H - 1)
    else:
        row = (y + h / 2.) / max(1, H - 1)
    col = (x + w / 2.) / max(1, W - 1)
    return (float(row), float(col)), "ok"


def anchored(anchor, spacing_mm, shape, d_row_mm, d_col_mm):
    """centre = anchor shifted by physical offsets (rows: +down, cols: +right), as (row_frac, col_frac)."""
    sy, sx = float(spacing_mm[0]), float(spacing_mm[1])
    H, W = shape
    return (min(1., max(0., anchor[0] + d_row_mm / (sy * max(1, H - 1)))),
            min(1., max(0., anchor[1] + d_col_mm / (sx * max(1, W - 1)))))


def series_views(plane, native_shape, spacings, size, views="v1", anchor=None, spacing_mm=None, patella=None):
    if views in ("v3k", "v4"):
        # knee-centred set from the anchor bank (SKM-TEA joint centre projected through the geometry bank):
        # v4 adds the full field of view (id 0) of every series in front of the locals, so nothing is out of sight.
        full = [(0, None, (.5, .5))] if views == "v4" else []
        # centre local (1) at the joint; PF local (4) at the patella anchor when the segmentation found it, else
        # anterior-superior of the joint; posterior local (5); coronal medial (2) / lateral (3) 30 mm either side.
        # Anterior = +col (sagittal), -row (axial) in the canonical cache.
        shp = tuple(native_shape[-2:])
        a = anchor if anchor is not None else (.5, .5)
        sp = spacing_mm if spacing_mm is not None else (1., 1.)
        if plane == "Sagittal":
            pf = patella if patella is not None else anchored(a, sp, shp, -25., +40.)   # patella centre ~40 mm anterior of the joint centre (gold montages: +30 cut it)
            return full + [(1, LOCAL_MM, a), (4, LOCAL_MM, pf), (5, LOCAL_MM, anchored(a, sp, shp, +5., -35.))]
        if plane == "Axial":
            pf = patella if patella is not None else anchored(a, sp, shp, -40., 0.)
            return full + [(1, LOCAL_MM, a), (4, LOCAL_MM, pf), (5, LOCAL_MM, anchored(a, sp, shp, +35., 0.))]
        return full + [(2, LOCAL_MM, anchored(a, sp, shp, 0., -30.)), (3, LOCAL_MM, anchored(a, sp, shp, 0., +30.))]
    if plane == "Coronal":
        if views == "v2add":
            return []
        from training.coronal_views import coronal_bilateral_views
        out = coronal_bilateral_views(tuple(native_shape[-2:]), spacings, size, LOCAL_MM)
        if views == "v3":                     # bilateral locals only (view id 0 is the full field)
            return [v for v in out if int(v[0]) != 0]
        return out
    if views == "v2add":                      # add-on cache: ONLY the rows v2 adds to v1 (coronal adds none)
        return [(4, LOCAL_MM, PF_CENTER[plane])] if plane in PF_CENTER else []
    if views == "v3":
        return [(1, LOCAL_MM, (.5, .5)), (4, LOCAL_MM, PF_CENTER[plane]), (5, LOCAL_MM, POST_CENTER[plane])]
    out = [(0, None, (.5, .5)), (1, LOCAL_MM, (.5, .5))]
    if views == "v2":
        out.append((4, LOCAL_MM, PF_CENTER[plane]))
    elif views != "v1":
        raise ValueError("unknown view set " + str(views))
    return out


class StudySlices(torch.utils.data.Dataset):
    def __init__(self, cache, uids, size, views="v1", anchors=None):
        self.cache, self.uids, self.size, self.views = cache, uids, size, views
        self.anchors = anchors                   # directory of <sha24(study)>.json from knee_anchor_bank_v2.py (views v3k)
        self._resolver = None

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        try:
            return self.build(uid)
        except Exception as exc:
            return {"uid": uid, "error": repr(exc)[:400], "px": torch.zeros((0, self.size, self.size), dtype=torch.uint8),
                    "meta": torch.zeros((0, len(META_COLS)), dtype=torch.float32), "blocks": [], "removed": 0}

    def build(self, uid):
        import cv2
        cv2.setNumThreads(1)
        from training.input_provenance import view_geometry
        from training.intensity_mapping import normalize_intensity
        from training.native_data import select_display_window
        from training.exact_duplicates import ExactDuplicateResolver
        if self._resolver is None:
            self._resolver = ExactDuplicateResolver(None, None, False)
        S = self.size
        with open(study_json(self.cache, uid), encoding="utf-8") as fh:
            study = json.load(fh)
        # study JSONs carry the builder's (Windows) path of every series: relocate to this cache before anything reads them;
        # a series the native cache never rendered (12 studies / 31 series) is dropped so the study keeps its other series
        series, skipped = [], []
        for s in study["series"]:
            path = s["path"]
            if not os.path.exists(path):
                path = os.path.join(self.cache, "series", os.path.basename(path.replace(chr(92), "/")))
            if not os.path.exists(path):
                skipped.append(s["SeriesInstanceUID"]); continue
            series.append(dict(s, path=path))
        if not series:
            raise ValueError("no series file present for this study (%d missing)" % len(skipped))
        study = dict(study, series=series)
        plan = self._resolver.resolve(study)
        self._resolver.cache.clear()
        px, meta, blocks = [], [], []
        for real_idx in plan["representative_indices"]:
            s = study["series"][real_idx]
            path = s["path"]
            volume = np.load(path, mmap_mode="r", allow_pickle=False)
            pos = np.asarray(s["positions"], float)
            if volume.ndim != 3 or len(pos) != len(volume) or not len(pos) or not np.isfinite(pos).all():
                raise ValueError("native volume/slice-position mapping is incomplete")
            if s["plane"] == "Sagittal" and s.get("side") == "R":
                pos = -pos
            order = np.argsort(pos, kind="stable")
            pos = pos[order]
            spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.
            window = select_display_window(s, POLICY)
            fluid = s.get("fluid_sensitive", -1)
            fat = s.get("fat_suppression", -1)
            sidx = len(blocks)
            vol = np.asarray(volume[order])                      # slices in physical order, read once
            order_rep = order                                     # native slice index behind each stored slice (in-plane spacing lookups)
            if 0 < spacing < THIN_MM:
                # thin 3-D stacks (0.4-1.5 mm): average groups of slices into ~SLAB_MM slabs, so every slice contributes
                # (a per-slice window sampler at a physical stride would skip ~90 % of a 0.4 mm stack) and the series
                # carries the same slab statistics as a 2-D acquisition.  Same code at Kaggle inference (StudySlices).
                g = max(1, int(round(SLAB_MM / spacing)))
                n_new = len(pos) // g + (1 if len(pos) % g >= max(1, g // 2) else 0)
                if n_new >= 2:
                    chunks = [vol[j * g:min(len(vol), (j + 1) * g)] for j in range(n_new)]
                    vol = np.stack([np.rint(c.astype(np.float32).mean(0)).astype(vol.dtype) for c in chunks])
                    pos = np.array([pos[j * g:min(len(pos), (j + 1) * g)].mean() for j in range(n_new)])
                    order_rep = np.array([order[j * g] for j in range(n_new)])
                    spacing = float(np.median(np.diff(pos)))
            block = {"series": s["SeriesInstanceUID"], "plane": s["plane"], "n_slices": int(len(pos)), "views": []}
            anchor = pat = None
            sp0 = None
            if self.views in ("v3k", "v4"):
                if self.anchors:
                    ap_ = os.path.join(self.anchors, hashlib.sha256(uid.encode()).hexdigest()[:24] + ".json")
                    ent = (json.load(open(ap_)).get("series", {}) if os.path.exists(ap_) else {}).get(s["SeriesInstanceUID"])
                    if ent and ent.get("in_field", True):
                        anchor = tuple(float(v) for v in ent["joint"])
                        pat = tuple(float(v) for v in ent["patella"]) if ent.get("patella") else None
                if anchor is None:
                    anchor = (.5, .5)                                # no anchor entry: field centre (the foreground heuristic was worse on g02)
                sp0 = s["spacing_mm"][int(order_rep[0])]
            for view_id, crop, center_xy in series_views(s["plane"], volume.shape, s["spacing_mm"], S, self.views, anchor, sp0, pat):
                block["views"].append([int(view_id), len(px)])
                for k in range(len(pos)):
                    sp = s["spacing_mm"][int(order_rep[k])]
                    g = view_geometry(vol.shape[1:], sp, S, crop_mm=crop, center=center_xy)
                    x, y, x1, y1 = g.crop_xyxy
                    pxo, pyo, _, _ = g.valid_xyxy
                    oh, ow = g.resized_shape
                    resized = cv2.resize(vol[k, y:y1, x:x1].astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
                    resized = normalize_intensity(resized, window, POLICY)
                    out = np.zeros((S, S), np.uint8)
                    out[pyo:pyo + oh, pxo:pxo + ow] = np.rint(np.clip(resized, 0, 1) * 255).astype(np.uint8)
                    px.append(out)
                    relative = 2 * (pos[k] - pos[0]) / max(pos[-1] - pos[0], 1e-6) - 1
                    meta.append([sidx, s["slot"], PLANE_ID[s["plane"]], view_id, relative, np.log1p(abs(spacing)),
                                 g.extent / 130., -1 if fluid is None else fluid, -1 if fat is None else fat,
                                 k, len(pos), real_idx])
            blocks.append(block)
            del volume, vol
        if px:
            px = torch.from_numpy(np.stack(px))
            meta = torch.tensor(meta, dtype=torch.float32)
        else:
            px = torch.zeros((0, S, S), dtype=torch.uint8)
            meta = torch.zeros((0, len(META_COLS)), dtype=torch.float32)
        return {"uid": uid, "error": None, "px": px, "meta": meta, "blocks": blocks, "skipped_series": skipped,
                "removed": int(plan.get("removed_series", 0))}


def count_upper(cache, uid, size, views="v1"):
    with open(study_json(cache, uid), encoding="utf-8") as fh:
        study = json.load(fh)
    n = 0
    for s in study["series"]:
        try:
            v = len(series_views(s["plane"], s["native_shape"], s["spacing_mm"], size, "v3" if views in ("v3k", "v4") else views)) + (1 if views == "v4" else 0)
        except Exception:
            v = 3
        n += v * len(s["positions"])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="F:/rsnaknee/cache_native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", default=None)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--margin-gb", type=float, default=60.)
    ap.add_argument("--anchors", default=None, help="anchor bank directory (knee_anchor_bank.py) for --views v3k")
    ap.add_argument("--views", default="v1", choices=["v1", "v2", "v2add", "v3", "v3k", "v4"],
                    help="v2 = v1 + patellofemoral local view for sagittal/axial; v2add = only that added view (an add-on cache "
                         "that compact_train reads next to a v1 cache with --px-extra)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    import pandas as pd
    uids = pd.read_csv(os.path.join(a.cache, "patients.csv"), dtype=str)[UIDK].tolist()
    known = set(uids)
    first = json.load(open(a.first, encoding="utf-8")) if a.first else []
    first = [u for u in first if u in known]
    rest = sorted(known - set(first))
    rng = np.random.default_rng(a.seed)
    rest = [rest[i] for i in rng.permutation(len(rest))]
    order = first + rest
    if a.limit:
        order = order[:a.limit]
    index_path = os.path.join(a.out, "index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path, encoding="utf-8"))
        if index["order"] != order or index["config"]["size"] != a.size or index["config"].get("views", "v1") != a.views:
            raise SystemExit("existing index was built for another order/size/views; use a new --out")
    else:
        counts = [count_upper(a.cache, u, a.size, a.views) for u in order]
        index = {"order": order, "n_upper": int(sum(counts)), "done": {}, "blocks": {}, "cursor": 0, "errors": {},
                 "removed_series": 0, "meta_cols": META_COLS,
                 "config": {"size": a.size, "local_mm": LOCAL_MM, "policy": POLICY,
                            "views": a.views, "anchors": a.anchors,
                            "view_recipe": "global_local+coronal_bilateral" + ("+pf_local" if a.views == "v2" else ""),
                            "duplicates": "collapse_exact",
                            "rows": "one per (series, view, slice), physical order", "gold_first": len(first)}}
        json.dump(index, open(index_path, "w", encoding="utf-8"))
    N = max(1, index["n_upper"])
    gb = N * a.size * a.size / 2 ** 30
    print("studies", len(order), "row upper bound", N, "pixel file GB %.1f" % gb, "count seconds %.1f" % (time.time() - t0), flush=True)
    import shutil
    free = shutil.disk_usage(a.out).free / 2 ** 30
    if not os.path.exists(os.path.join(a.out, "px.u8.npy")) and free < gb + a.margin_gb:
        raise SystemExit("not enough free space: need %.0f GB + %.0f GB margin, have %.0f GB" % (gb, a.margin_gb, free))

    def open_mm(path, shape, dtype):
        if os.path.exists(path):
            mm = np.load(path, mmap_mode="r+")
            if tuple(mm.shape) != tuple(shape):
                raise SystemExit("existing memmap has another shape: " + path)
            return mm
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    px_mm = open_mm(os.path.join(a.out, "px.u8.npy"), (N, a.size, a.size), np.uint8)
    meta_mm = open_mm(os.path.join(a.out, "meta.f32.npy"), (N, len(META_COLS)), np.float32)
    todo = [u for u in order if u not in index["done"] and u not in index["errors"]]
    loader = torch.utils.data.DataLoader(StudySlices(a.cache, todo, a.size, a.views, a.anchors), batch_size=None, shuffle=False,
                                         num_workers=a.workers, prefetch_factor=2 if a.workers else None,
                                         persistent_workers=bool(a.workers))
    state = {"t": time.time(), "studies": 0, "last": time.time()}

    def save(final=False):
        px_mm.flush(); meta_mm.flush()
        tmp = index_path + ".tmp"
        json.dump(index, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, index_path)
        el = time.time() - state["t"]
        sps = state["studies"] / max(el, 1e-6)
        prog = {"done_studies": len(index["done"]), "errors": len(index["errors"]), "total_studies": len(order),
                "rows_written": index["cursor"], "elapsed_s": round(el, 1), "studies_per_s": round(sps, 3),
                "eta_min": round((len(todo) - state["studies"]) / max(sps, 1e-6) / 60, 1), "final": final,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(prog, open(os.path.join(a.out, "progress.json"), "w"))
        print(json.dumps(prog), flush=True)

    for item in loader:
        uid = item["uid"]
        state["studies"] += 1
        if item["error"]:
            index["errors"][uid] = item["error"]
            continue
        n = len(item["px"])
        cur = index["cursor"]
        if cur + n > N:
            raise RuntimeError("row count exceeds the allocated upper bound")
        if n:
            px_mm[cur:cur + n] = item["px"].numpy()
            meta_mm[cur:cur + n] = item["meta"].numpy()
        for b in item["blocks"]:
            b["views"] = [[v, cur + off] for v, off in b["views"]]
        index["done"][uid] = [cur, n, len(item["blocks"])]
        index["blocks"][uid] = item["blocks"]
        index["cursor"] = cur + n
        index["removed_series"] += item["removed"]
        if time.time() - state["last"] > 90:
            save(); state["last"] = time.time()
    save(final=True)
    print("DONE rows", index["cursor"], "studies", len(index["done"]), "errors", len(index["errors"]),
          "minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
