"""Knee anchor bank v2: the knee joint centre (and patella centre) of every study in world coordinates, projected into
each series' pixel grid.  Library (AnchorBank, used by compact_infer.py at Kaggle inference) + CLI (the 5090 bank).

Per study: every aligned sagittal series yields a candidate = centroid of the SKM-TEA joint tissues (any of femoral /
tibial cartilage / meniscus confidently present: max over the three channels > 0.5) on 7 slices ~4 mm apart around the
middle of the stack.  A candidate is kept when its mass >= MASS_MIN pixels, it projects inside the central 15-85 % of
its own series' rows and columns, and it lies within PRIOR_MM of the slab prior.  Candidates within AGREE_MM are
mass-averaged; otherwise the heaviest wins ('disagree' recorded).  With no candidate the anchor is the slab prior.
Slab prior = centre of the intersection of the orthogonal stacks: LR from the sagittal stack, AP from the coronal stack,
SI from the sagittal+coronal field centres (header geometry only, no model).
Fixes over v1 (found on the 58 gold montages): v1's 'sum of three probabilities > 0.5' mask fired on calf tissue in
large-FOV scans (joint at row 0.91-0.96 accepted); v1 took only the first fluid-sensitive series; v1 rejected
oblique-sagittal protocols (cos < 0.97) wholesale; v1's fallback was the sagittal field centre.
Non-fluid-sensitive sagittal series (T1 / PD without fat suppression) give ~zero mass: expected, not a flip error
(checked on 136 gold series x 4 flip combinations).

CLI (5090):  python knee_anchor_bank_v2.py --out F:/rsnaknee/anchors_v2 [--uids list.json] [--limit N]
Output anchors/<sha24(study)>.json: {"study", "version": 2, "joint_lps", "patella_lps", "slab_prior_lps", "method",
  "why": [...], "series": {sid: {"joint": [row_frac, col_frac], "patella": [...] | null, "in_field", "dist_mm"}}}
No label is read anywhere here."""
import argparse, csv, hashlib, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F

GEO = "F:/rsnaknee/geometry"            # 5090 defaults (CLI); the library takes explicit directories
CACHE = "F:/rsnaknee/cache_native"
LO, HI = 0.15, 0.85          # plausibility window (fractions of the source series' rows / cols)
AGREE_MM = 20.0
PRIOR_MM = 35.0              # a segmentation candidate farther than this from the slab prior is rejected (g35: 85 mm)
OBLIQUE_COS = 0.85           # v1 used 0.97 (14 deg): oblique-sagittal knee protocols were rejected wholesale
SPAN_MM = 4.0                # spacing between the segmented slices (3-D stacks otherwise contribute a 3 mm slab)
MASS_MIN = 250               # v1: 400
N_SLICES = 7


def key_for(uid, sid):
    return hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24]


def study_key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:24]


def load_model(weights, dev):
    """the SKM-TEA V-Net through the mainline loader (needs the `training` package on sys.path)."""
    from training.anatomy import load_segmenter
    return load_segmenter(weights, dev)


def project(aff, X):
    """aff [Z,3,3] (col,row,1 -> LPS mm); X [3] LPS -> (z, col, row, |distance to that plane|) of the nearest slice."""
    o = aff[:, :, 2]
    n = np.cross(aff[0, :, 0], aff[0, :, 1]); n /= np.linalg.norm(n)
    d = (o - X[None]) @ n
    z = int(np.argmin(np.abs(d)))
    M = aff[z, :, :2]
    cr, *_ = np.linalg.lstsq(M, X - o[z], rcond=None)
    return z, float(cr[0]), float(cr[1]), float(abs(d[z]))


def centroid(prob, thr=0.5):
    m = prob > thr
    if m.sum() < 40:
        return None, 0
    ys, xs = np.nonzero(m)
    w = prob[m]
    return (float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())), int(m.sum())


class AnchorBank:
    def __init__(self, model, dev, geo_dir=None):
        self.model, self.dev, self.geo_dir = model, dev, geo_dir

    def geometry(self, uid, sid, geo_dir=None):
        p = os.path.join(geo_dir or self.geo_dir, key_for(uid, sid) + ".json")
        return json.load(open(p)) if os.path.exists(p) else None

    @torch.no_grad()
    def segment(self, series, geo, n_slices=N_SLICES):
        from training.anatomy_runtime import native_volume
        vol = native_volume(series["path"])
        mean, std = float(vol.mean()), max(float(vol.std()), 1.)
        aff = np.asarray(geo["affines_col_row_origin_lps_mm"], float)
        if abs(aff[0, 1, 0]) / np.linalg.norm(aff[0, :, 0]) < OBLIQUE_COS or abs(aff[0, 2, 1]) / np.linalg.norm(aff[0, :, 1]) < OBLIQUE_COS:
            return None, "oblique"
        flip_x = aff[0, 1, 0] < 0; flip_y = aff[0, 2, 1] > 0          # same convention as training.anatomy.segment_series
        m = len(vol) // 2
        dz = float(np.linalg.norm(aff[1, :, 2] - aff[0, :, 2])) if len(aff) > 1 else 4.
        step = max(1, int(round(SPAN_MM / max(dz, 1e-3))))
        zs = sorted({min(len(vol) - 1, max(0, m + k * step)) for k in range(-(n_slices // 2), n_slices // 2 + 1)})
        out = []
        for z in zs:
            x = (np.asarray(vol[z], np.float32) - mean) / std
            if flip_x: x = x[:, ::-1]
            if flip_y: x = x[::-1]
            sy, sx = series["spacing_mm"][z]
            shape = (max(16, round(x.shape[1] * sx / .3125)), max(16, round(x.shape[0] * sy / .3125)))
            x = cv2.resize(np.ascontiguousarray(x), shape, interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(x.copy())[None, None].to(self.dev)
            h, w = t.shape[-2:]; t = F.pad(t, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
            p = self.model(t).sigmoid()[..., :h, :w]
            p = F.interpolate(p, size=vol.shape[-2:], mode="bilinear", align_corners=False)[0].float().cpu().numpy()
            if flip_x: p = p[:, :, ::-1]
            if flip_y: p = p[:, ::-1]
            out.append((z, np.ascontiguousarray(p)))
        return out, "ok"

    def candidate(self, s, geo):
        aff = np.asarray(geo["affines_col_row_origin_lps_mm"], float)
        segs, why = self.segment(s, geo)
        if segs is None:
            return None, why
        pts, wts, ppts, pwts = [], [], [], []
        for z, p in segs:
            joint = np.max(p[1:4], axis=0)                       # any joint tissue confidently present
            c, n = centroid(joint, 0.5)
            if c is not None:
                pts.append(aff[z] @ np.array([c[1], c[0], 1.])); wts.append(n)
            cp, npx = centroid(p[0], 0.5)
            if cp is not None and npx >= 60:
                ppts.append(aff[z] @ np.array([cp[1], cp[0], 1.])); pwts.append(npx)
        mass, pmass = int(sum(wts)), int(sum(pwts))
        if mass < MASS_MIN:
            return None, "mass<%d(%d)" % (MASS_MIN, mass)
        X = np.average(np.array(pts), axis=0, weights=np.array(wts, float))
        Xp = np.average(np.array(ppts), axis=0, weights=np.array(pwts, float)) if pmass >= 120 else None
        if Xp is not None and not (Xp[1] < X[1] and Xp[2] > X[2] - 10):   # patella: anterior (smaller LPS y), not below the joint
            Xp = None
        h, w = s["native_shape"][-2:]
        z, c, r, _ = project(aff, X)
        rf, cf = r / max(1, h - 1), c / max(1, w - 1)
        if not (LO <= rf <= HI and LO <= cf <= HI):
            return None, "implausible(rf=%.2f,cf=%.2f)" % (rf, cf)
        return dict(X=X, Xp=Xp, mass=mass, pmass=pmass, sid=s["SeriesInstanceUID"]), "ok"

    def slab_prior(self, uid, d, geo_dir=None):
        per = {}
        for s in d["series"]:
            geo = self.geometry(uid, s["SeriesInstanceUID"], geo_dir)
            if geo is None or not geo["quality"]["aligned"]:
                continue
            aff = np.asarray(geo["affines_col_row_origin_lps_mm"], float)
            h, w = s["native_shape"][-2:]; z = len(aff) // 2
            per.setdefault(s["plane"], []).append(aff[z] @ np.array([(w - 1) / 2., (h - 1) / 2., 1.]))
        if not per:
            return None
        G = np.zeros(3)
        for ax, planes in enumerate((("Sagittal",), ("Coronal",), ("Sagittal", "Coronal"))):
            cs = [c for pl in planes for c in per.get(pl, [])] or [c for pl in per for c in per[pl]]
            G[ax] = float(np.median([c[ax] for c in cs]))
        return G

    def record(self, uid, d, geo_dir=None):
        """anchor record of one study from its native-cache study dict `d` and the geometry directory."""
        sags = [s for s in d["series"] if s["plane"] == "Sagittal" and os.path.exists(s["path"])]
        sags.sort(key=lambda s: -(s.get("fluid_sensitive") or 0))
        rec = {"study": uid, "version": 2, "joint_lps": None, "patella_lps": None, "method": None, "why": [], "series": {}}
        G = self.slab_prior(uid, d, geo_dir)
        rec["slab_prior_lps"] = None if G is None else [float(v) for v in G]
        cands = []
        for s in sags:
            geo = self.geometry(uid, s["SeriesInstanceUID"], geo_dir)
            if geo is None or not geo["quality"]["aligned"]:
                rec["why"].append("no_geo" if geo is None else "not_aligned"); continue
            try:
                c, why = self.candidate(s, geo)
            except Exception as exc:                                   # one bad series must not lose the study
                c, why = None, "error:" + repr(exc)[:80]
            if c is None:
                rec["why"].append(why); continue
            if G is not None and np.linalg.norm(c["X"] - G) > PRIOR_MM:
                rec["why"].append("far_from_prior(%.0fmm)" % np.linalg.norm(c["X"] - G)); continue
            cands.append(c)
        X = Xp = None
        if cands:
            cands.sort(key=lambda c: -c["mass"])
            best = cands[0]
            close = [c for c in cands if np.linalg.norm(c["X"] - best["X"]) <= AGREE_MM]
            if len(close) >= 2:
                X = np.average(np.array([c["X"] for c in close]), axis=0, weights=np.array([c["mass"] for c in close], float))
                rec["method"] = "skm_tea_joint_avg%d" % len(close)
            else:
                X = best["X"]; rec["method"] = "skm_tea_joint"
                if len(cands) >= 2:
                    rec["why"].append("disagree(%.0fmm)" % np.linalg.norm(cands[1]["X"] - best["X"]))
            pc = [c for c in close if c["Xp"] is not None] or [c for c in cands if c["Xp"] is not None]
            if pc:
                Xp = max(pc, key=lambda c: c["pmass"])["Xp"]
        elif G is not None:
            X = G; rec["method"] = "slab_prior"
        else:
            rec["why"].append("no_geometry")
        if X is not None:
            rec["joint_lps"] = [float(v) for v in X]
            if Xp is not None:
                rec["patella_lps"] = [float(v) for v in Xp]
            for s in d["series"]:
                geo = self.geometry(uid, s["SeriesInstanceUID"], geo_dir)
                if geo is None:
                    continue
                aff = np.asarray(geo["affines_col_row_origin_lps_mm"], float)
                h, w = s["native_shape"][-2:]
                z, c, r, dist = project(aff, X)
                entry = {"joint": [r / max(1, h - 1), c / max(1, w - 1)], "in_field": bool(0 <= r < h and 0 <= c < w),
                         "dist_mm": round(dist, 1), "patella": None}
                if Xp is not None:
                    zp, cp, rp, dp = project(aff, Xp)
                    entry["patella"] = [rp / max(1, h - 1), cp / max(1, w - 1)]
                rec["series"][s["SeriesInstanceUID"]] = entry
        return rec


def plausible_v1(rec, d):
    """does an existing record pass the v2 gate?  (joint fractions of every sagittal series inside the window)"""
    if not str(rec.get("method", "")).startswith("skm_tea_joint"):
        return False
    for s in d["series"]:
        if s["plane"] != "Sagittal":
            continue
        e = rec["series"].get(s["SeriesInstanceUID"])
        if e is None or not e["in_field"]:
            return False
        rf, cf = e["joint"]
        if not (LO <= rf <= HI and LO <= cf <= HI):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="F:/rsnaknee/anchors_v2")
    ap.add_argument("--geo", default=GEO)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uids", default=None)
    ap.add_argument("--only-rework", default=None, help="existing anchor dir: process only fallback / implausible records")
    ap.add_argument("--weights", default="F:/rsnaknee/pretrained/skm_tea/model-raw.ckpt")
    ap.add_argument("--code-root", default="D:/rsnaknee", help="directory holding the mainline `training` package")
    a = ap.parse_args()
    if a.code_root and os.path.isdir(a.code_root) and a.code_root not in sys.path:
        sys.path.insert(0, a.code_root)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    bank = AnchorBank(load_model(a.weights, dev), dev, a.geo)
    uids = [r["StudyInstanceUID"] for r in csv.DictReader(open(os.path.join(a.cache, "patients.csv")))]
    if a.uids: uids = json.load(open(a.uids))
    if a.limit: uids = uids[:a.limit]
    t0 = time.time(); stats = {"ok": 0, "ok_avg": 0, "disagree": 0, "fallback": 0, "no_geometry": 0, "patella": 0, "skipped_v1_ok": 0}
    why_all = {}
    for i, uid in enumerate(uids):
        outp = os.path.join(a.out, study_key(uid) + ".json")
        d = json.load(open(os.path.join(a.cache, "studies", study_key(uid) + ".json"), encoding="utf-8"))
        if a.only_rework:
            v1p = os.path.join(a.only_rework, study_key(uid) + ".json")
            if os.path.exists(v1p) and plausible_v1(json.load(open(v1p)), d):
                stats["skipped_v1_ok"] += 1
                continue
        elif os.path.exists(outp) and json.load(open(outp)).get("version") == 2:
            continue
        rec = bank.record(uid, d)
        m = rec["method"] or ""
        if m.startswith("skm_tea_joint"):
            stats["ok"] += 1
            if "avg" in m: stats["ok_avg"] += 1
        elif m == "slab_prior":
            stats["fallback"] += 1
        else:
            stats["no_geometry"] += 1
        if rec["patella_lps"] is not None: stats["patella"] += 1
        for w in rec["why"]:
            k = w.split("(")[0]; why_all[k] = why_all.get(k, 0) + 1
            if k == "disagree": stats["disagree"] += 1
        json.dump(rec, open(outp, "w"), indent=0)
        if (i + 1) % 100 == 0:
            print(i + 1, stats, "%.0fs" % (time.time() - t0), flush=True)
    print("done", stats, "why:", why_all, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
