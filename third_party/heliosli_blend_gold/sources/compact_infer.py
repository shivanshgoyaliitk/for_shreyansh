"""Compact-model inference worker (Kaggle or local): raw DICOM -> image scores for one list of studies.

Path, identical to how the compact model was trained and evaluated:
  raw DICOM  --mainline training.native_cache.prepare (one study at a time)-->  native uint16 series
             --px_cache.StudySlices.build (the SAME file that built the pixel cache)-->  288 px slices
             --compact_train evaluation windows (every third slice as centre, both views, 3 adjacent slices)-->
             --centre crop 256, fold models of compact_train.Net, mean logit-->  12 probabilities
Nothing is trained here.  No report, no label, no test-time fitting.  Failed or unreached studies are
listed as fallbacks for the controller.
"""
import argparse, contextlib, csv, hashlib, io, json, os, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import numpy as np
import torch

UIDK = "StudyInstanceUID"
FORCED_SIDE = {}        # study uid -> "L" / "R": the native render uses this side instead of tags / sagittal IPP x


class CStudies(torch.utils.data.Dataset):
    def __init__(self, data, split, uids, tmp, code, size, eval_stride=3, views="v1", slices=3, anchor_weights=None, phys_mm=0.0,
                 side_model=None, side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0, extent_meta=False):
        self.data, self.split, self.uids = str(data), split, list(uids)
        # extent (M4) fold models read three more meta columns: series index, plane, slices in the series -- the same cache
        # columns (0, 2, 10) compact_train.Studies appends with extent_meta
        self.extent_meta = bool(extent_meta)
        self.tmp, self.code, self.size = str(tmp), str(code), int(size)
        self.eval_stride, self.views, self.slices = int(eval_stride), str(views), int(slices)
        self.phys_mm = float(phys_mm)
        self.anchor_weights = None if anchor_weights is None else str(anchor_weights)
        if self.views in ("v3k", "v4") and not self.anchor_weights:
            raise ValueError("views v3k need --anchor-weights (SKM-TEA checkpoint) to rebuild the knee anchors at inference")
        self.native = None
        # image side check (bundles that ship a side model): studies without one explicit DICOM laterality tag are
        # scored by the orientation classifier on their canonical coronal/axial slices; a confidently mirrored study is
        # re-rendered with the other side (native render, geometry, anchors and views all follow).  None = off.
        self.side_model = None if side_model is None else str(side_model)
        self.side_threshold, self.side_max_x = float(side_threshold), float(side_max_x)
        self.side_position_mm = float(side_position_mm)
        self.side_net = None
        self.source_directory = True     # False: read DICOMs from the chunked competition archive under <data>/download

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        for p in (self.code, str(HERE)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import probe_infer as PI
        self.PI = PI
        self.native = PI.install_native(True)
        import px_cache
        import compact_train
        self.builder = px_cache.StudySlices(self.tmp, [], self.size, self.views)   # (self.slices is the window depth K)
        self.window_rows = compact_train.window_rows          # the trainer's own window list (eval mode)
        self.bank = None
        if self.views in ("v3k", "v4"):
            # knee-centred views: per-series geometry from the DICOM headers (mainline geometry bank, the code that
            # built the training geometry) + SKM-TEA joint anchors (knee_anchor_bank_v2, the code of the training bank)
            import training.geometry_bank as GB
            import knee_anchor_bank_v2 as KB
            self.GB, self.KB = GB, KB
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            if dev == "cpu":
                torch.set_num_threads(2)
            self.bank = KB.AnchorBank(KB.load_model(self.anchor_weights, dev), dev)
        import training.dicom_geometry as G
        if not getattr(G, "_forced_side_wrapped", False):
            import dataclasses
            orig = G.decide_study_side

            def decide(study, tag_values, sagittal_x, _orig=orig):
                r = _orig(study, tag_values, sagittal_x)
                f = FORCED_SIDE.get(str(study))
                return r if f is None else dataclasses.replace(r, side=f, source="image_side_check")
            G.decide_study_side = decide
            G._forced_side_wrapped = True
        if self.side_model:
            import side_common as SC
            self.SC = SC
            sdev = "cuda" if torch.cuda.is_available() else "cpu"
            net = SC.build_side_net()
            net.load_state_dict(torch.load(self.side_model, map_location="cpu", weights_only=True), strict=True)
            self.side_net, self.side_dev = net.eval().to(sdev), sdev

    def _prepare(self, uid, work):
        with contextlib.redirect_stdout(io.StringIO()):
            self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=self.source_directory)
        if self.bank is not None or self.side_net is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.GB.build(root=self.data, manifest=str(work / "manifest.json"), output=str(work / "geometry"),
                              split=self.split, source_directory=self.source_directory)

    def _native_and_geometry(self, uid, work):
        """native render + geometry bank for one study; with a side model, the image side check (may re-render)."""
        self.PI.OneStudySource.current = uid
        self._prepare(uid, work)
        if self.side_net is None:
            return None
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
            study = json.load(fh)
        sides = {s.get("side") for s in study["series"]}
        cur = next(iter(sides)) if len(sides) == 1 else None
        status = set()
        for s in study["series"]:
            gp = work / "geometry" / (hashlib.sha256((uid + "/" + s["SeriesInstanceUID"]).encode()).hexdigest()[:24] + ".json")
            if gp.exists():
                status.add(json.loads(gp.read_text(encoding="utf-8")).get("laterality_status"))
        sag = [float(np.median(s["positions"])) for s in study["series"] if s.get("plane") == "Sagittal" and s.get("positions")]
        x = float(np.median(sag)) if sag else None
        note = {"side": cur, "status": "|".join(sorted(str(v) for v in status)), "x": None if x is None else round(x, 1),
                "score": None, "flipped": False}
        if status == {"explicit_study_tags"} and cur in ("L", "R"):
            # one explicit DICOM laterality tag stands unless the knee lies clearly on the other side of the scanner
            # (10-07: 18 training studies had the tag contradicted by both the position and the images)
            pos = None if x is None or abs(x) < self.side_position_mm else ("L" if x >= 0 else "R")
            if pos is None or pos == cur:
                return note
            note["tag_contradicted_by_position"] = True
        if self.side_max_x > 0 and cur in ("L", "R") and x is not None and abs(x) >= self.side_max_x:
            return note                                   # far from the scanner centre: the sagittal IPP x rule holds
        ims = []
        for s in study["series"]:
            if s.get("plane") not in self.SC.SIDE_PLANES:
                continue
            p = Path(s["path"])
            if not p.exists():
                p = work / "series" / os.path.basename(str(s["path"]).replace(chr(92), "/"))
            if p.exists():
                ims.append(self.SC.side_inputs(np.load(p, mmap_mode="r")))
        ims = [v for v in ims if len(v)]
        if not ims:
            return note
        xb = torch.from_numpy(np.concatenate(ims)).float().div_(255.)[:, None]
        with torch.inference_mode():
            z = self.side_net(((xb - 0.45) / 0.25).to(self.side_dev)).float().squeeze(1)
        score = float(z.mean())
        note["score"] = round(score, 3)
        if score < -self.side_threshold:
            new = "R" if cur in (None, "L") else "L"
            FORCED_SIDE[uid] = new
            shutil.rmtree(work, ignore_errors=True)
            self._prepare(uid, work)
            with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                got = {s.get("side") for s in json.load(fh)["series"]}
            if got != {new}:
                raise ValueError("forced side %s not applied (%s)" % (new, got))
            note.update(flipped=True, new_side=new)
        return note

    def rows(self, uid, forced_side=None):
        """training-cache override builder: the same native -> geometry -> anchors -> views path for one study, with an
        optional forced side; returns (StudySlices.build output, set of sides in the rendered study)."""
        if self.native is None:
            self._setup()
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        work = Path(self.tmp) / ("r_" + key[:16])
        if forced_side is not None:
            FORCED_SIDE[uid] = forced_side
        try:
            self.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=self.source_directory)
                self.GB.build(root=self.data, manifest=str(work / "manifest.json"), output=str(work / "geometry"),
                              split=self.split, source_directory=self.source_directory)
            self.builder.cache = str(work)
            with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                study = json.load(fh)
            if self.bank is not None:
                rec = self.bank.record(uid, study, geo_dir=str(work / "geometry"))
                (work / "anchors").mkdir(exist_ok=True)
                with open(work / "anchors" / (self.KB.study_key(uid) + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                self.builder.anchors = str(work / "anchors")
            return self.builder.build(uid), {s.get("side") for s in study["series"]}
        finally:
            FORCED_SIDE.pop(uid, None)
            shutil.rmtree(work, ignore_errors=True)

    def __getitem__(self, i):
        uid = self.uids[i]
        t0 = time.perf_counter()
        work = Path(self.tmp) / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
        t1 = t0
        try:
            if self.native is None:
                self._setup()
            side_note = self._native_and_geometry(uid, work)
            t1 = time.perf_counter()
            self.builder.cache = str(work)
            if self.bank is not None:
                key = self.KB.study_key(uid)
                with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                    study = json.load(fh)
                rec = self.bank.record(uid, study, geo_dir=str(work / "geometry"))
                (work / "anchors").mkdir(exist_ok=True)
                with open(work / "anchors" / (key + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                self.builder.anchors = str(work / "anchors")
                anchor_note = {"method": rec["method"], "why": rec["why"][:4]}
            else:
                anchor_note = None
            t2 = time.perf_counter()
            item = self.builder.build(uid)
            px, meta = item["px"].numpy(), item["meta"].numpy()
            blocks = [{"n_slices": b["n_slices"], "views": [(0, v, st) for v, st in b["views"]]} for b in item["blocks"]]
            cand = self.window_rows(blocks, lambda f, r: meta[r], "eval", self.slices, self.eval_stride, phys_mm=self.phys_mm)
            if not cand:
                raise ValueError("no usable window")
            rows = np.array(cand, np.int64)                 # (file, centre row, K slice rows)
            img = np.stack([px[rows[:, 2 + d]] for d in range(self.slices)], 1)
            m = meta[rows[:, 1]]
            meta6 = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1).astype(np.float32)
            if self.extent_meta:
                meta6 = np.concatenate([meta6, np.stack([m[:, 0], m[:, 2], m[:, 10]], 1).astype(np.float32)], 1)
            out = {"uid": uid, "error": None, "images": torch.from_numpy(img), "meta": torch.from_numpy(meta6), "anchor": anchor_note,
                   "side": side_note}
        except Exception as exc:
            t2 = locals().get("t2", t1)
            out = {"uid": uid, "error": repr(exc)[:300], "images": torch.zeros((0, self.slices, self.size, self.size), dtype=torch.uint8),
                   "meta": torch.zeros((0, 9 if self.extent_meta else 6), dtype=torch.float32), "anchor": None}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["seconds"] = [round(t1 - t0, 3), round(time.perf_counter() - t2, 3), round(t2 - t1, 3)]   # decode, render, anchor
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--code", required=True)
    ap.add_argument("--models", required=True, help="folder with <arm>_s<seed>_f<fold>.pt state dicts")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--init", default=None, help="ofl_l backbone: the public OFL-L weights (safetensors) the fold models are rebuilt on")
    ap.add_argument("--freeze-blocks", type=int, default=16)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--views", default="v1", help="px_cache view set the fold models were trained with (v1 / v2)")
    ap.add_argument("--state-spec", default=None, help="state_spec json of the report-state construct the fold models were trained with")
    ap.add_argument("--slices", type=int, default=3, help="adjacent slices per window the fold models were trained with")
    ap.add_argument("--anchor-weights", default=None, help="views v3k: the SKM-TEA checkpoint (anatomy.pt of the code dataset) for the knee anchors")
    ap.add_argument("--phys-mm", type=float, default=0.0, help="window spacing in mm the fold models were trained with (compact_train --phys-mm)")
    ap.add_argument("--eval-stride", type=int, default=3, help="window centres: every k-th slice (3 = training-time validation default; 1 = every slice)")
    ap.add_argument("--assignment", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--item-timeout", type=float, default=900.)
    ap.add_argument("--deadline", type=float, default=0.)
    ap.add_argument("--dump-logits", default=None)
    ap.add_argument("--models2", default=None, help="second model group (another architecture), run on the same windows")
    ap.add_argument("--arm2", default=None, help="file prefix of the second group (default: --arm)")
    ap.add_argument("--backbone2", default=None)
    ap.add_argument("--w2", default=None, help="weight of group 2 per class: one number or 12 comma-separated (label order); "
                    "p = (1 - w) p_group1 + w p_group2")
    ap.add_argument("--side-model", default=None, help="orientation classifier (side_common.build_side_net state dict); off when absent")
    ap.add_argument("--side-threshold", type=float, default=2.0, help="flip a study only when its mean logit < -threshold")
    ap.add_argument("--side-max-x", type=float, default=0.0, help="> 0: skip the check for untagged studies this far (mm) from x = 0")
    ap.add_argument("--side-position-mm", type=float, default=20.0, help="a DICOM tag is re-checked only when the knee lies this far on the other side")
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp or (out / "tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if a.assignment:
        uids = json.loads(Path(a.assignment).read_text(encoding="utf-8"))
    else:
        with open(Path(a.data) / (a.split + ".csv"), encoding="utf-8") as fh:
            uids = [r[UIDK] for r in csv.DictReader(fh)]
    if len(uids) != len(set(uids)):
        raise SystemExit("duplicate study ids")

    dev = torch.device("cuda")
    for p in (a.code, str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import timm
    import compact_train as CT
    LABELS = CT.LABELS
    cap = torch.cuda.get_device_capability()[0]
    amp = a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16")
    state_spec = json.load(open(a.state_spec, encoding="utf-8")) if a.state_spec else None

    def load_group(folder, arm, backbone):
        fs = sorted(Path(folder).glob(arm + "_s*_f*.pt"))
        if not fs:
            raise SystemExit("no model files for arm %s in %s" % (arm, folder))
        ms, n_sp = [], 0
        for p in fs:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            extent = any(k.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for k in sd)   # M4 fold model (OA extent)
            if extent:
                N_EXTENT[0] += 1
            spatial = any(k == "sp_q" or k.endswith(".sp_q") for k in sd)       # P2 fold model (per-class spatial attention)
            n_sp += int(spatial)
            kw = {}
            if spatial:
                kw["spatial"] = True
            if extent:
                kw["extent"] = True
            m = CT.Net(backbone, a.init if backbone == CT.OFL_BACKBONE else None, freeze_blocks=a.freeze_blocks, crop=a.crop,
                       state_spec=state_spec, in_chans=a.slices, **kw)
            if backbone == CT.OFL_BACKBONE:
                m.load_deploy_state_dict(sd)          # fold model = public weights + trained tail/head
            else:
                m.load_state_dict(sd, strict=True)
            ms.append(m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last))
        return fs, ms, n_sp

    N_EXTENT = [0]
    files, models, n_spatial = load_group(a.models, a.arm, a.backbone)
    files2, models2, n_spatial2, W2 = [], [], 0, None
    if a.models2:
        if not a.backbone2 or a.backbone2 == CT.OFL_BACKBONE or a.backbone == CT.OFL_BACKBONE:
            raise SystemExit("--models2 needs --backbone2 (timm backbones only, no OFL shared trunk)")
        files2, models2, n_spatial2 = load_group(a.models2, a.arm2 or a.arm, a.backbone2)
        w = [float(x) for x in (a.w2 or "0.5").split(",")]
        w = w * len(LABELS) if len(w) == 1 else w
        if len(w) != len(LABELS) or not all(0.0 <= x <= 1.0 for x in w):
            raise SystemExit("--w2: one weight or %d weights in [0, 1]" % len(LABELS))
        W2 = torch.tensor(w, dtype=torch.float64, device=dev)
    shared_trunk = a.backbone == CT.OFL_BACKBONE and len(models) > 1
    if shared_trunk:
        # all fold models share the frozen trunk bit-for-bit; keep one copy and run it once per window batch
        sd0 = models[0].enc.state_dict()
        for m in models[1:]:
            sdm = m.enc.state_dict()
            for k in models[0].enc.frozen_keys:
                if not torch.equal(sdm[k], sd0[k]):
                    raise SystemExit("fold models do not share the frozen trunk: " + k)
    print("MODEL_READY", json.dumps({"gpu": torch.cuda.get_device_name(), "amp": amp, "models": len(models),
                                     "models2": len(models2), "spatial_models": n_spatial + n_spatial2, "extent_models": N_EXTENT[0],
                                     "studies": len(uids), "seconds": round(time.time() - t_start, 1)}), flush=True)
    off = (a.size - a.crop) // 2

    def run_model(m, img, meta, mask, mode):
        if mode == "fp32":
            return m(img, meta, mask, chunk=a.chunk).float()
        with torch.autocast("cuda", dtype=torch.float16 if mode == "fp16" else torch.bfloat16):
            return m(img, meta, mask, chunk=a.chunk).float()

    def heads_from_tokens(m, tok, meta, mask):
        """compact_train.Net.head: encoder output -> class logits (the same code the trainer used, state construct included)."""
        return m.head(tok, meta, mask)

    def run_shared(img, meta, mask, mode):
        """OFL fold models: frozen trunk once per window chunk, then every fold's trained tail + head."""
        x = img[mask]
        m0 = models[0]
        tails = [[] for _ in models]
        step = a.chunk or len(x)
        ctx = contextlib.nullcontext() if mode == "fp32" else torch.autocast("cuda", dtype=torch.float16 if mode == "fp16" else torch.bfloat16)
        with ctx:
            for s0 in range(0, len(x), step):
                xs = x[s0:s0 + step].float().div_(255.)
                xs = ((xs - m0.mean) / m0.std).contiguous(memory_format=torch.channels_last)
                tr = m0.enc.trunk(xs)
                for i, m in enumerate(models):
                    tails[i].append(m.enc.tail(tr))
            zs = [heads_from_tokens(m, torch.cat(tails[i]), meta, mask) for i, m in enumerate(models)]
        return torch.stack([z[0].float() for z in zs])

    @torch.inference_mode()
    def score(images, meta):
        img = images.to(dev, non_blocking=True)[None, :, :, off:off + a.crop, off:off + a.crop]
        meta = meta.to(dev)[None]
        mask = torch.ones((1, img.shape[1]), dtype=torch.bool, device=dev)
        if shared_trunk:
            zs = run_shared(img, meta, mask, amp)
            if not torch.isfinite(zs).all():
                zs = run_shared(img, meta, mask, "fp32")
            return zs.mean(0), zs
        zs = []
        for m in models:
            z = run_model(m, img, meta, mask, amp)
            if not torch.isfinite(z).all():
                z = run_model(m, img, meta, mask, "fp32")
            zs.append(z[0])
        zs = torch.stack(zs)
        if not models2:
            return zs.mean(0), zs
        zs2 = []
        for m in models2:                                     # second architecture on the same windows (one decode)
            z = run_model(m, img, meta, mask, amp)
            if not torch.isfinite(z).all():
                z = run_model(m, img, meta, mask, "fp32")
            zs2.append(z[0])
        zs2 = torch.stack(zs2)
        p1, p2 = torch.sigmoid(zs.mean(0).double()), torch.sigmoid(zs2.mean(0).double())
        pf = ((1 - W2) * p1 + W2 * p2).clamp(1e-12, 1 - 1e-12)
        return torch.log(pf / (1 - pf)), torch.cat([zs, zs2])

    fh = open(out / "predictions.csv", "w", encoding="utf-8", newline="")
    wr = csv.writer(fh)
    wr.writerow([UIDK] + LABELS)
    fallback, stats = {}, {"decode_s": 0., "render_s": 0., "gpu_s": 0., "windows": 0, "wait_s": 0., "rows": 0, "anchor_s": 0.}
    per_study, dump = [], {}
    anchor_methods = {}
    side_counts = {"checked": 0, "flipped": 0}

    def handle(item):
        uid = item["uid"]
        if item["error"] or not len(item["images"]):
            fallback[uid] = item["error"] or "no usable window"
            return
        tg = time.perf_counter()
        try:
            zmean, zs = score(item["images"], item["meta"])
        except Exception as exc:
            fallback[uid] = "gpu: " + repr(exc)[:200]
            return
        if not torch.isfinite(zmean).all():
            fallback[uid] = "non-finite score"
            return
        p = torch.sigmoid(zmean).double().cpu().numpy()
        dt = time.perf_counter() - tg
        wr.writerow([uid] + ["%.6f" % float(x) for x in p])
        fh.flush()
        stats["rows"] += 1
        stats["decode_s"] += item["seconds"][0]
        stats["render_s"] += item["seconds"][1]
        stats["anchor_s"] += item["seconds"][2] if len(item["seconds"]) > 2 else 0.
        if item.get("side") and item["side"].get("score") is not None:
            side_counts["checked"] += 1
            side_counts["flipped"] += int(bool(item["side"].get("flipped")))
        if item.get("anchor"):
            anchor_methods[item["anchor"]["method"]] = anchor_methods.get(item["anchor"]["method"], 0) + 1
        stats["gpu_s"] += dt
        stats["windows"] += len(item["images"])
        per_study.append([uid, len(item["images"]), item["seconds"][0], item["seconds"][1], round(dt, 3)])
        if a.dump_logits:
            dump[uid] = zs.float().cpu().numpy().tolist()

    pos = 0
    timed_out = False
    while pos < len(uids) and not timed_out:
        ds = CStudies(a.data, a.split, uids[pos:], tmp, a.code, a.size, a.eval_stride, a.views, a.slices, a.anchor_weights, a.phys_mm,
                      a.side_model, a.side_threshold, a.side_max_x, a.side_position_mm, extent_meta=N_EXTENT[0] > 0)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                             prefetch_factor=a.prefetch if a.workers else None, persistent_workers=False,
                                             multiprocessing_context="spawn" if a.workers else None,
                                             timeout=a.item_timeout if a.workers else 0)
        try:
            mark = time.perf_counter()
            for item in loader:
                stats["wait_s"] += time.perf_counter() - mark
                handle(item)
                pos += 1
                if pos % 20 == 0 or pos == len(uids) or pos <= 3:
                    el = time.time() - t_start
                    print(json.dumps({"done": pos, "of": len(uids), "scored": stats["rows"], "fallback": len(fallback),
                                      "elapsed_s": round(el, 1), "s_per_study": round(el / max(pos, 1), 2),
                                      "decode_s": round(stats["decode_s"], 1), "render_s": round(stats["render_s"], 1),
                                      "gpu_s": round(stats["gpu_s"], 1), "anchor_s": round(stats["anchor_s"], 1),
                                      "anchors": anchor_methods, "windows": stats["windows"]}), flush=True)
                if a.deadline and time.time() > a.deadline:
                    timed_out = True
                    break
                mark = time.perf_counter()
        except Exception as exc:
            if pos < len(uids):
                fallback[uids[pos]] = "loader: " + repr(exc)[:200]
            pos += 1
        del loader
    for u in uids[pos:]:
        fallback[u] = "not reached before the deadline"
    fh.close()
    shutil.rmtree(tmp, ignore_errors=True)
    if a.dump_logits:
        Path(a.dump_logits).write_text(json.dumps(dump), encoding="utf-8")
    receipt = {"status": "PASS", "arm": a.arm, "models": [p.name for p in files], "backbone": a.backbone, "shared_trunk": bool(shared_trunk), "init": a.init,
               "spatial_models": n_spatial, "extent_models": N_EXTENT[0], "group2": None if not models2 else {"models": [p.name for p in files2], "backbone": a.backbone2,
                                                                                   "spatial_models": n_spatial2, "w2": W2.tolist()},
               "studies": len(uids), "scored": stats["rows"], "fallback": fallback, "timed_out": timed_out, "amp": amp,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "timm": timm.__version__,
               "seconds": round(time.time() - t_start, 1), "stats": {k: round(v, 1) for k, v in stats.items()},
               "per_study": per_study[:50], "anchor_methods": anchor_methods, "views": a.views, "phys_mm": a.phys_mm,
               "side_check": None if not a.side_model else dict(side_counts, threshold=a.side_threshold, max_x=a.side_max_x, position_mm=a.side_position_mm,
                                                                  model_sha256=hashlib.sha256(Path(a.side_model).read_bytes()).hexdigest()),
               "anchor_weights_sha256": (hashlib.sha256(Path(a.anchor_weights).read_bytes()).hexdigest() if a.anchor_weights else None),
               "sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                          for n in ("px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py")
                          + (("side_common.py",) if a.side_model else ())
                          + (("knee_anchor_bank_v2.py",) if a.views in ("v3k", "v4") else ())},
               "model": "compact fine-tuned image model on the pixel cache"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("WORKER_DONE", json.dumps({k: receipt[k] for k in ("studies", "scored", "timed_out", "seconds", "amp")}),
          "fallback", len(fallback), flush=True)


if __name__ == "__main__":
    main()
