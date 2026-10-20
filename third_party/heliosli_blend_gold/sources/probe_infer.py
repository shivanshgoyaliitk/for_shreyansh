"""probe_ inference worker (Kaggle or local): raw DICOM -> label-free image scores for one list of studies.

Path, identical to how the probe features were cached:
  raw DICOM  --mainline training.native_cache.prepare (one study at a time)-->  native uint16 series
             --fe_extract.StudyWindows.build (the SAME file that built the feature cache)-->  384 px windows
             --frozen OrthoFoundation-L public SSL weights-->  CLS + 6x6 patch cells --PCA-256 (saved pca.npz)-->
             --saved probe heads (probe_train.Head, K fold models x seeds), mean logit-->  12 probabilities
Nothing is trained here.  No report, no label, no test-time adaptation.  A study that cannot be decoded, or
that is not reached before the deadline, is listed as a fallback (the controller fills it with the mean score).
Everything here is a probe_ scaffold (frozen encoder + light heads); it is not the mainline model.
"""
import argparse, contextlib, csv, hashlib, inspect, io, json, os, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import numpy as np
import torch

UIDK = "StudyInstanceUID"


class OneStudySource:
    """Directory reader that lists only the study being prepared (the stock one lists every DICOM of the split)."""
    current = None

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split

    def read(self, item):
        return (self.root / (item.filename if hasattr(item, "filename") else item)).read_bytes()

    def infolist(self):
        base = self.root / (self.split + "_series") / OneStudySource.current
        return [SimpleNamespace(filename=p.relative_to(self.root).as_posix()) for p in sorted(base.glob("*/*.dcm"))]

    def close(self):
        pass


def quantize_series_slicewise(images):
    """Same global float32 extrema and per-voxel arithmetic as training.native_cache.quantize_series, without
    the whole-volume float32 copy (taken from the mainline's accepted Kaggle memory patch)."""
    lo = min(float(np.asarray(x, np.float32).min()) for x in images)
    hi = max(float(np.asarray(x, np.float32).max()) for x in images)
    scale = max(hi - lo, 1e-6)
    encoded = np.empty((len(images), *images[0].shape), dtype=np.uint16)
    for z, image in enumerate(images):
        volume = np.asarray(image, np.float32)
        encoded[z] = np.rint((volume - lo) / scale * 65535).clip(0, 65535).astype(np.uint16)
    window = np.percentile(encoded[:, ::4, ::4], [1, 99]).astype(float).tolist()
    return encoded, {"minimum": lo, "maximum": hi, "max_absolute_quantization_error_bound": scale / 131070,
                     "display_window_uint16": window}


def install_native(lean=True):
    import training.native_cache as native
    if lean:
        src = inspect.getsource(native.prepare)
        for before, after in [
            ("pydicom.dcmread(io.BytesIO(archive.read(i)))",
             "pydicom.dcmread(io.BytesIO(archive.read(i)), stop_before_pixels=True)"),
            ("geom.pixels_from_dataset(objects[sid][name],path=name)",
             "geom.pixels_from_dataset(pydicom.dcmread(io.BytesIO(archive.read(name))),path=name)")]:
            if src.count(before) != 1:
                raise RuntimeError("training.native_cache.prepare differs from the frozen source")
            src = src.replace(before, after)
        exec(compile(src, "probe_infer:prepare", "exec"), native.__dict__)
        native.quantize_series = quantize_series_slicewise
    native.DirectorySource = OneStudySource
    return native


class Studies(torch.utils.data.Dataset):
    def __init__(self, data, split, uids, tmp, code, lean=True):
        self.data, self.split, self.uids = str(data), split, list(uids)
        self.tmp, self.code, self.lean = str(tmp), str(code), lean
        self.native = None
        self.windows = None

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        if self.code not in sys.path:
            sys.path.insert(0, self.code)
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        self.native = install_native(self.lean)
        import fe_extract
        self.windows = fe_extract.StudyWindows(self.tmp, [])

    def __getitem__(self, i):
        uid = self.uids[i]
        t0 = time.perf_counter()
        work = Path(self.tmp) / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
        t1 = t0
        try:
            if self.native is None:
                self._setup()
            OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=True)
            t1 = time.perf_counter()
            self.windows.cache = str(work)
            item = self.windows.build(uid)
            item["error"] = None
        except Exception as exc:
            item = {"uid": uid, "error": repr(exc)[:300], "images": torch.zeros((0, 3, 384, 384), dtype=torch.uint8),
                    "meta": torch.zeros((0, 12), dtype=torch.float32), "series": [], "removed": 0}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        item["seconds"] = [round(t1 - t0, 3), round(time.perf_counter() - t1, 3)]
        return item


def load_heads(heads_dir, arm, hidden, dev):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from probe_train import Head, LABELS
    if Path(heads_dir).is_file():                     # one file: {"<arm>_s<seed>_<fold>.pt": state_dict}
        blob = torch.load(heads_dir, map_location="cpu", weights_only=True)
        items = sorted((k, v) for k, v in blob.items() if k.startswith(arm + "_s"))
    else:
        items = [(p.name, torch.load(p, map_location="cpu", weights_only=True))
                 for p in sorted(Path(heads_dir).glob(arm + "_s*_*.pt"))]
    if not items:
        raise SystemExit("no head files for arm " + arm)
    heads = []
    for name, state in items:
        h = Head(hidden)
        h.load_state_dict(state, strict=True)
        heads.append(h.eval().requires_grad_(False).to(dev))
    return heads, LABELS, [name for name, _ in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with <split>.csv, <split>_series.csv, <split>_series/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--code", required=True, help="folder that contains the mainline 'training' package")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pca", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--assignment", default=None, help="json list of study ids; default = every id of <split>.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--item-timeout", type=float, default=900., help="seconds to wait for one study from the loader")
    ap.add_argument("--deadline", type=float, default=0., help="unix time after which no new study is started")
    ap.add_argument("--dump-features", default=None)
    ap.add_argument("--stock-native", action="store_true", help="use the unpatched native builder (parity checks)")
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
    if a.code not in sys.path:
        sys.path.insert(0, a.code)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import fe_extract as FE
    import timm
    from safetensors.torch import load_file
    cap = torch.cuda.get_device_capability()[0]
    amp = a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False, num_classes=0, global_pool="",
                              img_size=FE.IMG)
    model.load_state_dict(load_file(a.weights), strict=True)
    model.rope = None
    model.eval().requires_grad_(False).to(dev)
    prefix = int(getattr(model, "num_prefix_tokens", 5))
    mean = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
    g = FE.IMG // 16
    blk = g // FE.GRID
    z = np.load(a.pca)
    pca_mean, pca_comp = torch.from_numpy(z["mean"]).to(dev), torch.from_numpy(z["components"]).to(dev)
    zn = np.load(a.norm)
    c_mu, c_sd = torch.from_numpy(zn["mu"]).to(dev), torch.from_numpy(zn["sd"]).to(dev)
    heads, LABELS, head_names = load_heads(a.heads, a.arm, a.hidden, dev)
    print("MODEL_READY", json.dumps({"gpu": torch.cuda.get_device_name(), "amp": amp, "heads": len(heads), "studies": len(uids), "seconds": round(time.time() - t_start, 1)}), flush=True)
    retries = {"fp32_batches": 0}

    def forward_tokens(x, mode):
        if mode == "fp32":
            return model.forward_features(x).float()
        dt = torch.float16 if mode == "fp16" else torch.bfloat16
        with torch.autocast("cuda", dtype=dt):
            return model.forward_features(x).float()

    @torch.inference_mode()
    def encode(images_u8):
        out_cls, out_cells = [], []
        for chunk in images_u8.split(a.batch):
            x = chunk.to(dev, non_blocking=True).float().div_(255.)
            x = (x - mean) / std
            tok = forward_tokens(x, amp)
            if not torch.isfinite(tok).all():
                tok = forward_tokens(x, "fp32")
                retries["fp32_batches"] += 1
            b = tok.shape[0]
            if tok.shape[1] != prefix + g * g:
                raise ValueError("unexpected token count %s" % (tuple(tok.shape),))
            patch = tok[:, prefix:].reshape(b, FE.GRID, blk, FE.GRID, blk, tok.shape[-1]).mean((2, 4))
            out_cls.append(tok[:, 0])
            out_cells.append(patch.reshape(b, FE.GRID * FE.GRID, tok.shape[-1]))
        return torch.cat(out_cls), torch.cat(out_cells)

    @torch.inference_mode()
    def score(cls, cells, meta):
        cls16 = cls.half()                                             # the cache stores float16
        cells16 = ((cells - pca_mean) @ pca_comp).half()
        x_cells = ((cells16.float() - c_mu) / c_sd).clamp_(-8, 8).half().float()
        x_cls = cls16.float()
        meta = meta.to(dev)
        slot = meta[:, 1].long().clamp(0, 9)[None]
        view = meta[:, 3].long().clamp(0, 5)[None]
        geom = meta[:, 4:8][None]
        mask = torch.ones((1, len(cls)), dtype=torch.bool, device=dev)
        zs = torch.stack([h(x_cls[None], x_cells[None], slot, view, geom, mask)[0] for h in heads])
        return zs.mean(0), zs, cls16, cells16

    pred_path = out / "predictions.csv"
    fh = open(pred_path, "w", encoding="utf-8", newline="")
    wr = csv.writer(fh)
    wr.writerow([UIDK] + LABELS)
    fallback, stats = {}, {"decode_s": 0., "render_s": 0., "gpu_s": 0., "windows": 0, "wait_s": 0., "rows": 0}
    per_study = []
    if a.dump_features:
        Path(a.dump_features).mkdir(parents=True, exist_ok=True)

    def handle(item):
        uid = item["uid"]
        if item["error"] or not len(item["images"]):
            fallback[uid] = item["error"] or "no usable window"
            return
        tg = time.perf_counter()
        try:
            cls, cells = encode(item["images"])
            zmean, zs, cls16, cells16 = score(cls, cells, item["meta"])
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
        stats["gpu_s"] += dt
        stats["windows"] += len(cls)
        per_study.append([uid, len(cls), item["seconds"][0], item["seconds"][1], round(dt, 3)])
        if a.dump_features:
            np.savez(os.path.join(a.dump_features, hashlib.sha256(uid.encode()).hexdigest()[:24] + ".npz"), uid=uid,
                     cls=cls16.cpu().numpy(), cells=cells16.cpu().numpy(), meta=item["meta"].numpy(),
                     logits=zs.float().cpu().numpy(), series=np.array(item["series"]))

    pos = 0
    timed_out = False
    while pos < len(uids) and not timed_out:
        ds = Studies(a.data, a.split, uids[pos:], tmp, a.code, lean=not a.stock_native)
        # spawn: the same start method as the Windows runs this was tested with (no fork-after-CUDA state);
        # timeout: a stuck decode must surface as an error for that study, never as a silent hang
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
                                      "gpu_s": round(stats["gpu_s"], 1), "windows": stats["windows"]}), flush=True)
                if a.deadline and time.time() > a.deadline:
                    timed_out = True
                    break
                mark = time.perf_counter()
        except Exception as exc:                       # a loader worker died: skip that study, restart after it
            if pos < len(uids):
                fallback[uids[pos]] = "loader: " + repr(exc)[:200]
            pos += 1
        del loader
    for u in uids[pos:]:
        fallback[u] = "not reached before the deadline"
    fh.close()
    shutil.rmtree(tmp, ignore_errors=True)
    receipt = {"status": "PASS", "arm": a.arm, "heads": head_names, "studies": len(uids), "scored": stats["rows"],
               "fallback": fallback, "timed_out": timed_out, "amp": amp, "fp32_retry_batches": retries["fp32_batches"],
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "timm": timm.__version__,
               "seconds": round(time.time() - t_start, 1), "stats": {k: round(v, 1) for k, v in stats.items()},
               "per_study": per_study[:50],
               "native": "stock" if a.stock_native else "lean (header pass, slice-wise quantise)",
               "sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                          for n in ("fe_extract.py", "probe_train.py", "probe_infer.py")},
               "scaffold": "probe_: frozen encoder + light heads"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("WORKER_DONE", json.dumps({k: receipt[k] for k in ("studies", "scored", "timed_out", "seconds", "amp")}),
          "fallback", len(fallback), flush=True)


if __name__ == "__main__":
    main()
