"""Compact fine-tuned image model on the per-slice pixel cache.  The user authorised fine-tuning a compact model on 2026-10-05 evening
(the mainline OrthoFoundation training hold on the 5090 is untouched).  v1.1: fold models are saved.

Same contract as probe_train.py (so probe_eval.py and fuse_eval.py read its outputs unchanged):
  - never reads official labels; gold studies are never trained on; targets come from the same CSVs
  - K-fold over the non-gold pool with the same hash folds; writes gold predictions (mean of K models)
    and out-of-fold predictions for every pool study
  - arms: S_strict, P_presence, E_prior, E_lrprior, or name=bce:<column> / name=lr:<column> from --extra

Model: 2.5D windows (three physically adjacent slices as channels) -> CNN encoder -> per-class attention
over the windows of a study -> 12 logits.  Training samples a fixed number of windows per study; evaluation
uses every third slice as a centre (the mainline's cover sampling) in both views.
"""
import argparse, contextlib, csv, hashlib, json, math, os, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C = len(LABELS)


PREDICT_CHUNK = int(os.environ.get("CT_PREDICT_CHUNK", "512"))   # v7: evaluation windows per encoder call (512 = v6)


def covering_centers(n, phase=1, stride=3):
    if stride <= 1:
        return list(range(n))
    c = list(range(min(phase, n - 1), n, stride))
    if c[0] > 1:
        c.insert(0, 0)
    if c[-1] < n - 2:
        c.append(n - 1)
    return c


def phys_step(spacing_mm, unit_mm):
    """slices per ~unit_mm of anatomy: 1 for an ordinary 3-4 mm series, ~9 for a 0.4 mm 3-D stack (unit_mm <= 0: always 1)."""
    if unit_mm <= 0:
        return 1
    return max(1, int(round(unit_mm / max(float(spacing_mm), 0.1))))


def window_rows(blocks, meta_of, mode, slices, eval_stride=3, cover_stride=3, phys_mm=0.0, rng=None, slice_jitter=0.0,
                train_sampling="cover"):
    """The window list of one study.  blocks: [{'n_slices': n, 'views': [(file, view_id, start), ...]}]; meta_of(file, row)
    -> the px meta row of that window (column 5 = log1p(slice spacing in mm)).  Returns tuples
    (file, centre row, row of slice -r, ..., row of slice +r).  'train' covers every (series, view) at stride
    cover_stride x phys_step with a random phase (and takes the K neighbours two steps apart with probability slice_jitter);
    'eval' covers at stride eval_stride x phys_step.  phys_mm = 0 reproduces the old per-slice behaviour."""
    cand = []
    r_ = slices // 2
    for b in blocks:
        n = b["n_slices"]
        for f, view_id, start in b["views"]:
            ps = phys_step(math.expm1(float(meta_of(f, start)[5])), phys_mm)
            if mode == "train":
                if train_sampling == "cover":
                    st = cover_stride * ps
                    ks = covering_centers(n, int(rng.integers(0, st)), st)
                else:
                    ks = range(n)
            else:
                ks = covering_centers(n, 1, eval_stride * ps)
            for k in ks:
                step_ = ps * (2 if (mode == "train" and slice_jitter > 0 and rng.random() < slice_jitter) else 1)
                cand.append((f, start + k) + tuple(start + min(n - 1, max(0, k + d * step_)) for d in range(-r_, r_ + 1)))
    return cand


class Studies(torch.utils.data.Dataset):
    """Returns uint8 windows [W,3,S,S] and meta [W,6] for one study.  mode 'train' samples, 'eval' covers."""

    def __init__(self, px_dir, uids, n_win, mode, seed, eval_stride=3, train_sampling="uniform", cover_stride=3, px_extra=None, slices=3,
                 slice_jitter=0.0, phys_mm=0.0, px_override=None, extent_meta=False):
        self.dir, self.uids, self.n_win, self.mode, self.seed = px_dir, uids, n_win, mode, seed
        self.extent_meta = bool(extent_meta)
        self.eval_stride, self.train_sampling, self.cover_stride = eval_stride, train_sampling, cover_stride
        self.slices = int(slices)
        self.slice_jitter = float(slice_jitter)
        self.phys_mm = float(phys_mm)
        assert self.slices % 2 == 1 and self.slices >= 1
        index = json.load(open(os.path.join(px_dir, "index.json"), encoding="utf-8"))
        # blocks carry (file, view_id, start): file 0 = the main cache, file 1 = the add-on cache (e.g. the v2 patellofemoral
        # view built with px_cache --views v2add); a study missing from the add-on simply has no add-on windows
        self.blocks = {u: [dict(b, views=[(0, v, st) for v, st in b["views"]]) for b in index["blocks"][u]] for u in uids}
        self.dirs = [px_dir]
        if px_extra:
            extra = json.load(open(os.path.join(px_extra, "index.json"), encoding="utf-8"))
            self.dirs.append(px_extra)
            n_add = 0
            for u in uids:
                eb = extra["blocks"].get(u)
                if not eb:
                    continue
                main = self.blocks[u]
                if [b["series"] for b in eb] != [b["series"] for b in main]:
                    raise SystemExit("add-on cache series order differs from the main cache for study " + u)
                for b, e in zip(main, eb):
                    b["views"] = b["views"] + [(1, v, st) for v, st in e["views"]]
                    n_add += len(e["views"])
            print("add-on cache %s: %d extra (series, view) blocks for %d studies" % (px_extra, n_add, len(uids)), flush=True)
        if px_override:
            # override cache (sidefix): the listed studies are read entirely from it (same format, own row space), e.g.
            # the studies whose knee side the image check corrected, re-rendered through the inference path
            if px_extra:
                raise SystemExit("--px-override cannot be combined with --px-extra")
            ov = json.load(open(os.path.join(px_override, "index.json"), encoding="utf-8"))
            f_ov = len(self.dirs)
            self.dirs.append(px_override)
            n_ov = 0
            for u in uids:
                ob = ov["blocks"].get(u)
                if ob is not None:
                    self.blocks[u] = [dict(b, views=[(f_ov, v, st) for v, st in b["views"]]) for b in ob]
                    n_ov += 1
            print("override cache %s: %d of %d studies read from it" % (px_override, n_ov, len(uids)), flush=True)
        self.px = self.meta = None
        self.epoch = 0

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        if self.px is None:
            self.px = [np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r") for d in self.dirs]
            self.meta = [np.load(os.path.join(d, "meta.f32.npy"), mmap_mode="r") for d in self.dirs]
        uid = self.uids[i]
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + i) if self.mode == "train" else None
        # 'cover': every series/view is covered at stride cover_stride (x the physical step) with a random phase, so a
        # focal finding spanning >= one stride is in the sampled set of every epoch (uniform sampling of ~10% of the
        # windows misses a 3-slice finding in about half of the epochs).  Rows: (file, centre row, K slice rows k-r..k+r).
        cand = window_rows(self.blocks[uid], lambda f, r: self.meta[f][r], self.mode, self.slices, self.eval_stride,
                           self.cover_stride, self.phys_mm, rng, self.slice_jitter, self.train_sampling)
        if self.mode == "train" and len(cand) > self.n_win:
            cand = [cand[j] for j in np.sort(rng.choice(len(cand), self.n_win, replace=False))]
        rows = np.array(cand, np.int64)
        S = self.px[0].shape[-1]
        img = np.empty((len(rows), self.slices, S, S), np.uint8)
        m = np.empty((len(rows), self.meta[0].shape[1]), np.float32)
        for f in range(len(self.dirs)):
            sel = np.flatnonzero(rows[:, 0] == f)
            if not len(sel):
                continue
            r = rows[sel]
            img[sel] = np.stack([np.asarray(self.px[f][r[:, 2 + d]]) for d in range(self.slices)], 1)
            m[sel] = np.asarray(self.meta[f][r[:, 1]])
        meta = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1)   # slot, view, rel_pos, log_spacing, extent, fluid
        if self.extent_meta:                                   # + series index, plane (0 sag / 1 cor / 2 ax), slices in the series
            meta = np.concatenate([meta, np.stack([m[:, 0], m[:, 2], m[:, 10]], 1)], 1)
        return torch.from_numpy(img), torch.from_numpy(meta.astype(np.float32)), i


def collate(batch):
    W = max(len(b[0]) for b in batch)
    S = batch[0][0].shape[-1]
    img = torch.zeros((len(batch), W, batch[0][0].shape[1], S, S), dtype=torch.uint8)
    meta = torch.zeros((len(batch), W, batch[0][1].shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(batch), W), dtype=torch.bool)
    for j, (x, m, _) in enumerate(batch):
        img[j, :len(x)] = x
        meta[j, :len(x)] = m
        mask[j, :len(x)] = True
    return img, meta, mask, torch.tensor([b[2] for b in batch])


OFL_BACKBONE = "ofl_l"


class OFLEncoder(nn.Module):
    """OrthoFoundation-L (public self-supervised weights) as the window encoder, run exactly as the feature extractor
    runs it: timm Eva vit_large_patch16_dinov3_qkvb, RoPE removed, 5 prefix tokens, ImageNet normalisation (done by
    Net).  Feature = [CLS token, mean patch token] (2048).  The patch embedding, the tokens, norm_pre and the first
    `freeze` of the 24 blocks are frozen and run without a graph; the remaining blocks and the final norm are trained.
    `deploy_state_dict` returns only what differs from the public weights, so a fold model can be shipped small and
    rebuilt on top of the public weights."""

    def __init__(self, init, freeze=16, img_size=256):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False, num_classes=0, global_pool="",
                                     img_size=img_size)
        if init:
            if str(init).endswith(".safetensors"):
                from safetensors.torch import load as load_bytes
                with open(str(init), "rb") as fh:          # one sequential read; the mmap loader page-faults 4 KB at a time
                    sd = load_bytes(fh.read())             # (0.4 MB/s from a USB disk: 48 min per fold on the 5090)
            else:
                sd = torch.load(init, map_location="cpu")
            self.vit.load_state_dict(sd, strict=True)
        self.vit.rope = None
        self.freeze = int(freeze)
        self.num_features = 2 * self.vit.embed_dim
        self.frozen_keys = set()
        frozen_modules = [self.vit.patch_embed, self.vit.norm_pre] + list(self.vit.blocks[: self.freeze])
        for mod in frozen_modules:
            for p in mod.parameters():
                p.requires_grad_(False)
        for name in ("cls_token", "reg_token", "pos_embed"):
            t = getattr(self.vit, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad_(False)
        for k, p in self.vit.named_parameters():
            if not p.requires_grad:
                self.frozen_keys.add("vit." + k)

    def trunk(self, x):
        v = self.vit
        with torch.no_grad():
            x = v.patch_embed(x)
            x, _ = v._pos_embed(x)
            x = v.norm_pre(x)
            for blk in v.blocks[: self.freeze]:
                x = blk(x, rope=None)
        return x

    def tail(self, x):
        v = self.vit
        for blk in v.blocks[self.freeze:]:
            x = blk(x, rope=None)
        x = v.norm(x)
        return torch.cat([x[:, 0], x[:, v.num_prefix_tokens:].mean(1)], -1)

    def forward(self, x):
        return self.tail(self.trunk(x))


class Net(nn.Module):
    def __init__(self, backbone, init, hidden=384, drop=0.1, freeze_blocks=16, crop=256, state_spec=None, in_chans=3, spatial=False,
                 extent=False):
        super().__init__()
        import timm
        self.backbone = backbone
        self.crop = int(crop)
        self.in_chans = int(in_chans)
        self.spatial = bool(spatial)
        self.extent = bool(extent)
        self._ext_area = None
        if self.spatial and backbone == OFL_BACKBONE:
            raise SystemExit("--spatial-attn needs a timm backbone (the OFL encoder returns pooled features)")
        if backbone == OFL_BACKBONE:
            assert self.in_chans == 3, "the OFL patch embedding is a public 3-channel weight"
            self.enc = OFLEncoder(init, freeze=freeze_blocks, img_size=crop)
        else:
            self.enc = self._timm_encoder(backbone, init, self.in_chans)
        self.has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.enc.modules())
        self._build_head(hidden, drop)
        self.state_spec = state_spec
        self.has_state = state_spec is not None
        if self.has_state:
            self._build_state_head(hidden, state_spec)
        if self.spatial:                        # created last: every baseline parameter keeps the baseline's initial values
            d = self.enc.num_features
            self.sp_key = nn.Linear(d, 128)     # keys from per-position normalised features
            self.sp_pos = nn.Linear(16, 128, bias=False)
            nn.init.zeros_(self.sp_pos.weight)
            self.sp_q = nn.Parameter(torch.zeros(C, 128))   # zero query = uniform weights = the baseline average
        if self.extent:                         # OA defect extent (after every other parameter: same init as the baseline)
            self.ext_u = nn.Parameter(torch.zeros(C, hidden))
            self.ext_b = nn.Parameter(torch.full((C,), -4.0))   # sigmoid(-4) = 0.018: area starts ~4 mm^2, not ~3,000 (audit 3)
            self.ext_gamma = nn.Parameter(torch.zeros(C))    # zero = the baseline logit at init
            self.ext_view = nn.Parameter(torch.zeros(C, 6))  # per (class, view) weight inside a series; 0 = uniform (audit 4)
            em = torch.zeros(C)
            for l in ("Medial OA", "Lateral OA", "PF OA"):
                em[LABELS.index(l)] = 1.0
            self.register_buffer("ext_mask", em)

    def _build_state_head(self, hidden, spec):
        """Report-state heads: for class c a K_c-way read-out of the class token (which report state the images show),
        OA-sign / acuity binaries on the parent class token, and the frozen prior vector per state (mean report prior of the state's category over the pool, not fitted on gold).
        The class logit gains  gate_c * (logit(sum_s p(s|x) prior_c(s)) - logit(base_c)): the 'align to the official
        label' mapping applied to predicted states; gate starts at 0 so the direct path is untouched at init."""
        cats = spec["categories"]
        self.state_k = [len(cats[l]) for l in LABELS]
        kmax = max(self.state_k)
        self.state_w = nn.Parameter(torch.randn(C, kmax, hidden) * .02)
        self.state_b = nn.Parameter(torch.zeros(C, kmax))
        prior = torch.full((C, kmax), float("nan"))
        valid = torch.zeros((C, kmax), dtype=torch.bool)
        for i, l in enumerate(LABELS):
            for j, pv in enumerate(spec["prior"][l]):
                prior[i, j] = float(pv); valid[i, j] = True
        self.register_buffer("state_prior", torch.nan_to_num(prior, nan=0.5).clamp(1e-3, 1 - 1e-3))
        self.register_buffer("state_valid", valid)
        self.register_buffer("state_base_logit", torch.logit(torch.tensor([float(spec["base"][l]) for l in LABELS]).clamp(1e-3, 1 - 1e-3)))
        self.state_gate = nn.Parameter(torch.zeros(C))
        self.bin_names = [(b, l) for b, labs in spec["binaries"].items() for l in labs]
        self.bin_parent = [LABELS.index(l) for _, l in self.bin_names]
        self.bin_w = nn.Parameter(torch.randn(len(self.bin_names), hidden) * .02)
        self.bin_b = nn.Parameter(torch.zeros(len(self.bin_names)))

    def state_logits(self, study):
        """study: [B, C, hidden] class tokens -> state logits [B, C, Kmax] (invalid slots -inf), binary logits [B, nbin]."""
        sl = torch.einsum("bch,ckh->bck", study, self.state_w).float() + self.state_b
        sl = sl.masked_fill(~self.state_valid[None], float("-inf"))
        bl = torch.einsum("bjh,jh->bj", study[:, self.bin_parent], self.bin_w).float() + self.bin_b
        return sl, bl

    def state_evidence(self, sl):
        """logit of the official probability implied by the predicted state distribution, relative to the class base rate."""
        p = sl.softmax(-1)                                          # -inf slots -> 0
        e = (p * self.state_prior[None]).sum(-1).clamp(1e-3, 1 - 1e-3)
        return torch.logit(e) - self.state_base_logit[None]

    def deploy_state_dict(self):
        """Everything except the frozen public OFL weights (identical to state_dict() for other backbones)."""
        frozen = {"enc." + k for k in getattr(self.enc, "frozen_keys", set())}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k not in frozen}

    def load_deploy_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        frozen = {"enc." + k for k in getattr(self.enc, "frozen_keys", set())}
        if unexpected or set(missing) - frozen:
            raise ValueError("deploy state does not match: missing %s unexpected %s" % (sorted(set(missing) - frozen)[:5], unexpected[:5]))

    def _timm_encoder(self, backbone, init, in_chans=3):
        import timm
        # init == "timm": public ImageNet weights from the local hub cache (HF_HOME); a path: a state dict file
        is_vit = backbone.startswith("vit_")
        kw = dict(pretrained=(init == "timm"), num_classes=0, in_chans=in_chans,
                  drop_path_rate=0.1 if ("convnext" in backbone or is_vit) else 0.0)
        if is_vit:
            kw["dynamic_img_size"] = True       # keeps the pretrained grid (DINOv2 518 / DINOv3 256); position embeddings or RoPE
        else:                                   # are interpolated to the crop at forward; pooling = the model's own (token / avg)
            kw["global_pool"] = "avg"
        if backbone.startswith(("coatnet", "maxvit", "maxxvit")):
            kw["img_size"] = self.crop          # relative-position tables are built for this input size (CoAtNet / MaxViT)
        if self.spatial:                        # keep the spatial grid: CNN -> normed map [N, d, h, w], ViT -> all tokens [N, T, d]
            kw["global_pool"] = ""
        enc = timm.create_model(backbone, **kw)
        if init and init != "timm":
            if str(init).endswith(".safetensors"):
                from safetensors.torch import load as load_bytes
                with open(str(init), "rb") as fh:
                    sd = load_bytes(fh.read())
            else:
                sd = torch.load(init, map_location="cpu")
                sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
            if in_chans != 3:                       # inflate the pretrained RGB stem: mean over RGB, replicated K times, scaled 3/K
                for k, w in list(sd.items()):
                    if w.ndim == 4 and w.shape[1] == 3 and k in enc.state_dict() and enc.state_dict()[k].shape[1] == in_chans:
                        sd[k] = w.mean(1, keepdim=True).repeat(1, in_chans, 1, 1) * (3.0 / in_chans)
                        print("inflated %s %s -> %s" % (k, tuple(w.shape), tuple(sd[k].shape)), flush=True)
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            print("init %s: missing %d unexpected %d" % (init, len(missing), len(unexpected)), flush=True)
        self.sp_head_norm = None
        if self.spatial and isinstance(getattr(getattr(enc, "head", None), "norm", None), nn.Module) \
                and not isinstance(enc.head.norm, nn.Identity):
            # CNN heads pool THEN normalise: keep that order per class (pool the raw map, then this norm on each class's
            # pooled vector), so a zero spatial query reproduces the baseline feature LN(mean_p x_p) exactly
            self.sp_head_norm = enc.head.norm
            enc.head.norm = nn.Identity()
        return enc

    def _build_head(self, hidden, drop):
        d = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden))
        self.slot = nn.Embedding(10, hidden)
        self.view = nn.Embedding(6, hidden)
        self.geom = nn.Linear(4, hidden, bias=False)
        for m in (self.slot, self.view):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.geom.weight)
        self.drop = nn.Dropout(drop)
        self.win_proj = nn.Linear(hidden, hidden)
        self.q_win = nn.Parameter(torch.randn(C, hidden) * .02)
        self.out_norm = nn.LayerNorm(hidden)
        self.cls_w = nn.Parameter(torch.randn(C, hidden) * .02)
        self.bias = nn.Parameter(torch.zeros(C))
        if self.in_chans == 3:
            self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        else:                                       # grey slices: one ImageNet-mean/std per channel
            self.register_buffer("mean", torch.full((1, self.in_chans, 1, 1), .449))
            self.register_buffer("std", torch.full((1, self.in_chans, 1, 1), .226))

    def spatial_pool(self, x):
        """Encoder output with its spatial grid ([N, d, h, w] CNN map or [N, T, d] ViT tokens) -> per-class window features
        [N, C, d]: class c weighs the window's positions with its own query (sp_q[c]; zero at init = the plain average)
        over keys tanh(W f(p) + W_pos pe(p)), so a class can read its own structure inside a window shared with others."""
        if x.ndim == 4:
            h, w = x.shape[-2:]
            f = x.flatten(2).transpose(1, 2)                                      # [N, h*w, d]
        else:
            f = x[:, getattr(self.enc, "num_prefix_tokens", 0):]
            h = w = int(round(f.shape[1] ** 0.5))
            assert h * w == f.shape[1], "non-square ViT token grid"
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device), indexing="ij")
        fr = torch.tensor([1., 2., 4., 8.], device=f.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * fr, xx.reshape(-1, 1) * fr
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)              # [h*w, 16] fixed 2-D position features
        kin = F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype)                # keys: per-position normalised features
        k = torch.tanh(self.sp_key(kin) + self.sp_pos(pe.to(f.dtype))[None])    # [N, P, 128]
        att = torch.einsum("npk,ck->npc", k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum("npc,npd->ncd", att, f)                                 # [N, C, d]
        if self.sp_head_norm is not None:                                        # CNN: the head norm after pooling, per class
            n_ = self.sp_head_norm
            g = F.layer_norm(g.float(), (g.shape[-1],), n_.weight, n_.bias, n_.eps).to(f.dtype)
        return g

    def extent_area(self, e, meta, mask):
        """e [B, W, C] window defect logits -> A [B, C] (mm^2, an area proxy): inside each series the mean sigmoid(e) of
        each view's windows, combined over the views present by a learned per-(class, view) softmax (uniform at init: a
        compartment sits inside only some of the locals), times the series' slab thickness (slices x spacing) = mm of
        slab with defect evidence; per plane the max over series; then sagittal x coronal (tibiofemoral OA) or axial x
        sagittal (PF OA); other classes 0."""
        V = 6
        sid = meta[..., 6].long()
        # one slot per series, never merged: 16 covers every training study (at most 14 series); a deployed study with more
        # series gets more slots (lossless -- an empty slot has mean 0 and the per-plane max runs over values >= 0)
        S_MAX = max(16, int(sid.max()) + 1)
        vid = meta[..., 1].long().clamp(0, V - 1)
        q = torch.sigmoid(e.float()) * mask[..., None]
        ohv = F.one_hot(sid * V + vid, S_MAX * V).float() * mask[..., None]     # [B, W, S*V]
        cntv = ohv.sum(1)                                                          # [B, S*V]
        mean_v = (torch.einsum("bwk,bwc->bkc", ohv, q) / cntv.clamp_min(1)[..., None]).view(len(e), S_MAX, V, -1)
        present = (cntv > 0).view(len(e), S_MAX, V, 1)
        wv = self.ext_view.float().t()[None, None].masked_fill(~present, -1e4).softmax(2) * present   # [B, S, V, C]
        mean = (wv * mean_v).sum(2)                                                # [B, S, C]; 0 for a series without windows
        oh = F.one_hot(sid, S_MAX).float() * mask[..., None]                     # [B, W, S]
        cnt = oh.sum(1)                                                            # [B, S]
        span = torch.expm1(meta[..., 3].float()).clamp_min(0) * meta[..., 8].float()   # slab thickness (mm) of each window's series
        span_s = torch.einsum("bws,bw->bs", oh, span) / cnt.clamp_min(1)
        plane_s = (torch.einsum("bws,bw->bs", oh, meta[..., 7].float()) / cnt.clamp_min(1)).round()
        E = mean * span_s[..., None]                                               # [B, S, C] mm of slab with defect evidence
        per_plane = []
        for pl in (0, 1, 2):
            ok = ((plane_s == pl) & (cnt > 0))[..., None]
            per_plane.append(E.masked_fill(~ok, 0.0).amax(1))                     # [B, C]; 0 when the plane is absent
        sag, cor, ax = per_plane
        area = torch.zeros_like(sag)
        for l in ("Medial OA", "Lateral OA"):
            c = LABELS.index(l)
            area[:, c] = sag[:, c] * cor[:, c]
        c = LABELS.index("PF OA")
        area[:, c] = ax[:, c] * sag[:, c]
        return area

    def head(self, tok, meta, mask, return_aux=False):
        """Encoder output of the real windows [N, d] (or [N, C, d] with --spatial-attn) -> class logits [B, C] (and the
        state/binary logits when asked)."""
        B, W = mask.shape
        f = self.proj(tok.float())
        m = meta
        if self.spatial:                                                          # class-specific window tokens [B, W, C, H]
            emb = self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = f.new_zeros((B, W) + tuple(f.shape[1:]))
            t[mask] = f
            t = self.drop(t + emb[:, :, None, :])
            s2 = torch.einsum("bwch,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float("-inf"))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum("bwc,bwch->bch", a2, t))
            if self.extent:
                e_win = (t.float() * self.ext_u[None, None]).sum(-1) + self.ext_b
        else:
            t = f.new_zeros((B, W, f.shape[-1]))
            t[mask] = f
            t = t + self.slot(m[..., 0].long().clamp(0, 9)) + self.view(m[..., 1].long().clamp(0, 5)) + self.geom(m[..., 2:6])
            t = self.drop(t)
            s2 = torch.einsum("bwh,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
            s2 = s2.masked_fill(~mask[:, :, None], float("-inf"))
            a2 = s2.softmax(1).to(t.dtype)
            study = self.out_norm(torch.einsum("bwc,bwh->bch", a2, t))
            if self.extent:
                e_win = torch.einsum("bwh,ch->bwc", t.float(), self.ext_u) + self.ext_b
        z = (study * self.cls_w).sum(-1).float() + self.bias
        if self.extent:
            area = self.extent_area(e_win, meta, mask)
            self._ext_area = area
            z = z + self.ext_gamma * self.ext_mask * torch.log1p(area / 100.0)
        if not self.has_state:
            return (z, None, None) if return_aux else z
        sl, bl = self.state_logits(study)
        z = z + self.state_gate[None] * self.state_evidence(sl)
        return (z, sl, bl) if return_aux else z

    def forward(self, img, meta, mask, chunk=0, return_aux=False):
        x = img[mask]                                               # only real windows go through the encoder
        n = len(x)
        if n % 32 and (not self.training or not self.has_bn):     # pad the encoder batch to a multiple of 32: a handful of shapes
            x = torch.cat([x, x.new_zeros((32 - n % 32,) + tuple(x.shape[1:]))])   # instead of one per step (allocator, cuDNN autotune);
        feats = []
        step = chunk or len(x)
        for s in range(0, len(x), step):
            xs = x[s:s + step].float().div_(255.)
            xs = ((xs - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            e = self.enc(xs)
            feats.append(self.spatial_pool(e) if self.spatial else e)
        return self.head(torch.cat(feats)[:n], meta, mask, return_aux=return_aux)


def bce_soft(z, target, weight):
    loss = F.binary_cross_entropy_with_logits(z, target, reduction="none") * weight
    den = weight.sum(0)
    ok = den > 0
    return (loss.sum(0)[ok] / den[ok]).mean()


def lr_loss(z, lr):
    return (-torch.logaddexp(F.logsigmoid(z) + torch.log(lr), F.logsigmoid(-z))).mean(0).mean()


def state_loss(sl, bl, st_idx, st_w, bin_t):
    """Masked cross-entropy over report states (per class, weight-normalised) + masked BCE over the binaries.
    st_idx [B, C] long (-1 unknown), st_w [B, C], bin_t [B, nbin] (nan unknown)."""
    total, n = sl.new_zeros(()), 0
    for c in range(sl.shape[1]):
        ok = st_idx[:, c] >= 0
        if ok.any():
            ce = F.cross_entropy(sl[ok, c], st_idx[ok, c], reduction="none") * st_w[ok, c]
            total = total + ce.sum() / st_w[ok, c].sum().clamp_min(1e-6); n += 1
    known = ~torch.isnan(bin_t)
    if known.any():
        t = torch.nan_to_num(bin_t)
        bce = F.binary_cross_entropy_with_logits(bl, t, reduction="none") * known
        per = bce.sum(0) / known.sum(0).clamp_min(1)
        total = total + per[known.any(0)].mean(); n += 1
    return total / max(n, 1)


class PairBatchSampler:
    """Training batches of `batch` dataset indices.  With probability `frac` a batch opens with a matched pair -- a positive
    and a hard negative of one pair-family class that share the co-finding signature (bucket drawn with the given
    probabilities) -- and is filled with base draws; otherwise all draws are base draws (weighted with replacement, as
    the rare-boost sampler).  len = n // batch: the same number of steps per epoch as the plain loader."""

    def __init__(self, n, batch, weights, buckets, frac, seed):
        self.n, self.batch, self.frac = int(n), int(batch), float(frac)
        self.w = torch.ones(self.n, dtype=torch.double) if weights is None else torch.as_tensor(weights, dtype=torch.double)
        self.buckets = buckets
        self.bprob = torch.tensor([b[2] for b in buckets], dtype=torch.double) if buckets else None
        self.generator = torch.Generator().manual_seed(int(seed))
        assert self.batch >= 2

    def __len__(self):
        return self.n // self.batch

    def _base(self, k):
        return torch.multinomial(self.w, k, replacement=True, generator=self.generator).tolist() if k > 0 else []

    def __iter__(self):
        g = self.generator
        for _ in range(len(self)):
            if self.buckets and float(torch.rand(1, generator=g)) < self.frac:
                pos, neg, _ = self.buckets[int(torch.multinomial(self.bprob, 1, generator=g))]
                i = pos[int(torch.randint(0, len(pos), (1,), generator=g))]
                j = neg[int(torch.randint(0, len(neg), (1,), generator=g))]
                yield [int(i), int(j)] + self._base(self.batch - 2)
            else:
                yield self._base(self.batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True)
    ap.add_argument("--px-extra", default=None, help="add-on pixel cache (px_cache --views v2add) read next to --px; its windows join every study")
    ap.add_argument("--px-override", default=None, help="override cache (sidefix): studies listed there are read from it instead of --px")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="E_prior")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--init", default="F:/rsnaknee/pretrained/resnet34-imagenet.pth")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--train-windows", type=int, default=48)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--slices", type=int, default=3, help="adjacent slices per window = input channels (odd; CNN backbones only beyond 3)")
    ap.add_argument("--slice-jitter", type=float, default=0.0, help="probability that a training window uses neighbours two slices apart")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trainable weights (0 = off); EMA weights are evaluated and saved")
    ap.add_argument("--bn-recal-batches", type=int, default=0, help="v7: BatchNorm backbones only (CoAtNet, ResNet): after --ema copies the "
                    "averaged weights in, recompute the BatchNorm running statistics for them over this many training batches "
                    "(centre crop, no augmentation) before evaluating and saving; also applied by --reeval-run.  0 = off "
                    "(backbones without BatchNorm are never touched)")
    ap.add_argument("--include-review-holdout", action="store_true", help="add the 120 review-holdout studies (full report targets, "
                    "fold -3) to the training pool; they were held out for the report-reader evaluation, not for image training")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the timm encoder (keeps 208 windows/study on 32 GB cards)")
    ap.add_argument("--phys-mm", type=float, default=0.0, help="window spacing in mm of anatomy: the covering stride and the K-slice "
                    "neighbour step are multiplied by round(phys_mm / slice spacing), so a 0.4 mm 3-D stack is sampled like a 3.5 mm "
                    "2-D series (0 = per-slice, the old behaviour)")
    ap.add_argument("--rare-boost", action="store_true", help="oversample studies with rare report states (MCL>=2/acute, fracture, high PF OA, Baker>=moderate, contusion); needs --state-targets")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-blocks", type=int, default=16, help="ofl_l backbone: number of leading transformer blocks kept frozen (of 24)")
    ap.add_argument("--llrd", type=float, default=1.0, help="ViT backbones (ofl_l, timm vit_*): layer-wise learning-rate decay per block from the top (1 = off); block i of n gets lr * llrd**(n - 1 - i), embeddings lr * llrd**n")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-workers", type=int, default=3)
    ap.add_argument("--forward-only", action="store_true", help="one batch forward/backward, NO optimizer step, then exit")
    ap.add_argument("--compile", action="store_true", help="torch.compile the encoder in place (dynamic shapes); same maths, 1.4x (ConvNeXt-small) / 1.9x (DINOv3-B) on a PRO 6000; weight names unchanged")
    ap.add_argument("--resume", action="store_true", help="continue every fold from its end-of-epoch checkpoint in <out>/ckpt when present")
    ap.add_argument("--ckpt-every", type=int, default=1, help="write the fold checkpoint every N epochs (0 = never); removed when the fold ends")
    ap.add_argument("--stop-after-epoch", type=int, default=-1, help="testing: exit right after the checkpoint of this epoch")
    ap.add_argument("--eval-stride", type=int, default=3, help="evaluation windows: every k-th slice as centre (3 = the validated default; 1 = every slice)")
    ap.add_argument("--train-sampling", default="uniform", choices=["uniform", "cover"],
                    help="uniform = --train-windows random windows of the study; cover = every series/view at --cover-stride with a random phase (capped at --train-windows)")
    ap.add_argument("--cover-stride", type=int, default=3)
    ap.add_argument("--state-targets", default=None, help="state_targets_<tag>.csv from make_state_targets.py (report-state construct)")
    ap.add_argument("--state-spec", default=None, help="state_spec_<tag>.json (categories, prior vectors, binaries)")
    ap.add_argument("--state-weight", type=float, default=0.5, help="weight of the state/binary losses next to the arm loss")
    ap.add_argument("--spatial-attn", action="store_true", help="per-class spatial attention inside each window (encoder keeps its "
                    "grid; class queries start at zero = plain average), then per-class window attention over class-specific tokens")
    ap.add_argument("--pair-rank", type=float, default=0.0, help="weight of the matched-pair ranking loss on the 8 pair families "
                    "(positive vs hard negative of a class with the same co-finding signature); needs --state-targets; 0 = off")
    ap.add_argument("--pair-frac", type=float, default=0.33, help="share of training batches that open with a matched pair (--pair-rank)")
    ap.add_argument("--seed-base", type=int, default=0, help="first seed (seeds seed_base .. seed_base+seeds-1); a replicate of a recipe")
    ap.add_argument("--extent", action="store_true", help="M4: OA defect extent per series and plane (area proxy) into the OA logits")
    ap.add_argument("--extent-weight", type=float, default=0.2, help="weight of the extent supervision (>= 1 cm^2 stated vs normal/low grade)")
    ap.add_argument("--holdout-fold", type=int, default=None,
                    help="clean ruler fold for pseudo-label rounds: this hash fold is excluded from EVERY training set (no model is trained "
                         "for it); its predictions are the mean of the other folds' models, so a teacher->student chain evaluated on it "
                         "with local_ruler --folds-only is leak-free")
    ap.add_argument("--reeval-run", default=None, help="predict-only: load models/<arm>_s<seed>_f<k>.pt from this run directory and rewrite the out-of-fold "
                    "predictions into --out with the current --eval-stride (no training)")
    ap.add_argument("--aug", default="crop", choices=["crop", "v2"], help="v2 = crop + per-study rotation/scale + gamma/contrast jitter (no flips: laterality)")
    ap.add_argument("--aug-res", type=float, default=0.0, help="v2 only: probability that a training study is down-sampled in-plane by "
                    "1.25-2x and up-sampled back (simulates 0.33-0.5 mm scanners); 0 = off, random stream unchanged")
    ap.add_argument("--only-folds", default=None, help="diagnostic: comma list of folds to train (others skipped); outputs are written with a partial_ prefix and must not be used as OOF")
    ap.add_argument("--check-targets", action="store_true", help="pre-launch check on CPU: build the pool and every arm's targets and the "
                    "state targets with the training code path, print a summary, and exit (no data loading, no model, no GPU)")
    ap.add_argument("--i-have-user-authorisation-to-finetune", action="store_true")
    a = ap.parse_args()
    if not a.forward_only and not a.check_targets and not a.i_have_user_authorisation_to_finetune:
        raise SystemExit("refusing to update an encoder: the user has only authorised light heads on cached features. "
                         "Use --forward-only, or pass the authorisation flag after the user says so.")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cpu") if a.check_targets else torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    index = json.load(open(os.path.join(a.px, "index.json"), encoding="utf-8"))
    done = {u for u, v in index["done"].items() if v[1] > 0}
    rows = list(csv.DictReader(open(a.targets, encoding="utf-8")))
    by_uid = {r["StudyInstanceUID"]: r for r in rows}
    if a.extra:
        for r in csv.DictReader(open(a.extra, encoding="utf-8")):
            if r["StudyInstanceUID"] in by_uid:
                by_uid[r["StudyInstanceUID"]].update({k: v for k, v in r.items() if k != "StudyInstanceUID"})
    gold = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]
    # a non-gold study whose report labels were blanked (e.g. the report describes another body part) carries no target:
    # it leaves the pool, exactly like a gold row (logged below; bad_report_studies.json lists the blanked studies)
    blanked = {r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and any(r.get(L + "|prior", "") == "" for L in LABELS)}
    pool = sorted(r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and r["StudyInstanceUID"] in done
                  and r["StudyInstanceUID"] not in blanked
                  and ((r["review_holdout"] == "0" and int(r["fold"]) != -3) or a.include_review_holdout))
    assert not (set(pool) & set(gold))
    for u in gold:
        assert all(by_uid[u][L + "|strict"] == "" and by_uid[u][L + "|prior"] == "" for L in LABELS)
    K = a.folds
    pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])
    print("pool %d gold %d folds %s | non-gold studies with blanked report labels left out: %d" % (
        len(pool), len(gold), np.bincount(pool_fold, minlength=K).tolist(), len(blanked & done)), flush=True)

    def col(uids, suffix, default):
        out = np.full((len(uids), C), default, np.float32)
        for i, u in enumerate(uids):
            for j, L in enumerate(LABELS):
                v = by_uid[u].get(L + "|" + suffix, "")
                if v != "":
                    out[i, j] = float(v)
        return torch.from_numpy(out).to(dev)

    specs = {}
    for item in a.arms.split(","):
        if "=" in item:
            name, spec = item.split("=", 1)
            specs[name] = tuple(spec.split(":"))

    def arm_tensors(arm):
        if arm == "S_strict":
            return "bce", col(pool, "strict", 0.), col(pool, "strict_w", 0.)
        if arm == "P_presence":
            return "bce", col(pool, "presence", 0.), col(pool, "presence_w", 0.)
        spec = {"E_prior": ("bce", "prior"), "E_lrprior": ("lr", "lr_prior")}.get(arm) or specs[arm]
        kind, column = spec[0], spec[1]
        if len(spec) > 2 and spec[2]:                       # name=kind:column:weight_column  (0 = cell carries no information)
            w = col(pool, spec[2], 0.)
            t = col(pool, column, 0.)
            return kind, t, w
        t = col(pool, column, float("nan"))
        assert not torch.isnan(t).any()
        return kind, t, torch.ones_like(t)

    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))

    state_spec = None
    if a.state_targets:
        state_spec = json.load(open(a.state_spec, encoding="utf-8"))
        srows = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(a.state_targets, encoding="utf-8"))}
        st_idx = torch.full((len(pool), C), -1, dtype=torch.long)
        st_w = torch.zeros((len(pool), C))
        bin_names = [(b, l) for b, labs in state_spec["binaries"].items() for l in labs]
        bin_t = torch.full((len(pool), len(bin_names)), float("nan"))
        for i, u in enumerate(pool):
            r = srows.get(u)
            if r is None:
                continue
            for c, l in enumerate(LABELS):
                v = r.get(l + "|state", "")
                if v != "":
                    st_idx[i, c] = int(v); st_w[i, c] = float(r.get(l + "|sw") or 1.0)
            for j, (b, l) in enumerate(bin_names):
                v = r.get(l + "|" + b, "")
                if v != "":
                    bin_t[i, j] = float(v)
        st_idx, st_w, bin_t = st_idx.to(dev), st_w.to(dev), bin_t.to(dev)
        print("state targets: known cells per class %s | binaries known %s" % (
            (st_idx >= 0).sum(0).tolist(), (~torch.isnan(bin_t)).sum(0).tolist()), flush=True)

    # ---- P1 pair families: class, positive states, hard-negative states, co-finding variables (from the report states)
    pair_fam, pair_info = [], {}
    if a.pair_rank > 0:
        if state_spec is None:
            raise SystemExit("--pair-rank needs --state-targets")
        cats = state_spec["categories"]
        Li = {l: c for c, l in enumerate(LABELS)}
        bni = {(b, l): j for j, (b, l) in enumerate(bin_names)}
        sti, bint = st_idx.cpu().numpy(), bin_t.cpu().numpy()

        def sset(l, *names):
            return [cats[l].index(x) for x in names]

        def inset(l, idx_):                     # 1 / 0 per study, -1 when the report is silent on l
            v = sti[:, Li[l]]
            return np.where(v < 0, -1, np.isin(v, idx_).astype(np.int64))

        def osteo(l):                           # "osteophytes written or not" (the table has 0 only for a normal compartment)
            v = bint[:, bni[("sign_osteophytes", l)]]
            return (v == 1).astype(np.int64)

        TORN_M, HIGH = ("tear", "macerated_or_destroyed"), ("high_grade_size_unstated", "high_grade_large")
        acl_t = inset("ACL", sset("ACL", "partial_tear_unspecified", "tear_unspecified", "complete_tear"))
        mm_t = inset("Medial Meniscus", sset("Medial Meniscus", *TORN_M))
        lm_t = inset("Lateral Meniscus", sset("Lateral Meniscus", *TORN_M))
        hi = {l: inset(l, sset(l, *HIGH)) for l in ("Medial OA", "Lateral OA", "PF OA")}
        cont = inset("Contusion", sset("Contusion", "contusion_or_bruise", "traumatic_marrow_edema"))
        ev = sti[:, Li["Effusion"]]
        eff_b = np.select([ev < 0, ev == 0, ev == 1, np.isin(ev, [2, 3])], [-1, 0, 1, 2], 3)
        fams = [("Lateral Meniscus", TORN_M, ("normal", "degeneration_no_tear"), [acl_t, mm_t, hi["Lateral OA"]]),
                ("Medial Meniscus", TORN_M, ("normal", "degeneration_no_tear"), [acl_t, lm_t, hi["Medial OA"]]),
                ("Lateral OA", HIGH, ("normal", "low_grade_chondral"), [osteo("Lateral OA"), hi["Medial OA"], hi["PF OA"]]),
                ("Medial OA", HIGH, ("normal", "low_grade_chondral"), [osteo("Medial OA"), hi["Lateral OA"], hi["PF OA"]]),
                ("PF OA", HIGH, ("normal", "low_grade_chondral"), [osteo("PF OA"), hi["Medial OA"], hi["Lateral OA"]]),
                ("Synovitis", ("synovitis", "synovial_thickening"), ("other_synovial_finding", "none"), [eff_b]),
                ("MCL", ("partial_tear_or_grade2", "high_grade_partial_tear", "complete_tear"),
                 ("normal", "periligamentous_edema_or_grade1", "chronic_or_healed_change"), [mm_t, cont, acl_t]),
                ("ACL", ("partial_tear_unspecified", "tear_unspecified", "complete_tear"),
                 ("normal", "signal_or_degeneration", "low_grade_sprain_or_partial"), [cont, lm_t, mm_t])]
        role = np.zeros((len(pool), C), np.int64)            # +1 positive / -1 hard negative of the class's family / 0
        sig = np.zeros((len(pool), C), np.int64)             # co-finding signature: pairs only within one signature
        pw = np.zeros((len(pool), C), np.float32)             # report prior of the study's state (pair weight)
        for l, pos_s, neg_s, cof in fams:
            c = Li[l]
            v = sti[:, c]
            role[np.isin(v, sset(l, *pos_s)), c] = 1
            role[np.isin(v, sset(l, *neg_s)), c] = -1
            unknown_cof = np.zeros(len(pool), bool)
            for arr in cof:
                unknown_cof |= np.asarray(arr) < 0
            role[unknown_cof, c] = 0                 # an unknown co-finding cannot be matched: the study leaves this family
            code = np.zeros(len(pool), np.int64)
            for arr in cof:
                code = code * 8 + (np.asarray(arr, np.int64) + 1)
            sig[:, c] = code
            ok = v >= 0
            pw[ok, c] = np.asarray(state_spec["prior"][l], np.float32)[v[ok]]
            pair_fam.append(c)
        pair_info = {"role": role, "sig": sig}
        role_t, sig_t, pw_t = torch.from_numpy(role).to(dev), torch.from_numpy(sig).to(dev), torch.from_numpy(pw).to(dev)
        print("pair families: " + "; ".join("%s pos %d neg %d" % (LABELS[c], int((role[:, c] == 1).sum()), int((role[:, c] == -1).sum()))
                                              for c in pair_fam), flush=True)

    ext_y = None
    if a.extent:
        if state_spec is None:
            raise SystemExit("--extent needs --state-targets")
        ext_y = torch.full((len(pool), C), float("nan"))
        for l in ("Medial OA", "Lateral OA", "PF OA"):
            c, cats_ = LABELS.index(l), state_spec["categories"][l]
            pos_i, neg_i = cats_.index("high_grade_large"), {cats_.index("normal"), cats_.index("low_grade_chondral")}
            si = st_idx[:, c].cpu()
            ext_y[si == pos_i, c] = 1.0
            for k_ in neg_i:
                ext_y[si == k_, c] = 0.0
        ext_y = ext_y.to(dev)
        print("extent supervision: positives (>= 1 cm^2 stated) %s, negatives %s" % (
            [int((ext_y[:, LABELS.index(l)] == 1).sum()) for l in ("Medial OA", "Lateral OA", "PF OA")],
            [int((ext_y[:, LABELS.index(l)] == 0).sum()) for l in ("Medial OA", "Lateral OA", "PF OA")]), flush=True)

    def extent_loss(area, sel):
        """>= 1 cm^2 stated (1) vs normal / low grade (0), weighted by the report certainty like the state loss (audit 5)."""
        y = ext_y[sel]
        known = ~torch.isnan(y)
        if not bool(known.any()):
            return area.new_zeros(())
        w_ = st_w[sel].float()[known]
        logit_ = 2.0 * (torch.log1p(area.float()) - math.log(101.0))
        l_ = F.binary_cross_entropy_with_logits(logit_[known], y[known], reduction="none")
        return (l_ * w_).sum() / w_.sum().clamp_min(1e-6)

    def pair_buckets(uids):
        """(positive dataset indices, negative dataset indices, probability) per (family class, signature) with both
        present; every family gets the same total probability, buckets within it ~ sqrt(n_pos * n_neg)."""
        idx = {u: i for i, u in enumerate(pool)}
        pi = np.array([idx[u] for u in uids])
        out = []
        for c in pair_fam:
            r, sg = pair_info["role"][pi, c], pair_info["sig"][pi, c]
            fam = []
            for v in np.unique(sg[r != 0]):
                P_ = np.flatnonzero((r == 1) & (sg == v)); N_ = np.flatnonzero((r == -1) & (sg == v))
                if len(P_) and len(N_):
                    fam.append((torch.from_numpy(P_), torch.from_numpy(N_), math.sqrt(len(P_) * len(N_))))
            tot = sum(b[2] for b in fam)
            out += [(p_, n_, w_ / tot / len(pair_fam)) for p_, n_, w_ in fam]
        return out

    def pair_loss(z, sel):
        """Matched-pair ranking: for each ordered (positive i, hard negative j) of a family class in the batch with the same
        co-finding signature, w * softplus(z_j - z_i), w = (prior_i - prior_j) * min(sw_i, sw_j); mean over active classes."""
        r, sg, pr, cw = role_t[sel], sig_t[sel], pw_t[sel], st_w[sel]
        zf = z.float()
        tot, cnt = zf.new_zeros(()), 0
        for c in pair_fam:
            P_ = torch.nonzero(r[:, c] == 1).flatten()
            N_ = torch.nonzero(r[:, c] == -1).flatten()
            if len(P_) == 0 or len(N_) == 0:
                continue
            same = (sg[P_, c][:, None] == sg[N_, c][None, :]).float()
            w = (pr[P_, c][:, None] - pr[N_, c][None, :]).clamp_min(0) * torch.minimum(cw[P_, c][:, None], cw[N_, c][None, :]) * same
            if float(w.sum()) <= 0:
                continue
            # mean over the valid pairs with w as each pair's weight (dividing by sum(w) would cancel w for a single pair)
            tot = tot + (w * F.softplus(zf[N_, c][None, :] - zf[P_, c][:, None])).sum() / same.sum()
            cnt += 1
        return tot / max(cnt, 1), cnt

    if a.check_targets:
        for item in a.arms.split(","):
            kind, t, w = arm_tensors(item.split("=")[0])
            print("check-targets arm %s (%s): %d x %d cells, finite %s, weighted cells %d, mean target per class %s" % (
                item.split("=")[0], kind, t.shape[0], t.shape[1], bool(torch.isfinite(t).all()), int((w > 0).sum()),
                [round(float(v), 3) for v in t.mean(0)]), flush=True)
        print("check-targets: bias0 finite %s; holdout fold %s has %d pool studies" % (
            bool(torch.isfinite(bias0).all()), a.holdout_fold, int((pool_fold == a.holdout_fold).sum()) if a.holdout_fold is not None else 0))
        bad = []
        for item in a.arms.split(","):
            kind, t, w = arm_tensors(item.split("=")[0])
            if not bool(torch.isfinite(t).all()):
                bad.append("non-finite targets in arm " + item)
            if kind == "lr" and bool((t <= 0).any()):
                bad.append("non-positive likelihood ratios in arm " + item)
            if kind == "bce" and bool(((t < 0) | (t > 1)).any()):
                bad.append("probability targets outside [0, 1] in arm " + item)
        if not bool(torch.isfinite(bias0).all()):
            bad.append("non-finite bias0 (a blank or non-numeric prior in the pool)")
        if state_spec is not None:
            kc = torch.tensor([len(state_spec["categories"][l]) for l in LABELS], device=st_idx.device)
            if bool((st_idx >= kc[None]).any()):
                bad.append("state index outside its class's category list")
        if bad:
            print("check-targets FAILED: " + "; ".join(bad), flush=True)
            raise SystemExit(1)
        print("check-targets OK", flush=True)
        raise SystemExit(0)

    def rare_weights(uids):
        """1 + 2[MCL state>=2 or acute] + 2[fracture state>=1] + [PF OA high] + [Baker>=moderate] + [traumatic edema/contusion], cap 4."""
        idx = {u: i for i, u in enumerate(pool)}
        L = {l: c for c, l in enumerate(LABELS)}
        bn = {(b, l): j for j, (b, l) in enumerate(bin_names)}
        w = []
        for u in uids:
            i = idx[u]
            st = st_idx[i].tolist()
            v = 1.0
            v += 2.0 * (st[L["MCL"]] >= 2 or (("acute", "MCL") in bn and float(bin_t[i, bn[("acute", "MCL")]]) == 1.0))
            v += 2.0 * (st[L["Fracture"]] >= 1)
            v += 1.0 * (st[L["PF OA"]] in (3, 4))
            v += 1.0 * (st[L["Baker's"]] in (3, 4, 5))
            v += 1.0 * (st[L["Contusion"]] in (4, 5))
            w.append(min(v, 4.0))
        return torch.tensor(w, dtype=torch.double)

    def make_loader(uids, mode, seed, shuffle):
        ds = Studies(a.px, uids, a.train_windows, mode, seed, eval_stride=a.eval_stride,
                     train_sampling=a.train_sampling, cover_stride=a.cover_stride, px_extra=a.px_extra, slices=a.slices,
                     slice_jitter=a.slice_jitter if mode == "train" else 0.0, phys_mm=a.phys_mm, px_override=a.px_override,
                     extent_meta=a.extent)
        train = mode == "train"                     # evaluation: one study per batch, few workers, no pinned copies
        sampler = None
        if train and a.rare_boost:
            if state_spec is None:
                raise SystemExit("--rare-boost needs --state-targets")
            w = rare_weights(uids)
            sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(uids), replacement=True,
                                                             generator=torch.Generator().manual_seed(seed + 7))
            shuffle = False
            print("rare-boost: mean weight %.2f, boosted studies %d of %d" % (float(w.mean()), int((w > 1).sum()), len(w)), flush=True)
        if train and a.pair_rank > 0:               # matched pairs on top of the same base distribution
            bk = pair_buckets(uids)
            bs = PairBatchSampler(len(uids), a.batch, None if sampler is None else sampler.weights, bk, a.pair_frac, seed + 7)
            print("pair-rank: %d buckets over %d families, %.0f%% of batches drawn as a matched pair (others: base draws, weighted with replacement)" % (
                len(bk), len(pair_fam), 100 * a.pair_frac), flush=True)
            return ds, torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=a.workers, collate_fn=collate,
                                                   persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                                   generator=torch.Generator().manual_seed(seed))
        return ds, torch.utils.data.DataLoader(ds, batch_size=a.batch if train else 1, shuffle=shuffle, sampler=sampler,
                                               num_workers=a.workers if train else min(a.workers, a.eval_workers),
                                               collate_fn=collate, drop_last=train, persistent_workers=False,
                                               pin_memory=train, prefetch_factor=2 if train else 1,
                                               generator=torch.Generator().manual_seed(seed))

    def augment(img, gen):
        B, W, _, S, _ = img.shape
        oy = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
        ox = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
        img = img[..., oy:oy + a.crop, ox:ox + a.crop]
        if a.aug != "v2":
            return img
        # per-study rotation (+-10 deg) and scale (0.9-1.1) through one affine grid; gamma 0.8-1.25; contrast 0.9-1.1.
        # No flips: a horizontal flip swaps medial/lateral, a vertical one anterior/posterior.
        Cc = img.shape[2]
        x = img.float().reshape(B * W, Cc, a.crop, a.crop)
        ang = (torch.rand(B, generator=gen) * 20 - 10) * math.pi / 180
        sc = torch.rand(B, generator=gen) * 0.2 + 0.9
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros(B)], 1), torch.stack([sin, cos, torch.zeros(B)], 1)], 1)
        theta = theta.repeat_interleave(W, 0).to(x.device)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if a.aug_res > 0:                       # per-study in-plane resolution loss (coarse scanners: 0.33-0.5 mm pixels)
            hit = torch.rand(B, generator=gen) < a.aug_res
            fac = torch.rand(B, generator=gen) * 0.75 + 1.25
            if bool(hit.any()):
                xs = x.view(B, W, Cc, a.crop, a.crop)
                for b in torch.nonzero(hit).flatten().tolist():
                    s_ = max(8, int(round(a.crop / float(fac[b]))))
                    lo = F.interpolate(xs[b], size=(s_, s_), mode="area")
                    xs[b] = F.interpolate(lo, size=(a.crop, a.crop), mode="bilinear", align_corners=False)
                x = xs.view(B * W, Cc, a.crop, a.crop)
        gamma = (torch.rand(B, generator=gen) * 0.45 + 0.8).repeat_interleave(W, 0).to(x.device).view(-1, 1, 1, 1)
        contrast = (torch.rand(B, generator=gen) * 0.2 + 0.9).repeat_interleave(W, 0).to(x.device).view(-1, 1, 1, 1)
        x = (x / 255.).clamp_(0, 1).pow(gamma)
        x = ((x - 0.5) * contrast + 0.5).clamp_(0, 1) * 255.
        return x.round_().to(torch.uint8).reshape(B, W, Cc, a.crop, a.crop)

    def center(img):
        S = img.shape[-1]
        o = (S - a.crop) // 2
        return img[..., o:o + a.crop, o:o + a.crop]

    @torch.no_grad()                                  # evaluation only: no autograd graph (same outputs, less time and memory)
    def predict(model, uids, seed):
        model.eval()
        ds, dl = make_loader(uids, "eval", seed, False)
        out = torch.zeros((len(uids), C), device=dev)
        for img, meta, mask, idx in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(center(img.to(dev, non_blocking=True)), meta.to(dev), mask.to(dev), chunk=PREDICT_CHUNK)
            out[idx.to(dev)] = z.float()
        return out

    @torch.no_grad()
    def bn_recalibrate(model, uids, seed, n_batches):
        """--ema copies only the trainable parameters into the shipped model, so BatchNorm running statistics still describe
        the live weights (CoAtNet screen 10-08: holdout 0.61, predictions nearly constant).  Recompute them for the shipped
        weights: cumulative averages over n_batches training batches, centre crop, no augmentation.  The whole model is in
        train mode so that no zero padding enters the statistics (Net.forward pads only in eval mode or without BatchNorm);
        the head's dropout comes after the encoder and does not reach them.  Returns the number of windows seen."""
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not bns or n_batches <= 0:
            return 0
        mom = [m.momentum for m in bns]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        model.train()
        ds_r, dl_r = make_loader(uids, "train", seed, True)
        ds_r.slice_jitter = 0.0                      # audit 10-08 defect 1: no slice-jitter augmentation in the statistics
        seen = used = 0
        stance = torch.compiler.set_stance("force_eager") if hasattr(torch.compiler, "set_stance") else contextlib.nullcontext()
        with stance:                                 # audit defect 2: never through a compiled encoder (guards, recompiles)
            for i, (img, meta, mask, idx) in enumerate(dl_r):
                if i >= n_batches:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(center(img.to(dev, non_blocking=True)), meta.to(dev), mask.to(dev))
                seen += int(mask.sum())
                used += 1
        for m, v in zip(bns, mom):
            m.momentum = v
        model.eval()
        return used, seen

    def reeval(arm, seed):
        only = None if not a.only_folds else {int(x) for x in a.only_folds.split(",")}
        pre = "partial_" if only else ""
        path = os.path.join(a.out, pre + "preds_%s_s%d.json" % (arm, seed))
        if os.path.exists(path):
            return
        src = json.load(open(os.path.join(a.reeval_run, "pool_uids.json"), encoding="utf-8"))
        if src != pool:
            raise SystemExit("pool of --reeval-run differs from the current pool")
        z_gold = torch.zeros((len(gold), C), device=dev)
        z_pool = torch.zeros((len(pool), C), device=dev)
        t_arm = time.time()
        H = a.holdout_fold
        for k in range(K):
            if only is not None and k not in only:
                continue
            if H is not None and k == H:                # no model is trained for the holdout fold
                continue
            ho = np.flatnonzero(pool_fold == k)
            model = Net(a.backbone, a.init, freeze_blocks=a.freeze_blocks, crop=a.crop, state_spec=state_spec, in_chans=a.slices, spatial=a.spatial_attn, extent=a.extent).to(dev).to(memory_format=torch.channels_last)
            sd = torch.load(os.path.join(a.reeval_run, "models", "%s_s%d_f%d.pt" % (arm, seed, k)), map_location="cpu")
            if a.backbone == OFL_BACKBONE:
                model.load_deploy_state_dict(sd)
            else:
                model.load_state_dict(sd)
            if model.has_bn and a.bn_recal_batches > 0:
                tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
                b_, w_ = bn_recalibrate(model, [pool[i] for i in tr], 9100 + 17 * seed + k, a.bn_recal_batches)
                print("  reeval fold %d: BatchNorm statistics recalibrated over %d batches (%d windows)" % (k, b_, w_), flush=True)
            z_gold += predict(model, gold, seed) / (K if H is None else K - 1)
            z_pool[torch.from_numpy(ho).to(dev)] = predict(model, [pool[i] for i in ho], seed)
            if H is not None:                       # the clean holdout fold, as in training (mean of the fold models)
                hh = np.flatnonzero(pool_fold == H)
                z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            print("  reeval %s seed %d fold %d done (eval stride %d) %.0fs" % (arm, seed, k, a.eval_stride, time.time() - t_arm), flush=True)
            del model
            torch.cuda.empty_cache()
        np.save(os.path.join(a.out, pre + "oof_%s_s%d.npy" % (arm, seed)), z_pool.cpu().numpy().astype(np.float32))
        out = {"arm": arm, "seed": seed, "reeval_of": a.reeval_run, "eval_stride": a.eval_stride, "gold_uids": gold, "labels": LABELS,
               "holdout_fold": H,
               "only_folds": sorted(only) if only else None,
               "gold_fold": [int(by_uid[u]["fold"]) for u in gold],
               "gold_logits": [[round(float(x), 5) for x in r] for r in z_gold.cpu().numpy()],
               "receipt": [{"arm": arm, "kind": "reeval", "pool_studies": len(pool), "gold_in_training": 0, "trained": False}]}
        json.dump(out, open(path, "w"))

    def run(arm, seed):
        if a.reeval_run:
            return reeval(arm, seed)
        only = None if not a.only_folds else {int(x) for x in a.only_folds.split(",")}
        path = os.path.join(a.out, ("partial_" if only else "") + "preds_%s_s%d.json" % (arm, seed))
        if os.path.exists(path):
            return
        kind, tgt, wgt = arm_tensors(arm)
        z_gold = torch.zeros((len(gold), C), device=dev)
        z_pool = torch.zeros((len(pool), C), device=dev)
        logs, steps_total = [], 0
        t_arm = time.time()
        H = a.holdout_fold
        for k in range(K):
            if only is not None and k not in only:
                print("  fold %d skipped (--only-folds)" % k, flush=True)
                continue
            if H is not None and k == H:
                print("  fold %d is the clean holdout: no model trained for it" % k, flush=True)
                continue
            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
            ho = np.flatnonzero(pool_fold == k)
            torch.manual_seed(1000 + 17 * seed)          # same init for every fold: heads start identical, so fold models can be souped
            gen = torch.Generator().manual_seed(5000 + 17 * seed + k)
            model = Net(a.backbone, a.init, freeze_blocks=a.freeze_blocks, crop=a.crop, state_spec=state_spec, in_chans=a.slices, spatial=a.spatial_attn, extent=a.extent).to(dev).to(memory_format=torch.channels_last)
            torch.manual_seed(1000 + 17 * seed + k)      # fold-specific stream for dropout / drop path / augmentation
            if a.grad_ckpt:                              # 32 GB cards: recompute encoder activations in backward (same maths, ~1.3x compute)
                model.enc.set_grad_checkpointing(True)
            if a.compile:                                # fused kernels; Module.compile keeps state_dict keys (checkpoints, deploy unchanged)
                import torch._dynamo as _dynamo          # aliased: a bare 'import torch...' here would make torch a local name
                _dynamo.config.cache_size_limit = max(32, _dynamo.config.cache_size_limit)
                model.enc.compile(dynamic=True)
            with torch.no_grad():
                model.bias.copy_(bias0)
            enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
            head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
            groups = [{"params": enc, "lr": a.lr}]
            vit_mod = model.enc.vit if a.backbone == OFL_BACKBONE else (model.enc if hasattr(model.enc, "blocks") else None)
            if vit_mod is not None and a.llrd != 1.0:
                nb = len(vit_mod.blocks)
                groups, seen = [], set()
                for i, blk in enumerate(vit_mod.blocks):
                    ps = [p for p in blk.parameters() if p.requires_grad]
                    if ps:
                        groups.append({"params": ps, "lr": a.lr * (a.llrd ** (nb - 1 - i))}); seen.update(id(p) for p in ps)
                rest = [p for p in enc if id(p) not in seen]          # final norm (top) and, when unfrozen, embeddings (bottom)
                top = [p for n, p in vit_mod.named_parameters() if n.startswith("norm.") and p.requires_grad]
                bottom = [p for p in rest if id(p) not in {id(q) for q in top}]
                if top:
                    groups.append({"params": top, "lr": a.lr})
                if bottom:
                    groups.append({"params": bottom, "lr": a.lr * (a.llrd ** nb)})
            if k == (sorted(only)[0] if only else 0) and seed == 0:
                print("trainable encoder params %.1fM, head params %.2fM, frozen encoder params %.1fM" % (
                    sum(p.numel() for p in enc) / 1e6, sum(p.numel() for p in head) / 1e6,
                    sum(p.numel() for n, p in model.named_parameters() if n.startswith("enc.") and not p.requires_grad) / 1e6), flush=True)
            opt = torch.optim.AdamW(groups + [{"params": head, "lr": a.head_lr}], weight_decay=a.wd)
            ema = None
            if a.ema > 0:                               # EMA over everything that trains (frozen parts are shared anyway)
                ema = {n_: p_.detach().clone() for n_, p_ in model.named_parameters() if p_.requires_grad}
            ds, dl = make_loader([pool[i] for i in tr], "train", 9000 + 17 * seed + k, True)
            total = len(dl) * a.epochs
            warm = max(1, int(.05 * total))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                      .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
            tr_t = torch.from_numpy(tr).to(dev)
            log = []
            # resumable state, written atomically at the end of every --ckpt-every epochs: weights, optimizer, LR schedule, EMA,
            # and every random stream that decides the next epoch (augmentation, sampler, loader, CPU and GPU global streams)
            ckpt_path = os.path.join(a.out, "ckpt", "%s_s%d_f%d.pt" % (arm, seed, k))
            start_ep = 0
            if a.resume and os.path.exists(ckpt_path):
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ck.get("epochs") != a.epochs or ck.get("train_studies") != len(tr):
                    raise SystemExit("checkpoint %s does not match this run (epochs / training set)" % ckpt_path)
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                if ema is not None:
                    ema = {n_: v.to(dev) for n_, v in ck["ema"].items()}
                gen.set_state(ck["gen"])
                if ck.get("sampler_gen") is not None:
                    (dl.batch_sampler.generator if isinstance(dl.batch_sampler, PairBatchSampler) else dl.sampler.generator).set_state(ck["sampler_gen"])
                if ck.get("loader_gen") is not None:
                    dl.generator.set_state(ck["loader_gen"])
                torch.set_rng_state(ck["cpu_rng"])
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
                log = list(ck["log"])
                start_ep = int(ck["epoch"]) + 1
                print("  resumed %s seed %d fold %d after epoch %d (%s)" % (arm, seed, k, ck["epoch"], ckpt_path), flush=True)
            for ep in range(start_ep, a.epochs):
                ds.epoch = ep
                model.train()
                tot = n = 0
                pr_batches = pr_terms = 0
                ext_acc = torch.zeros(6, device=dev)
                for img, meta, mask, idx in dl:
                    sel = tr_t[idx.to(dev)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z, sl, bl = model(augment(img.to(dev, non_blocking=True), gen), meta.to(dev), mask.to(dev), return_aux=True)
                    loss = bce_soft(z, tgt[sel], wgt[sel]) if kind == "bce" else lr_loss(z, tgt[sel])
                    if state_spec is not None:
                        loss = loss + a.state_weight * state_loss(sl, bl, st_idx[sel], st_w[sel], bin_t[sel])
                    if a.extent:
                        ext_l = extent_loss(model._ext_area, sel)
                        if a.extent_weight > 0:
                            loss = loss + a.extent_weight * ext_l
                        with torch.no_grad():                     # epoch log of the extent path (audit 6), no sync per step
                            la_, y_ = torch.log1p(model._ext_area.float() / 100.0), ext_y[sel]
                            ext_acc += torch.stack([(la_ * (y_ == 1)).sum(), (y_ == 1).sum().float(), (la_ * (y_ == 0)).sum(),
                                                    (y_ == 0).sum().float(), ext_l.detach().float(), torch.ones((), device=dev)])
                    if a.pair_rank > 0:
                        pl, pc = pair_loss(z, sel)
                        loss = loss + a.pair_rank * pl
                        pr_batches += int(pc > 0); pr_terms += pc
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if a.forward_only:
                        g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
                        print("forward-only check: loss %.4f, gradient mass %.3e, windows %d, optimizer steps 0" % (
                            float(loss), g, int(mask.sum())), flush=True)
                        return
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    opt.step()
                    sched.step()
                    if ema is not None:
                        with torch.no_grad():
                            d_ = min(a.ema, (1 + sched.last_epoch) / (10 + sched.last_epoch))   # warm EMA early on
                            for n_, p_ in model.named_parameters():
                                if n_ in ema:
                                    ema[n_].mul_(d_).add_(p_.detach(), alpha=1 - d_)
                    tot += float(loss.detach()) * len(idx)
                    n += len(idx)
                log.append(round(tot / max(1, n), 5))
                print("  %s seed %d fold %d epoch %d loss %.4f  %.0fs" % (arm, seed, k, ep, log[-1], time.time() - t_arm), flush=True)
                if a.extent:
                    ea = ext_acc.tolist()
                    print("  extent epoch %d: auxiliary loss %.4f (weight %.2f) | gamma MOA/LOA/PFOA %s | mean log1p(A/100) on >= 1 cm^2 stated %.3f vs normal/low grade %.3f" % (
                        ep, ea[4] / max(1.0, ea[5]), a.extent_weight,
                        [round(float(model.ext_gamma[LABELS.index(l)]), 3) for l in ("Medial OA", "Lateral OA", "PF OA")],
                        ea[0] / max(1.0, ea[1]), ea[2] / max(1.0, ea[3])), flush=True)
                if a.pair_rank > 0:
                    print("  pair-rank epoch %d: %d of %d batches had at least one matched-pair term (%d class terms; includes base batches that happened to pair)" % (ep, pr_batches, len(dl), pr_terms), flush=True)
                if a.ckpt_every > 0 and (ep + 1) % a.ckpt_every == 0 and ep + 1 < a.epochs:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    sgen = getattr(dl.sampler, "generator", None) if isinstance(dl.sampler, torch.utils.data.WeightedRandomSampler) else None
                    if isinstance(dl.batch_sampler, PairBatchSampler):
                        sgen = dl.batch_sampler.generator
                    torch.save({"epoch": ep, "epochs": a.epochs, "train_studies": len(tr), "log": log,
                                "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "ema": None if ema is None else {n_: v.detach().cpu() for n_, v in ema.items()},
                                "gen": gen.get_state(), "sampler_gen": None if sgen is None else sgen.get_state(),
                                "loader_gen": None if dl.generator is None else dl.generator.get_state(),
                                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, ckpt_path + ".tmp")
                    os.replace(ckpt_path + ".tmp", ckpt_path)
                if a.stop_after_epoch >= 0 and ep == a.stop_after_epoch:
                    print("  stop-after-epoch %d: exiting (test)" % ep, flush=True)
                    sys.exit(3)
            steps_total += total
            if ema is not None:                         # evaluate and ship the EMA weights
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if n_ in ema:
                            p_.copy_(ema[n_])
                if model.has_bn and a.bn_recal_batches > 0:
                    b_, w_ = bn_recalibrate(model, [pool[i] for i in tr], 9100 + 17 * seed + k, a.bn_recal_batches)
                    print("  BatchNorm statistics recalibrated for the EMA weights: %d batches, %d windows" % (b_, w_), flush=True)
                elif model.has_bn:
                    print("  WARNING: BatchNorm backbone with --ema and no --bn-recal-batches: the saved model pairs EMA weights with "
                          "the live weights' BatchNorm statistics (see the 10-08 CoAtNet screen)", flush=True)
            os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
            torch.save(model.deploy_state_dict(),
                       os.path.join(a.out, "models", "%s_s%d_f%d.pt" % (arm, seed, k)))
            if os.path.exists(ckpt_path):                 # the fold is complete: its resumable state is no longer needed
                os.remove(ckpt_path)
            z_gold += predict(model, gold, seed) / (K if H is None else K - 1)
            z_pool[torch.from_numpy(ho).to(dev)] = predict(model, [pool[i] for i in ho], seed)
            if H is not None:
                hh = np.flatnonzero(pool_fold == H)
                z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            logs.append([log[0], log[-1]])
            del model, opt
            torch.cuda.empty_cache()
        np.save(os.path.join(a.out, ("partial_" if only else "") + "oof_%s_s%d.npy" % (arm, seed)), z_pool.cpu().numpy().astype(np.float32))
        out = {"arm": arm, "seed": seed, "epochs": a.epochs, "nested": False, "gold_uids": gold, "only_folds": sorted(only) if only else None,
               "holdout_fold": H,
               "gold_fold": [int(by_uid[u]["fold"]) for u in gold], "labels": LABELS,
               "gold_logits": [[round(float(x), 5) for x in r] for r in z_gold.cpu().numpy()],
               "loss_first_last": logs, "optimizer_steps": steps_total, "train_seconds": round(time.time() - t_arm, 1),
               "receipt": [{"arm": arm, "kind": kind, "pool_studies": len(pool), "gold_in_training": 0,
                            "init_seed": 1000 + 17 * seed, "note": "fold models share one initialisation (soup-compatible)"}]}
        json.dump(out, open(path, "w"))

    cfg = {k: getattr(a, k) for k in vars(a)}
    cfg.update({"pool_studies": len(pool), "gold_studies": len(gold), "feature_studies": len(done), "labels": LABELS,
                "scaffold": "compact fine-tuned image model on the pixel cache"})
    json.dump(cfg, open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
    json.dump(pool, open(os.path.join(a.out, "pool_uids.json"), "w"))
    for seed in range(a.seed_base, a.seed_base + a.seeds):
        for arm in [x.split("=", 1)[0] for x in a.arms.split(",") if x]:
            run(arm, seed)
            if a.forward_only:
                return
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
