"""Shared model architecture for training and offline inference."""
import numpy as np
import torch
import torch.nn as nn
import timm

def make_neighbours(x):
    # (B,S,T,H,W) -> (B,S,T,3,H,W) from slices t-1,t,t+1 (edges clamped)
    prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
    nxt  = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
    return torch.stack([prev, x, nxt], dim=3)


class AttnPool(nn.Module):
    # Attention over slices within one plane.
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None):               # x (B,T,D)
        a = self.score(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, torch.finfo(a.dtype).min)
        a = a.softmax(dim=-1)
        return (x * a.unsqueeze(-1)).sum(1), a


class LabelAttnPool(nn.Module):
    # Per-label attention over planes: each finding picks the view it needs.
    def __init__(self, dim, n_labels):
        super().__init__()
        self.score = nn.Linear(dim, n_labels)

    def forward(self, x, mask):                    # x (B,S,D), mask (B,S)
        a = self.score(x)                          # (B,S,L)
        a = a.masked_fill(~mask.unsqueeze(-1), torch.finfo(a.dtype).min)
        a = a.softmax(dim=1)                       # over planes, per label
        z = torch.einsum("bsd,bsl->bld", x, a)     # (B,L,D)
        return z, a


class KneeNet(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b0.ns_jft_in1k", n_out=12, n_planes=3,
                 pretrained=True, drop=0.3, img=224, normalization="legacy",
                 encoder_chunk=64):
        super().__init__()
        kwargs = {"img_size": img} if backbone.startswith("vit_") else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, in_chans=3, **kwargs)
        self.encoder_chunk = encoder_chunk
        config = timm.data.resolve_model_data_config(self.enc)
        mean = config['mean'] if normalization == 'pretrained' else (0.449,)*3
        std = config['std'] if normalization == 'pretrained' else (0.226,)*3
        self.register_buffer('input_mean', torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer('input_std', torch.tensor(std).view(1,3,1,1), persistent=False)
        d = self.enc.num_features
        self.slice_pool = AttnPool(d)
        self.plane_emb = nn.Parameter(torch.zeros(n_planes, d))
        self.plane_pool = LabelAttnPool(d, n_out)
        self.drop = nn.Dropout(drop)
        self.head_w = nn.Parameter(torch.zeros(n_out, d))
        self.head_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.head_w, std=0.02)

    def forward(self, x, present):
        # autocast entered HERE, not in the training loop: it is thread-local, so a
        # loop-level autocast never reaches a threaded forward. Verified reaching the
        # encoder as float16.
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            B, S, T, H, W = x.shape
            x = x.float().div(255.)   # ImageNet grey stats
            x3 = make_neighbours(x).reshape(B * S * T, 3, H, W)

            x3 = (x3 - self.input_mean) / self.input_std
            # Chunk evaluation to bound memory; training keeps BatchNorm batch statistics intact.
            if not self.training and self.encoder_chunk and len(x3) > self.encoder_chunk:
                features = torch.cat([self.enc(part) for part in x3.split(self.encoder_chunk)])
            else:
                features = self.enc(x3)
            f = features.reshape(B * S, T, -1)
            v, _ = self.slice_pool(f)
            v = v.reshape(B, S, -1) + self.plane_emb.unsqueeze(0)

            m = present.clone()
            m[~m.any(dim=1)] = True                # guard: never all-masked
            z, plane_att = self.plane_pool(v, m)
            z = self.drop(z)
            logits = (z * self.head_w.unsqueeze(0)).sum(-1) + self.head_b
        return logits, plane_att



def slice_thirds(n):
    # Lower, middle and upper thirds of the cached slice band.
    parts = np.array_split(np.arange(n), 3)
    if min(len(p) for p in parts) == 0:
        return [np.array([0]), np.array([(n - 1) // 2]), np.array([n - 1])]
    return parts


def spread_groups(n):
    # 3-slice inputs that each span the band: one slice from every third. For six cached
    # slices that is [0, 2, 4] and [1, 3, 5]; a 3-slice input is its own single group.
    parts = slice_thirds(n)
    return [[int(p[g]) for p in parts] for g in range(min(len(p) for p in parts))]


class SlotQueryHead(nn.Module):
    # Per-finding query attention over series slots (public DINOv2 member head).
    def __init__(self, dim, n_slots, n_out, hidden=256, drop=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):                    # x (B,S,D), mask (B,S)
        h = self.proj(x) + self.slot_emb
        a = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        a = a.masked_fill(~mask.unsqueeze(1), torch.finfo(a.dtype).min).softmax(-1)
        z = self.drop(torch.einsum("bos,bsh->boh", a, h))
        return (z * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias, a


class SlotQueryNet(nn.Module):
    """One 3-slice image per series slot, encoder partly fine-tuned, query head on top.

    Training feeds one spread triplet per slot (T=3). Evaluation feeds every cached slice
    and averages logits over the spread groups, so no slice position is left unseen.
    """
    def __init__(self, backbone="vit_small_patch14_dinov2.lvd142m", n_out=12, n_slots=6,
                 pretrained=True, img=336, drop=0.2, hidden=256, unfreeze_last=6,
                 normalization="pretrained", encoder_chunk=48):
        super().__init__()
        kwargs = {"img_size": img} if backbone.startswith("vit_") else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, in_chans=3, **kwargs)
        self.img = img
        self.encoder_chunk = encoder_chunk
        self.tokens = hasattr(self.enc, "blocks") and hasattr(self.enc, "num_prefix_tokens")
        config = timm.data.resolve_model_data_config(self.enc)
        mean = config['mean'] if normalization == 'pretrained' else (0.449,)*3
        std = config['std'] if normalization == 'pretrained' else (0.226,)*3
        self.register_buffer('input_mean', torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer('input_std', torch.tensor(std).view(1,3,1,1), persistent=False)
        if unfreeze_last is not None and hasattr(self.enc, "blocks"):
            # Early blocks stay at their self-supervised weights; only late blocks adapt.
            for p in self.enc.parameters(): p.requires_grad = False
            blocks = list(self.enc.blocks)
            for block in blocks[max(0, len(blocks) - unfreeze_last):]:
                for p in block.parameters(): p.requires_grad = True
            for name in ("norm", "fc_norm"):
                module = getattr(self.enc, name, None)
                if isinstance(module, nn.Module):
                    for p in module.parameters(): p.requires_grad = True
        dim = self.enc.num_features * 2
        self.head = SlotQueryHead(dim, n_slots, n_out, hidden=hidden, drop=drop)

    def encode(self, x):                           # (N,3,H,W) normalised -> (N,2D)
        f = self.enc.forward_features(x)
        if self.tokens:
            # CLS token plus mean of patch tokens; register tokens are excluded.
            return torch.cat([f[:, 0], f[:, self.enc.num_prefix_tokens:].mean(1)], dim=1)
        if f.ndim == 4:
            return torch.cat([f.mean((-2, -1)), f.amax((-2, -1))], dim=1)
        return torch.cat([f, f], dim=1)

    def forward(self, x, present):
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            B, S, T, H, W = x.shape
            m = present.clone()
            m[~m.any(dim=1)] = True                # guard: never all-masked
            keep = m.reshape(-1)
            groups = spread_groups(T)
            logits, attention = 0., 0.
            for group in groups:
                images = x[:, :, group].reshape(B * S, 3, H, W)[keep].float().div(255.)
                if H != self.img:
                    images = nn.functional.interpolate(images, size=(self.img, self.img),
                                                       mode="bilinear", align_corners=False)
                images = (images - self.input_mean) / self.input_std
                # Absent slots are never encoded; their features stay zero and masked.
                if not self.training and self.encoder_chunk and len(images) > self.encoder_chunk:
                    f = torch.cat([self.encode(part) for part in images.split(self.encoder_chunk)])
                else:
                    f = self.encode(images)
                features = f.new_zeros(B * S, f.shape[-1])
                features[keep] = f
                z, a = self.head(features.reshape(B, S, -1), m)
                logits, attention = logits + z, attention + a
        return logits / len(groups), attention / len(groups)


class Reader25D(nn.Module):
    """2.5D study reader (public ConvNeXt reader layout).

    Every cached window of three adjacent slices is one CNN image. Its feature becomes a
    token tagged with its series slot and its position along the stack; a small
    transformer mixes all tokens of the study and each finding pools them with its own
    attention. Training augments each series with one shared rigid/intensity jitter (no
    flips: laterality is already normalised) and encodes a random subset of windows.
    """
    def __init__(self, backbone="convnext_tiny.in12k_ft_in1k", n_out=12, n_slots=6,
                 pretrained=True, img=240, dim=384, depth=2, drop=0.2, drop_path=0.1,
                 normalization="pretrained", band=(0.06, 0.94), train_windows=6,
                 encoder_chunk=96, slot_dropout=0.0, aux_dim=0):
        super().__init__()
        self.slot_dropout = slot_dropout     # training only: hide whole series (protocols vary by site)
        # Runtime switches for XLA/TPU training; neither changes the parameters or the checkpoint.
        # static_shapes: encode a fixed number of windows for every slot (absent ones masked), so
        # tensor shapes never depend on the data. augment_in_model=False: the data loader already
        # applied the per-series jitter (augment_series_cpu), so the model only normalises.
        self.static_shapes = False
        self.augment_in_model = True
        # Test-time window subsets (eval only; inference WINDOW_TTA): encode a random subset of this many
        # windows per series, as in training, instead of all of them. None: every window.
        self.subset_windows = None
        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        if backbone.startswith("vit_"):
            kwargs["img_size"] = img
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kwargs)
        self.img, self.band, self.train_windows = img, tuple(band), train_windows
        self.encoder_chunk = encoder_chunk
        config = timm.data.resolve_model_data_config(self.enc)
        mean = config['mean'] if normalization == 'pretrained' else (0.449,)*3
        std = config['std'] if normalization == 'pretrained' else (0.226,)*3
        self.register_buffer('input_mean', torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer('input_std', torch.tensor(std).view(1,3,1,1), persistent=False)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim),
                                  nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(n_slots, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        # Report-grounded auxiliary head (training only): regresses the study's report embedding
        # from the finding-pooled and the mean study tokens. Its output is kept in last_aux; the
        # logits and the inference path do not depend on it.
        self.aux = (nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, aux_dim))
                    if aux_dim else None)
        self.last_aux = None

    def _aux(self, h, pad, pooled):
        if self.aux is None:
            return
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        mean = (h * keep).sum(1) / keep.sum(1).clamp_min(1)
        self.last_aux = self.aux(torch.cat([pooled.mean(1), mean], -1))

    def views(self, x, series, n_series):
        # x float [N,3,H,W] in 0..1; one random transform per series in training.
        if x.shape[-1] != self.img:
            x = nn.functional.interpolate(x, size=(self.img, self.img), mode="bilinear",
                                          align_corners=False, antialias=x.shape[-1] > self.img)
        if self.training and self.augment_in_model:
            dev = x.device
            ang = (torch.rand(n_series, device=dev) - .5) * 2 * np.radians(12)
            sc = 1 + (torch.rand(n_series, device=dev) - .5) * .24
            tx, ty = ((torch.rand(2, n_series, device=dev) - .5) * .12)
            theta = torch.stack([torch.stack([torch.cos(ang) * sc, -torch.sin(ang) * sc, tx], 1),
                                 torch.stack([torch.sin(ang) * sc, torch.cos(ang) * sc, ty], 1)], 1)
            grid = nn.functional.affine_grid(theta[series], x.shape, align_corners=False)
            x = nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                                          align_corners=False)
            gamma = torch.exp((torch.rand(n_series, device=dev) - .5) * .5)[series]
            gain = (1 + (torch.rand(n_series, device=dev) - .5) * .2)[series]
            x = x.clamp_min(1e-4).pow(gamma.view(-1, 1, 1, 1)) * gain.view(-1, 1, 1, 1)
        return (x - self.input_mean) / self.input_std

    def forward(self, x, present):
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            B, S, T, H, W = x.shape
            K = T // 3
            m = present.clone()
            if self.training and self.slot_dropout > 0:
                kept = m & (torch.rand(m.shape, device=m.device) >= self.slot_dropout)
                # A study that lost every series keeps one of its present series, chosen at random.
                pick = torch.rand(m.shape, device=m.device).masked_fill(~m, -1).argmax(dim=1)
                lost = ~kept.any(dim=1) & m.any(dim=1)
                one = nn.functional.one_hot(pick, m.shape[1]).bool()
                m = kept | (lost.unsqueeze(1) & one)   # fixed-shape boolean algebra (XLA-safe)
            m = m | ~m.any(dim=1, keepdim=True)    # guard: never all-masked
            if self.static_shapes:
                return self._forward_static(x, m, B, S, K, H, W)
            tokens = m.unsqueeze(-1).expand(B, S, K).clone()
            n_win = self.train_windows if self.training else getattr(self, 'subset_windows', None)
            if n_win and n_win < K:
                # Encode a random subset of windows per series: stack augmentation that
                # also bounds activation memory. Positions stay exact.
                rank = torch.rand(B, S, K, device=x.device).argsort(-1).argsort(-1)
                tokens &= rank < n_win
            keep = tokens.reshape(-1)
            series = torch.arange(B * S, device=x.device).repeat_interleave(K)[keep]
            images = x[:, :, :3 * K].reshape(B * S * K, 3, H, W)[keep].float().div(255.)
            images = self.views(images, series, B * S).contiguous(memory_format=torch.channels_last)
            if not self.training and self.encoder_chunk and len(images) > self.encoder_chunk:
                f = torch.cat([self.enc(part) for part in images.split(self.encoder_chunk)])
            else:
                f = self.enc(images)
            feat = f.new_zeros(B * S * K, f.shape[-1])
            feat[keep] = f
            slot = torch.arange(S, device=x.device).view(1, S, 1).expand(B, S, K)
            pos = torch.linspace(self.band[0], self.band[1], K, device=x.device)
            h = (self.proj(feat).view(B, S, K, -1) + self.slot_emb(slot)
                 + self.pos_emb(pos.view(1, 1, K, 1).to(feat.dtype)))
            h = h.reshape(B, S * K, -1)
            pad = ~tokens.reshape(B, S * K)
            if self.tf is not None:
                h = self.tf(h, src_key_padding_mask=pad)
            a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
            pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
            logits = (pooled * self.cls_w).sum(-1) + self.cls_b
            self._aux(h, pad, pooled)
        return logits, a


def _reader_forward_static(self, x, m, B, S, K, H, W):
    """Reader25D forward with data-independent shapes (XLA/TPU). Every slot encodes the same
    number of windows; absent or dropped slots are encoded too but masked, exactly as the
    dynamic path masks them, so outputs agree up to floating-point error."""
    n_win = self.train_windows if self.training else getattr(self, 'subset_windows', None)
    n_win = n_win if n_win and n_win < K else K
    if n_win < K:
        idx = torch.rand(B, S, K, device=x.device).argsort(-1)[..., :n_win].sort(-1).values
    else:
        idx = torch.arange(K, device=x.device).view(1, 1, K).expand(B, S, K)
    windows = x[:, :, :3 * K].reshape(B, S, K, 3, H, W)
    windows = torch.gather(windows, 2, idx.reshape(B, S, n_win, 1, 1, 1).expand(B, S, n_win, 3, H, W))
    images = windows.reshape(B * S * n_win, 3, H, W).float().div(255.)
    series = torch.arange(B * S, device=x.device).repeat_interleave(n_win)
    images = self.views(images, series, B * S)
    if not self.training and self.encoder_chunk and len(images) > self.encoder_chunk:
        f = torch.cat([self.enc(part) for part in images.split(self.encoder_chunk)])
    else:
        f = self.enc(images)
    tokens = m.unsqueeze(-1).expand(B, S, n_win)
    slot = torch.arange(S, device=x.device).view(1, S, 1).expand(B, S, n_win)
    pos = torch.linspace(self.band[0], self.band[1], K, device=x.device)[idx]
    h = (self.proj(f).view(B, S, n_win, -1) + self.slot_emb(slot)
         + self.pos_emb(pos.unsqueeze(-1).to(f.dtype)))
    h = h.reshape(B, S * n_win, -1)
    pad = ~tokens.reshape(B, S * n_win)
    if self.tf is not None:
        h = self.tf(h, src_key_padding_mask=pad)
    a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
    pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
    logits = (pooled * self.cls_w).sum(-1) + self.cls_b
    self._aux(h, pad, pooled)
    return logits, a


Reader25D._forward_static = _reader_forward_static


def augment_series_cpu(x, generator=None):
    """Reader25D's training jitter on the CPU, for loaders that feed XLA devices: one rotation
    (+-12 deg), scale (+-12%), shift (+-6%), gamma and gain per series, shared by all its slices.
    x: uint8 [S, T, H, W] of one study; returns uint8 of the same shape (values clipped to 0..255)."""
    S, T, H, W = x.shape
    rnd = lambda *shape: torch.rand(*shape, generator=generator)
    ang = (rnd(S) - .5) * 2 * np.radians(12)
    sc = 1 + (rnd(S) - .5) * .24
    tx, ty = (rnd(2, S) - .5) * .12
    theta = torch.stack([torch.stack([torch.cos(ang) * sc, -torch.sin(ang) * sc, tx], 1),
                         torch.stack([torch.sin(ang) * sc, torch.cos(ang) * sc, ty], 1)], 1)
    xf = x.float().div(255.)
    grid = nn.functional.affine_grid(theta, (S, T, H, W), align_corners=False)
    xf = nn.functional.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp((rnd(S) - .5) * .5).view(S, 1, 1, 1)
    gain = (1 + (rnd(S) - .5) * .2).view(S, 1, 1, 1)
    xf = xf.clamp_min(1e-4).pow(gamma) * gain
    return (xf.clamp(0, 1) * 255).round().to(torch.uint8)


def window_centres(n_slices):
    # Centre slice of each cached 3-slice window (v3 layout); plain slices otherwise.
    return list(range(1, n_slices, 3)) if n_slices % 3 == 0 and n_slices >= 6 else list(range(n_slices))


class RadImageNetHead(nn.Module):
    """Frozen official RadImageNet ResNet-50 slice encoder + per-finding query head.

    Follows the public RadImageNet arm (E9/E11): every window-centre slice is one
    grey image scaled to [-1, 1]; its 2048-d global-average feature is a token tagged
    with slot and stack position; one query per finding cross-attends over all tokens
    and is fused with the mean token. Only the head trains.
    """
    def __init__(self, n_out=12, n_slots=6, n_windows=8, img=224, head_dim=512,
                 weights_path=None, encoder_chunk=128):
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict({k.split("backbone.", 1)[1]: v for k, v in state.items()},
                                          strict=True)
        for p in self.backbone.parameters(): p.requires_grad = False
        self.img, self.n_windows, self.encoder_chunk = img, n_windows, encoder_chunk
        dim = 2048
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_dim), nn.GELU())
        self.plane = nn.Parameter(torch.randn(n_slots, head_dim) * .01)
        self.position = nn.Parameter(torch.randn(n_windows, head_dim) * .01)
        self.query = nn.Parameter(torch.randn(n_out, head_dim) * .02)
        self.attn = nn.MultiheadAttention(head_dim, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(nn.LayerNorm(head_dim * 4), nn.Linear(head_dim * 4, head_dim),
                                  nn.GELU(), nn.Dropout(.15))
        self.weight = nn.Parameter(torch.randn(n_out, head_dim) * .02)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()                       # frozen BatchNorm statistics
        return self

    @torch.no_grad()
    def encode(self, x, present):
        # x uint8 (B,S,T,H,W) -> tokens (B, S*K, 2048) float16, token mask (B, S*K)
        B, S, T, H, W = x.shape
        centres = window_centres(T)[:self.n_windows]
        K = len(centres)
        keep = present.unsqueeze(-1).expand(B, S, K).reshape(-1)
        images = x[:, :, centres].reshape(B * S * K, 1, H, W)[keep].float().div(127.5).sub(1.)
        if H != self.img:
            images = nn.functional.interpolate(images, size=(self.img, self.img), mode="bilinear",
                                               align_corners=False)
        images = images.expand(-1, 3, -1, -1)
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            f = torch.cat([self.backbone(part).mean(dim=(2, 3)) for part in images.split(self.encoder_chunk)]) \
                if len(images) else images.new_zeros(0, 2048)
        tokens = torch.zeros(B * S * K, 2048, device=x.device, dtype=torch.float16)
        tokens[keep] = f.to(torch.float16)
        return tokens.view(B, S * K, 2048), keep.view(B, S * K).float()

    def head(self, feature, mask):
        n_slots = self.plane.shape[0]
        token = self.project(feature.float())
        token = token.view(len(token), n_slots, -1, token.shape[-1])
        token = token + self.plane[None, :, None] + self.position[None, None, :token.shape[2]]
        token = token.flatten(1, 2)
        pad = mask <= 0
        empty = pad.all(1)
        if empty.any():
            pad = pad.clone(); pad[empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token, key_padding_mask=pad, need_weights=False)[0]
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = mean.expand(-1, len(self.query), -1)
        fused = self.fuse(torch.cat([attended, mean, (attended - mean).abs(), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias

    def forward(self, x, present):
        m = present.clone()
        m[~m.any(dim=1)] = True
        return self.head(*self.encode(x, m)), None


def build_model(cfg, n_out, n_planes, pretrained=True):
    # A checkpoint's recipe rebuilds its own architecture; old recipes have no arch key.
    arch = cfg.get('arch', 'kneenet')
    if arch == 'kneenet':
        return KneeNet(cfg['backbone'], n_out, n_planes, pretrained=pretrained,
                       drop=cfg.get('drop', .3), img=cfg['img'],
                       normalization=cfg.get('normalization', 'legacy'))
    if arch == 'slot_query':
        return SlotQueryNet(cfg['backbone'], n_out, n_planes, pretrained=pretrained,
                            img=cfg['img'], drop=cfg.get('drop', .2),
                            hidden=cfg.get('head_hidden', 256),
                            unfreeze_last=cfg.get('unfreeze_last', 6),
                            normalization=cfg.get('normalization', 'pretrained'))
    if arch == 'reader25d':
        return Reader25D(cfg['backbone'], n_out, n_planes, pretrained=pretrained,
                         img=cfg['img'], drop=cfg.get('drop', .2),
                         dim=cfg.get('head_hidden', 384), drop_path=cfg.get('drop_path', .1),
                         band=tuple(cfg['preprocessing']['slice_band']),
                         train_windows=cfg.get('train_windows', 6),
                         normalization=cfg.get('normalization', 'pretrained'),
                         slot_dropout=cfg.get('slot_dropout', 0.0), aux_dim=cfg.get('aux_dim', 0))
    if arch == 'radhead':
        # The frozen encoder ships inside every checkpoint; weights_path is training-only.
        return RadImageNetHead(n_out, n_planes, n_windows=cfg.get('n_windows', 8), img=cfg['img'],
                               head_dim=cfg.get('head_hidden', 512),
                               weights_path=cfg.get('weights_path') if pretrained else None)
    raise ValueError(f'Unknown architecture {arch!r}')
