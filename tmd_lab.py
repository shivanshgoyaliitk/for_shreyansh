"""Shared harness for the TMD improvement notebooks.

Everything the notebooks have in common lives here so each notebook only has to
state its own improvement:

* data + BERT fine-tuning (optionally with synonym augmentation)
* the paper's InfoGAN (architecture of Tables 7/8, hyperparameters of Table 6)
* projection variants (paper's sampling search, ensembles, learned encoder, hybrid)
* the attack protocol of the paper (TextDefender / Li et al. 2021 settings)
* comparison against the paper's Table 1

Nothing here has been executed by its author. Read the warnings in the notebooks.
"""
import copy
import json
import math
import os
import random
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Paper numbers (Table 1, BERT rows) and per-dataset configuration
# --------------------------------------------------------------------------
METRICS = ['clean', 'pwws', 'textfooler', 'bertattack']
METRIC_LABEL = {'clean': 'clean acc', 'pwws': 'PWWS', 'textfooler': 'TextFooler', 'bertattack': 'BERT-Attack'}

# order: clean accuracy, PWWS, TextFooler, BERT-Attack (accuracy under attack, %)
PAPER = {
    'ag_news': {'vanilla': (94.39, 39.2, 27.9, 39.0), 'TMD': (94.29, 70.0, 50.0, 55.2),
                'best_baseline': (94.42, 39.3, 35.5, 52.3)},
    'imdb': {'vanilla': (92.15, 6.5, 2.3, 1.0), 'TMD': (92.17, 38.7, 44.2, 33.7),
             'best_baseline': (92.26, 41.4, 39.1, 30.7)},
    'yelp_polarity': {'vanilla': (95.28, 13.7, 10.5, 2.8), 'TMD': (95.24, 36.8, 40.9, 28.6),
                      'best_baseline': (95.39, 33.3, 31.2, 43.8)},
}

# The authors' fine-tuning (src/finetune/*_sweep.yaml + finetune.py): 10 epochs, batch 64, linear decay, a learning-rate
# grid chosen on validation accuracy, every epoch saved. We cannot afford the grid, so one rate is used, and the final
# epoch is kept (the wandb sweep compared runs by their last logged validation accuracy).
PAPER_FT = dict(epochs=10, lr=2e-5, bs=64, keep='last')

# Table 6 (BERT column), max_len from Appendix A.1, modify ratio from Sec. 4.1.
# NOTE: K = latent_dim (z) and d = code_dim (number of sub-manifolds), as in the authors' sweep configs
# (src/train/infogan_*_sweep.yaml: latent_dim in {50,100,200}, code_dim in {10..40}).
CFG = {
    'ag_news': dict(max_len=128, n_cls=4, K=50, d=20, g_lr=1e-4, d_lr=1e-4, p_lr=0.0, dg=1, rho=0.3),
    'imdb': dict(max_len=256, n_cls=2, K=50, d=40, g_lr=1e-4, d_lr=1e-4, p_lr=0.0, dg=1, rho=0.1),
    'yelp_polarity': dict(max_len=256, n_cls=2, K=100, d=20, g_lr=1e-4, d_lr=9e-5, p_lr=0.0, dg=1, rho=0.1),
}

# Recent `datasets` / `huggingface_hub` versions require namespaced repo ids
HF_ID = {'ag_news': 'fancyzhx/ag_news', 'imdb': 'stanfordnlp/imdb', 'yelp_polarity': 'fancyzhx/yelp_polarity'}

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _use_fork():
    """Make `import textattack` resolve to the authors' fork in TextDefender/ (pip textattack stays installed)."""
    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TextDefender')
    loaded = sys.modules.get('textattack')
    if loaded is not None and 'TextDefender' not in (getattr(loaded, '__file__', None) or ''):
        raise RuntimeError('pip textattack is already imported in this process: restart the runtime and call '
                           'tmd_lab.setup_env() before anything else imports textattack')
    if td not in sys.path:
        sys.path.insert(0, td)
    # The fork's ensemble wrapper imports TextDefender's `args` module, which pulls in every trainer and their
    # dependencies; only the ProgramArgs name is needed at import time, so a stub is enough.
    if 'args' not in sys.modules:
        stub = types.ModuleType('args')
        stub.ProgramArgs = type('ProgramArgs', (), {})
        sys.modules['args'] = stub


def setup_env(backend='fork'):
    """Select the attack backend ('fork' = the authors' code, 'pip' = our earlier rebuild) and keep TensorFlow (used
    only by TextAttack's sentence encoder) off the GPU. Call it before anything imports textattack."""
    global BACKEND
    BACKEND = backend
    if backend == 'fork':
        _use_fork()
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    try:  # resources TextAttack's constraints need; names changed across NLTK versions, so fetch all of them
        import nltk
        for r in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                  'universal_tagset', 'wordnet', 'omw-1.4', 'stopwords']:
            nltk.download(r, quiet=True)
    except Exception as e:
        print('nltk download skipped:', repr(e)[:120])
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:  # tensorflow missing or already initialised
        print('tensorflow GPU hiding skipped:', repr(e)[:120])
    torch.backends.cudnn.benchmark = False  # variable batch sizes (projection) make autotuning re-run constantly


# --------------------------------------------------------------------------
# InfoGAN (copied from src/models/generative_models/gan_based/infogan.py)
# --------------------------------------------------------------------------
class Gen(nn.Module):
    def __init__(self, latent_dim, code_dim):
        super().__init__()
        self.latent_dim, self.code_dim = latent_dim, code_dim
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim + code_dim, 512, 32, 1, 0, bias=False), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.ConvTranspose1d(512, 384, 4, 2, 1, bias=False), nn.BatchNorm1d(384), nn.ReLU(True),
            nn.ConvTranspose1d(384, 256, 4, 2, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.ConvTranspose1d(128, 1, 5, 3, 1, bias=False), nn.Tanh())

    def forward(self, z, c):
        inp = torch.cat([z, F.one_hot(c, self.code_dim).float()], dim=-1).unsqueeze(-1)
        return self.net(inp).squeeze(1)


class Dis(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 128, 5, 3, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 384, 4, 2, 1, bias=False), nn.BatchNorm1d(384), nn.LeakyReLU(0.2, True),
            nn.Conv1d(384, 512, 4, 2, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 768, 4, 2, 1, bias=False), nn.BatchNorm1d(768), nn.LeakyReLU(0.2, True))
        self.dis_head = nn.Conv1d(768, 1, 16, 1, 0, bias=False)
        self.enc_head = nn.Sequential(nn.Flatten(), nn.Linear(768 * 16, 128), nn.BatchNorm1d(128),
                                      nn.LeakyReLU(0.2, True), nn.Linear(128, code_dim))

    def forward(self, x):
        reps = self.body(x.unsqueeze(1))
        d_prob = torch.sigmoid(self.dis_head(reps).squeeze(-1))
        return d_prob, self.enc_head(reps)


class InfoGAN(nn.Module):
    def __init__(self, latent_dim, code_dim):
        super().__init__()
        self.latent_dim, self.code_dim = latent_dim, code_dim
        self.gen, self.dis = Gen(latent_dim, code_dim), Dis(code_dim)
        self.prior = nn.Parameter(torch.ones(code_dim) / code_dim)


def train_gan(E_tr, E_va, cfg, epochs=100, bs=1024, seed=42, log_every=5):
    """InfoGAN with learnable prior, trained as in the repo (D step each batch, G step every dg batches)."""
    torch.manual_seed(seed)
    gan = InfoGAN(latent_dim=cfg['K'], code_dim=cfg['d']).to(DEV)
    og = torch.optim.Adam(gan.gen.parameters(), lr=cfg['g_lr'], betas=(0.5, 0.999))
    od = torch.optim.Adam(gan.dis.parameters(), lr=cfg['d_lr'], betas=(0.5, 0.999))
    op = torch.optim.Adam([gan.prior], lr=cfg['p_lr'], betas=(0.5, 0.999)) if cfg['p_lr'] > 0 else None
    Etr, Eva = E_tr.to(DEV), E_va[:5000].to(DEV)
    best, best_state, pl_counter = 1e9, None, 1.0
    for ep in range(epochs):
        gan.train()
        perm = torch.randperm(len(Etr), device=DEV)
        for bi, i in enumerate(range(0, len(perm), bs)):
            real = Etr[perm[i:i + bs]]
            n = real.size(0)
            if n < 2:
                continue
            z = torch.randn(n, gan.latent_dim, device=DEV)
            c = torch.multinomial(torch.softmax(gan.prior, -1).detach(), n, replacement=True)
            one, zero = torch.ones(n, 1, device=DEV), torch.zeros(n, 1, device=DEV)
            fake = gan.gen(z, c).detach()
            pr, _ = gan.dis(real)
            pf, ef = gan.dis(fake)
            d_loss = (F.binary_cross_entropy(pr, one) + F.binary_cross_entropy(pf, zero)) / 2 + F.cross_entropy(ef, c)
            od.zero_grad(); d_loss.backward(); od.step()
            if bi % cfg['dg'] == 0:
                pf, ef = gan.dis(gan.gen(z, c))
                g_loss = F.binary_cross_entropy(pf, one) + F.cross_entropy(ef, c)
                og.zero_grad(); g_loss.backward(); og.step()
            if op is not None:
                with torch.no_grad():
                    e_real = gan.dis(real)[1]
                lp = F.log_softmax(gan.prior, -1)
                ce = -(torch.softmax(e_real, -1) * lp).sum(-1).mean()
                ent = -(lp.exp() * lp).sum()
                pl_counter *= 0.999
                p_loss = ce - pl_counter * 100.0 * ent
                op.zero_grad(); p_loss.backward(); op.step()
        if ep % log_every == log_every - 1 or ep == epochs - 1:
            rec = project(gan, Eva)[1].mean().item()
            print(f'  gan epoch {ep + 1}: val reconstruction loss {rec:.4f}', flush=True)
            if rec < best:
                best, best_state = rec, copy.deepcopy(gan.state_dict())
    gan.load_state_dict(best_state)
    return gan.eval()


# Table 4 search ranges of the paper (K = latent dim, d = code dim, dg = D iterations per G update)
GAN_SPACE = dict(
    g_lr=[1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4],
    d_lr=[7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3],
    p_lr=[0.0, 1e-4, 1e-3, 1e-2],
    K=[50, 100, 200], d=[10, 20, 30, 40], dg=[1, 2, 3, 4, 5])


def search_gans(lab, tag, E_tr, E_va, trials=8, epochs=100, seed=0):
    """Random hyperparameter search like the authors' wandb sweeps: train `trials` InfoGANs and keep the one with the
    lowest *validation* reconstruction loss (never test data). Each trial is cached, so an interrupted search resumes.
    The winner is saved as infogan_<tag>_best.*, which Lab.get_gan then picks up automatically."""
    rng = random.Random(seed)
    Eva = E_va[:5000].to(DEV)
    best = None
    for t in range(trials):
        hp = {k: rng.choice(v) for k, v in GAN_SPACE.items()}  # same sequence every call, so cached trials line up
        cfg = dict(lab.cfg, **hp)
        path = lab._path(f'infogan_{tag}_search{t}.pt')
        gan = InfoGAN(latent_dim=cfg['K'], code_dim=cfg['d']).to(DEV)
        if os.path.exists(path):
            gan.load_state_dict(torch.load(path))
            gan.eval()
        else:
            print(f'--- search trial {t}: {hp}', flush=True)
            gan = train_gan(E_tr, E_va, cfg, epochs=epochs, seed=lab.seed + t)
            torch.save(gan.state_dict(), path)
        torch.manual_seed(1234)
        rec = project(gan, Eva)[1].mean().item()
        print(f'trial {t}: val reconstruction loss {rec:.4f}  {hp}', flush=True)
        if best is None or rec < best[0]:
            best = (rec, t, hp)
    rec, t, hp = best
    gan = InfoGAN(latent_dim=hp['K'], code_dim=hp['d']).to(DEV)
    gan.load_state_dict(torch.load(lab._path(f'infogan_{tag}_search{t}.pt')))
    torch.save(gan.state_dict(), lab._path(f'infogan_{tag}_best.pt'))
    json.dump(dict(hp=hp, rec=rec, trial=t), open(lab._path(f'infogan_{tag}_best.json'), 'w'))
    print(f'best: trial {t}, val reconstruction loss {rec:.4f}, {hp}')
    return gan.eval(), hp


# --------------------------------------------------------------------------
# Learned projection encoder (replaces the random search over z)
# --------------------------------------------------------------------------
class LatentEncoder(nn.Module):
    """q(z | t, c): amortised GAN inversion. Predicts the latent that makes G(z, c) closest to t."""

    def __init__(self, latent_dim, code_dim, emb_dim=768, hidden=1024):
        super().__init__()
        self.code_dim = code_dim
        self.net = nn.Sequential(nn.Linear(emb_dim + code_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.mu, self.logvar = nn.Linear(hidden, latent_dim), nn.Linear(hidden, latent_dim)

    def forward(self, t, c):
        h = self.net(torch.cat([t, F.one_hot(c, self.code_dim).float()], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


def train_encoder(gan, E_tr, E_va, epochs=30, bs=1024, lr=1e-3, beta=1e-3, cyc=0.05,
                  noise_std=0.05, E_aug=None, aug_p=0.5, seed=0, log_every=5):
    """Train q(z|t,c) against the frozen generator.

    loss = ||G(z, c) - t_clean||  +  beta * KL(q || N(0, I))  +  cyc * ||mu(G(z0, c0), c0) - z0||^2
    The encoder input is the clean embedding plus gaussian noise, and, when E_aug is given, with
    probability aug_p the embedding of a synonym-perturbed version of the same text. The target stays the
    clean embedding, so the encoder is trained as a denoising projector, which is what a defense needs.
    """
    torch.manual_seed(seed)
    gan.eval()
    for p in gan.parameters():
        p.requires_grad_(False)
    enc = LatentEncoder(gan.latent_dim, gan.code_dim).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Etr, Eva = E_tr.to(DEV), E_va[:5000].to(DEV)
    Eaug = E_aug.to(DEV) if E_aug is not None else None
    prior = torch.softmax(gan.prior.detach(), -1)
    best, best_state = 1e9, None
    for ep in range(epochs):
        enc.train()
        perm = torch.randperm(len(Etr), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            tgt = Etr[idx]
            n = tgt.size(0)
            inp = tgt
            if Eaug is not None:
                use = (torch.rand(n, 1, device=DEV) < aug_p)
                inp = torch.where(use, Eaug[idx], tgt)
            inp = inp + noise_std * torch.randn_like(inp)
            with torch.no_grad():
                c = gan.dis(inp)[1].argmax(-1)
            mu, lv = enc(inp, c)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            rec = torch.norm(gan.gen(z, c) - tgt, dim=-1).mean()
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(-1)).mean()
            z0 = torch.randn(n, gan.latent_dim, device=DEV)
            c0 = torch.multinomial(prior, n, replacement=True)
            with torch.no_grad():
                t0 = gan.gen(z0, c0)
            mu0, _ = enc(t0, c0)
            cyc_loss = (mu0 - z0).pow(2).sum(-1).mean()
            loss = rec + beta * kl + cyc * cyc_loss
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % log_every == log_every - 1 or ep == epochs - 1:
            val = project(gan, Eva, enc=enc, mode='encoder')[1].mean().item()
            print(f'  encoder epoch {ep + 1}: val reconstruction loss {val:.4f}', flush=True)
            if val < best:
                best, best_state = val, copy.deepcopy(enc.state_dict())
    enc.load_state_dict(best_state)
    return enc.eval()


# --------------------------------------------------------------------------
# Projection (Eq. 8 and variants)
# --------------------------------------------------------------------------
@torch.no_grad()
def project(gan, t, k=20, chunk=256, enc=None, mode='sample', sigma_scale=1.0, n_prior=0):
    """Project embeddings t of shape (B, 768) onto the generator's manifold.

    mode 'sample'  : paper. c = argmax Q(c|t); draw k z ~ N(0, I); keep the generated embedding closest to t (L2).
    mode 'encoder' : z = mu(t, c) from the learned encoder; one deterministic generator pass.
    mode 'hybrid'  : candidates = encoder mean + k draws around it (+ n_prior draws from the prior); keep closest.
    Returns (projected embeddings, distance of each to its input).
    """
    gan.eval()
    if enc is not None:
        enc.eval()
    outs, dists = [], []
    for i in range(0, t.size(0), chunk):
        x = t[i:i + chunk]
        bs = x.size(0)
        c = gan.dis(x)[1].argmax(-1)
        if mode == 'sample':
            z = torch.randn(bs, k, gan.latent_dim, device=x.device)
        elif mode == 'fixed_bank':  # method 11: the paper's search with one fixed bank of k latents (deterministic TMD)
            z = _fixed_bank(k, gan.latent_dim, x.device)[None].expand(bs, k, -1)
        elif mode == 'encoder':  # method 9
            out = enc(x, c)
            z = (out[0] if isinstance(out, tuple) else out)[:, None]
        elif mode == 'enc_local':  # method 10: encoder output + (k-1) draws around it with std sigma_scale; keep closest
            out = enc(x, c)
            mu = out[0] if isinstance(out, tuple) else out
            noise = torch.randn(bs, k, gan.latent_dim, device=x.device) * sigma_scale
            noise[:, 0] = 0.0
            z = mu[:, None] + noise
        elif mode == 'hybrid':
            mu, lv = enc(x, c)
            std = (0.5 * lv).exp() * sigma_scale
            z = torch.cat([mu[:, None], mu[:, None] + std[:, None] * torch.randn(bs, k, gan.latent_dim, device=x.device)], 1)
            if n_prior > 0:
                z = torch.cat([z, torch.randn(bs, n_prior, gan.latent_dim, device=x.device)], 1)
        else:
            raise ValueError(mode)
        kk = z.size(1)
        fake = gan.gen(z.reshape(bs * kk, -1), c[:, None].expand(bs, kk).reshape(-1)).view(bs, kk, -1)
        d = torch.norm(fake - x[:, None], dim=-1)
        m = d.argmin(1)
        outs.append(fake[torch.arange(bs, device=x.device), m])
        dists.append(d.min(1).values)
    return torch.cat(outs), torch.cat(dists)


_BANKS = {}


def _fixed_bank(k, latent_dim, device, seed=0):
    key = (k, latent_dim, str(device), seed)
    if key not in _BANKS:
        g = torch.Generator(device='cpu').manual_seed(seed)
        _BANKS[key] = torch.randn(k, latent_dim, generator=g).to(device)
    return _BANKS[key]


# --------------------------------------------------------------------------
# Adversarially trained projection encoder (port of the team's infogan_enc.py / train_encoder.py)
# --------------------------------------------------------------------------
class ProjEncoder(nn.Module):
    """E(t, c) -> z (infogan_enc.Encoder): MLP 768 + code -> 1024 -> 512 -> latent, LayerNorm + LeakyReLU(0.2).
    Conditioning on the sub-manifold id keeps InfoGAN's disconnected structure."""

    def __init__(self, latent_dim, code_dim, emb_dim=768, hidden_dims=(1024, 512)):
        super().__init__()
        self.code_dim = code_dim
        dims = [emb_dim + code_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, c):
        return self.net(torch.cat([t, F.one_hot(c, self.code_dim).type_as(t)], dim=-1))


def typical_set_penalty(z, margin=0.1):
    """Penalise latents whose norm leaves the N(0, I) typical shell (|z| ~ sqrt(d))."""
    r = z.norm(dim=-1) / math.sqrt(z.shape[-1])
    return F.relu(r - (1 + margin)).pow(2).mean()


def train_proj_encoder(gan, E_tr, Y_tr=None, E_va=None, Y_va=None, E_aug=None, adv=None, head=None, init=None,
                       epochs=30, bs=512, lr=1e-3, w_lat=1.0, w_rec=1.0, w_den=0.0, w_typ=0.1, w_adv=1.0,
                       adv_bs=256, w_cls=0.0, noise_std=0.05, seed=42, log_every=5):
    """train_encoder.py: G, D/Q and the prior frozen (eval mode), only E trained.
      L = w_lat ||E(G(z,c),c) - z||^2 + w_rec ||G(E(t,c_t),c_t) - t||^2 + w_den ||G(E(t~,c~),c~) - t||^2
        + w_typ typical-set + w_adv ||G(E(t_adv,c_a),c_a) - t_clean||^2 + w_cls CE(head(projected), y)
    adv = (adversarial embeddings, their clean embeddings, labels); init = encoder to start from (previous round)."""
    torch.manual_seed(seed)
    gan.eval()
    for p in gan.parameters():
        p.requires_grad_(False)
    enc = ProjEncoder(gan.latent_dim, gan.code_dim).to(DEV)
    if init is not None:
        enc.load_state_dict(init.state_dict())
    if head is not None:
        head = copy.deepcopy(head).to(DEV).eval()
        for p in head.parameters():
            p.requires_grad_(False)
    assert w_cls == 0 or (head is not None and Y_tr is not None), 'w_cls needs head and Y_tr'
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    prior = torch.softmax(gan.prior.detach(), -1)
    Etr = E_tr.to(DEV)
    Ytr = Y_tr.to(DEV) if Y_tr is not None else None
    Eaug = E_aug.to(DEV) if E_aug is not None else None
    if adv is not None:
        adv = tuple(a.to(DEV) for a in adv)
    zd = gan.latent_dim
    for ep in range(epochs):
        enc.train()
        perm = torch.randperm(len(Etr), device=DEV)
        logs = dict(lat=0.0, rec=0.0, den=0.0, typ=0.0, adv=0.0, cls=0.0)
        nb = 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            t = Etr[idx]
            b = len(t)
            z = torch.randn(b, zd, device=DEV)
            c = torch.multinomial(prior, b, replacement=True)
            with torch.no_grad():
                tg = gan.gen(z, c)
            e_tg = enc(tg, c)
            l_lat = F.mse_loss(e_tg, z)
            with torch.no_grad():
                ct = gan.dis(t)[1].argmax(-1)
            zt = enc(t, ct)
            rec_t = gan.gen(zt, ct)
            l_rec = F.mse_loss(rec_t, t)
            l_cls = F.cross_entropy(head(rec_t), Ytr[idx]) if w_cls > 0 else torch.zeros((), device=DEV)
            l_typ = typical_set_penalty(torch.cat([zt, e_tg], 0))
            l_den = torch.zeros((), device=DEV)
            if w_den > 0:
                tn = Eaug[idx] if Eaug is not None else (t + noise_std * torch.randn_like(t)).clamp(-1, 1)
                with torch.no_grad():
                    cn = gan.dis(tn)[1].argmax(-1)
                l_den = F.mse_loss(gan.gen(enc(tn, cn), cn), t)
            l_adv = torch.zeros((), device=DEV)
            if adv is not None and w_adv > 0:
                j = torch.randint(0, len(adv[0]), (min(adv_bs, len(adv[0])),), device=DEV)
                ta, tc, ya = adv[0][j], adv[1][j], adv[2][j]
                with torch.no_grad():
                    ca = gan.dis(ta)[1].argmax(-1)
                rec_a = gan.gen(enc(ta, ca), ca)
                l_adv = F.mse_loss(rec_a, tc)
                if w_cls > 0:
                    l_cls = l_cls + F.cross_entropy(head(rec_a), ya)
            loss = w_lat * l_lat + w_rec * l_rec + w_den * l_den + w_typ * l_typ + w_adv * l_adv + w_cls * l_cls
            opt.zero_grad(); loss.backward(); opt.step()
            for k_, v in zip(logs, (l_lat, l_rec, l_den, l_typ, l_adv, l_cls)):
                logs[k_] += v.item()
            nb += 1
        if E_va is not None and (ep % log_every == log_every - 1 or ep == epochs - 1):
            th, d = project(gan, E_va[:5000].to(DEV), enc=enc, mode='encoder')
            msg = f'val RL {d.mean().item():.3f}'
            if head is not None and Y_va is not None:
                msg += f', projected val acc {(head(th).argmax(-1).cpu() == Y_va[:5000]).float().mean().item():.4f}'
            print(f'  encoder epoch {ep + 1}: ' + ' '.join(f'{k_}={v / nb:.4f}' for k_, v in logs.items()) + ', ' + msg,
                  flush=True)
    return enc.eval()


def train_head_adv(lab, E_tr, Y_tr, adv, epochs=3, lr=1e-3, bs=512, seed=0):
    """Control for the adversarial encoder: adversarial training of ONLY the classifier head on the same pairs
    (clean train embeddings + adversarial embeddings with their labels), no GAN and no encoder. If this matches the
    encoder's transfer robustness, the manifold is not what helps."""
    torch.manual_seed(seed)
    head = copy.deepcopy(lab.model.classifier).to(DEV)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Etr, Ytr = E_tr.to(DEV), Y_tr.to(DEV)
    ta, ya = adv[0].to(DEV), adv[2].to(DEV)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Etr), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            j = torch.randint(0, len(ta), (min(bs // 2, len(ta)),), device=DEV)
            loss = F.cross_entropy(head(Etr[idx]), Ytr[idx]) + F.cross_entropy(head(ta[j]), ya[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def mcnemar(lab, tag_a, tag_b, attack, n):
    """Exact McNemar test on the same examples: robust = attack failed. Needs per-example logs (fork backend)."""
    def robust(tag):
        rows = [json.loads(line) for line in open(examples_path(lab, tag, n, attack))]
        return {r['i']: r['status'] == 'F' for r in rows if r['status'] != 'D'}
    a, b = robust(tag_a), robust(tag_b)
    common = sorted(set(a) & set(b))
    only_a = sum(a[i] and not b[i] for i in common)
    only_b = sum(b[i] and not a[i] for i in common)
    m = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(min(only_a, only_b) + 1)) / 2 ** m) if m else 1.0
    return dict(attack=attack, examples=len(common), robust_only_a=only_a, robust_only_b=only_b, p=round(p, 4),
                a=tag_a, b=tag_b)


def transfer_eval(lab, victims, source_tag, attack, n, repeats=5):
    """Transfer test: texts that the attack fooled on `source_tag` (normally vanilla), fed once to each defense; the
    attack never queries the defense, so score masking cannot help. Stochastic defenses are averaged over repeats."""
    import pandas as pd
    rows = [json.loads(line) for line in open(examples_path(lab, source_tag, n, attack))]
    rows = [r for r in rows if r['status'] == 'S']
    clean, adv = [r['text'] for r in rows], [r['adv_text'] for r in rows]
    y = torch.tensor([r['label'] for r in rows])
    out = []
    for name, v in victims.items():
        stochastic = v.mode != 'none' and v.proj_mode in ('sample', 'enc_local', 'hybrid')
        reps = repeats if stochastic else 1
        def acc(texts):
            pred = torch.cat([v.scores(texts[i:i + 64]).argmax(-1).cpu() for i in range(0, len(texts), 64)])
            return (pred == y).float().mean().item() * 100
        out.append(dict(defense=name, n=len(rows), clean_acc=round(sum(acc(clean) for _ in range(reps)) / reps, 1),
                        adv_acc=round(sum(acc(adv) for _ in range(reps)) / reps, 1)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Synonym augmentation (counter-fitted neighbours, as in SAFER / TextDefender)
# --------------------------------------------------------------------------
class SynonymAugmenter:
    def __init__(self, neighbors_json, rate=0.3, top_n=10, seed=0):
        self.nb = json.load(open(neighbors_json))
        self.rate, self.top_n = rate, top_n
        self.rng = random.Random(seed)

    def __call__(self, text):
        words = text.split(' ')
        for i, w in enumerate(words):
            cands = self.nb.get(w.lower())
            if cands and self.rng.random() < self.rate:
                words[i] = self.rng.choice(cands[:self.top_n])
        return ' '.join(words)


# --------------------------------------------------------------------------
# Victim: language model + (optional) projection + head
# --------------------------------------------------------------------------
class Victim:
    """Scores texts. mode 'none' = vanilla classifier; otherwise project the pooled embedding first.

    R > 1 draws R independent projections and averages ('prob': mean softmax, 'logit': mean logits).
    head=None uses the fine-tuned model's own classifier layer.
    """

    def __init__(self, lab, gan=None, enc=None, head=None, mode='none', proj_mode='sample', k=20,
                 R=1, combine='prob', sigma_scale=1.0, n_prior=0):
        self.lab, self.gan, self.enc, self.head = lab, gan, enc, head
        self.mode, self.proj_mode, self.k, self.R, self.combine = mode, proj_mode, k, R, combine
        self.sigma_scale, self.n_prior = sigma_scale, n_prior

    @torch.no_grad()
    def from_emb(self, t):
        head = self.head if self.head is not None else self.lab.model.classifier
        head.eval()
        if self.mode == 'none':
            return head(t)
        outs = []
        for _ in range(self.R):
            th = project(self.gan, t, k=self.k, enc=self.enc, mode=self.proj_mode,
                         sigma_scale=self.sigma_scale, n_prior=self.n_prior)[0]
            outs.append(head(th))
        if len(outs) == 1:
            return outs[0]
        if self.combine == 'prob':
            return torch.stack([torch.softmax(o, -1) for o in outs]).mean(0)
        return torch.stack(outs).mean(0)

    @torch.no_grad()
    def scores(self, texts):
        return self.from_emb(self.lab.pooled(texts))

    def clean_accuracy(self, E, Y, bs=512):
        ok = 0
        for i in range(0, len(E), bs):
            ok += (self.from_emb(E[i:i + bs].to(DEV)).argmax(-1).cpu() == Y[i:i + bs]).sum().item()
        return ok / len(E)


def train_head(lab, gan, E_tr, Y_tr, E_va, Y_va, enc=None, proj_mode='sample', k=30, epochs=5, lr=1e-3,
               bs=512, clean_mix=0.0, seed=0):
    """Fine-tune only the classification layer on *projected* embeddings (fresh stochastic projections each epoch),
    so the head sees the same reconstruction noise it will see at test time. The original head is the fallback:
    the returned head is the epoch with the best projected validation accuracy (epoch 0 = untouched head)."""
    torch.manual_seed(seed)
    head = copy.deepcopy(lab.model.classifier).to(DEV)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Etr, Ytr = E_tr.to(DEV), Y_tr.to(DEV)

    def val_acc():
        v = Victim(lab, gan=gan, enc=enc, head=head, mode='tmd', proj_mode=proj_mode, k=k)
        return v.clean_accuracy(E_va, Y_va)

    best, best_state = val_acc(), copy.deepcopy(head.state_dict())
    print(f'  head epoch 0 (original head): projected val acc {best:.4f}')
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Etr), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, y = Etr[idx], Ytr[idx]
            th = project(gan, x, k=k, enc=enc, mode=proj_mode)[0]
            if clean_mix > 0:
                th = torch.where(torch.rand(x.size(0), 1, device=DEV) < clean_mix, x, th)
            loss = F.cross_entropy(head(th), y)
            opt.zero_grad(); loss.backward(); opt.step()
        va = val_acc()
        print(f'  head epoch {ep + 1}: projected val acc {va:.4f}', flush=True)
        if va > best:
            best, best_state = va, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval()


# --------------------------------------------------------------------------
# Lab: data, fine-tuning, embeddings, cached artifacts
# --------------------------------------------------------------------------
class Lab:
    def __init__(self, dataset='ag_news', work='/content/tmd_run', model_name='bert-base-uncased', seed=42):
        from transformers import AutoTokenizer
        self.dataset, self.model_name, self.seed = dataset, model_name, seed
        self.cfg = CFG[dataset]
        self.work = os.path.join(work, dataset)
        os.makedirs(self.work, exist_ok=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.amp = torch.bfloat16 if (DEV.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        self.model, self.model_tag = None, None
        self._load_data()

    # ----- data
    def _load_data(self):
        from datasets import load_dataset
        raw = load_dataset(HF_ID[self.dataset])
        if self.dataset == 'yelp_polarity':  # paper: 63k-sample subset
            idx = random.Random(42).sample(range(len(raw['train'])), 63000)
            raw['train'] = raw['train'].select(idx)
        split = raw['train'].train_test_split(0.1, seed=42)  # 10% validation
        self.train_ds, self.val_ds, self.test_ds = split['train'], split['test'], raw['test']
        print('train/val/test:', len(self.train_ds), len(self.val_ds), len(self.test_ds))

    def batches(self, ds, bs, shuffle=False):
        idx = list(range(len(ds)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = ds[idx[i:i + bs]]
            yield b['text'], torch.tensor(b['label'])

    def encode(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.cfg['max_len'], padding=True,
                        return_tensors='pt').to(DEV)

    # ----- model
    def _path(self, name):
        return os.path.join(self.work, name)

    def use_model(self, tag):
        from transformers import AutoModelForSequenceClassification
        if self.model_tag == tag:
            return self.model
        self.model = AutoModelForSequenceClassification.from_pretrained(self._path(f'finetuned_{tag}')).to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model_tag = tag
        return self.model

    def finetune(self, tag='base', epochs=3, lr=3e-5, bs=32, augment_fn=None, aug_prob=0.5, keep='best'):
        """Fine-tune (or load) the classifier. augment_fn(text)->text is applied to a fraction aug_prob of examples.

        keep='best' saves the epoch with the best validation accuracy; keep='last' saves the final epoch, as the
        authors' finetune.py + wandb sweep did (see PAPER_FT). Training state is checkpointed after every epoch, so an
        interrupted run resumes at the next epoch; a DONE marker records that training finished."""
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        path = self._path(f'finetuned_{tag}')
        state_path = self._path(f'finetune_state_{tag}.pt')
        done = os.path.join(path, 'DONE')
        legacy = os.path.exists(os.path.join(path, 'config.json')) and not os.path.exists(state_path)
        if os.path.exists(done) or legacy:
            print(f'[{tag}] loading fine-tuned model')
            return self.use_model(tag)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.cfg['n_cls']).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        steps = epochs * math.ceil(len(self.train_ds) / bs)
        sched = get_linear_schedule_with_warmup(opt, 0, steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.amp == torch.float16))
        best, start = 0.0, 0
        if os.path.exists(state_path):
            st = torch.load(state_path, map_location=DEV)
            model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
            sched.load_state_dict(st['sched']); scaler.load_state_dict(st['scaler'])
            best, start = st['best'], st['epoch']
            print(f'[{tag}] resuming fine-tuning at epoch {start}')
        for ep in range(start, epochs):
            model.train()
            t0 = time.time()
            for x, y in self.batches(self.train_ds, bs, shuffle=True):
                if augment_fn is not None:
                    x = [augment_fn(s) if random.random() < aug_prob else s for s in x]
                with torch.autocast('cuda', dtype=self.amp):
                    loss = model(**self.encode(x), labels=y.to(DEV)).loss
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            model.eval()
            ok = 0
            with torch.no_grad():
                for x, y in self.batches(self.val_ds, 256):
                    with torch.autocast('cuda', dtype=self.amp):
                        ok += (model(**self.encode(x)).logits.argmax(-1).cpu() == y).sum().item()
            va = ok / len(self.val_ds)
            print(f'[{tag}] epoch {ep}: val acc {va:.4f} ({time.time() - t0:.0f}s)', flush=True)
            if keep == 'last' or va > best:
                model.save_pretrained(path)
            best = max(best, va)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), best=best, epoch=ep + 1), state_path)
        open(done, 'w').write(json.dumps(dict(epochs=epochs, lr=lr, bs=bs, keep=keep, best_val=best)))
        os.remove(state_path)
        del model
        self.model_tag = None
        return self.use_model(tag)

    @torch.no_grad()
    def pooled(self, texts):
        """Text embedding t = F(x): BERT pooler output (tanh, in [-1, 1])."""
        with torch.autocast('cuda', dtype=self.amp):
            return self.model.bert(**self.encode(texts)).pooler_output.float()

    def embeddings(self, split, tag, augment_fn=None, name=None):
        """Pooled embeddings of a split under the current model, cached on disk."""
        assert self.model_tag == tag, 'call use_model/finetune first'
        path = self._path(f'emb_{tag}_{name or split}.pt')
        if os.path.exists(path):
            return torch.load(path)
        ds = {'train': self.train_ds, 'val': self.val_ds, 'test': self.test_ds}[split]
        E, Y = [], []
        for x, y in self.batches(ds, 256):
            if augment_fn is not None:
                x = [augment_fn(s) for s in x]
            E.append(self.pooled(x).cpu()); Y.append(y)
        out = (torch.cat(E), torch.cat(Y))
        torch.save(out, path)
        return out

    def get_gan(self, tag, E_tr, E_va, epochs=100, bs=1024):
        meta = self._path(f'infogan_{tag}_best.json')
        if os.path.exists(meta):  # a hyperparameter search (search_gans) already picked a winner
            hp = json.load(open(meta))['hp']
            gan = InfoGAN(latent_dim=hp['K'], code_dim=hp['d']).to(DEV)
            gan.load_state_dict(torch.load(self._path(f'infogan_{tag}_best.pt')))
            print(f'[{tag}] loaded searched InfoGAN {hp}')
            return gan.eval()
        path = self._path(f"infogan_{tag}_L{self.cfg['K']}C{self.cfg['d']}.pt")
        gan = InfoGAN(latent_dim=self.cfg['K'], code_dim=self.cfg['d']).to(DEV)
        if os.path.exists(path):
            gan.load_state_dict(torch.load(path))
            print(f'[{tag}] loaded InfoGAN')
            return gan.eval()
        gan = train_gan(E_tr, E_va, self.cfg, epochs=epochs, bs=bs, seed=self.seed)
        torch.save(gan.state_dict(), path)
        return gan

    def get_encoder(self, name, gan, E_tr, E_va, force=False, **kw):
        path = self._path(f'encoder_{name}_L{gan.latent_dim}C{gan.code_dim}.pt')
        if os.path.exists(path) and not force:
            enc = LatentEncoder(gan.latent_dim, gan.code_dim).to(DEV)
            enc.load_state_dict(torch.load(path))
            print(f'[{name}] loaded encoder')
            return enc.eval()
        enc = train_encoder(gan, E_tr, E_va, **kw)
        torch.save(enc.state_dict(), path)
        return enc

    # ----- attack protocol
    def attack_pairs(self, n, split='test', seed=None):
        """TextDefender: keep texts with more than 4 words, sample n without replacement (seeded). The test sample for
        a smaller n is the prefix of the sample for a larger n. split='train' gives adversarial-training material."""
        ds = {'train': self.train_ds, 'val': self.val_ds, 'test': self.test_ds}[split]
        texts, labels = ds['text'], ds['label']
        idx = [i for i, t in enumerate(texts) if len(t.split(' ')) > 4]
        rs = np.random.RandomState(self.seed if seed is None else seed)
        sel = rs.choice(idx, size=min(n, len(idx)), replace=False)
        return [(texts[i], int(labels[i])) for i in sel]


# --------------------------------------------------------------------------
# Attacks with the paper's protocol
# --------------------------------------------------------------------------
# BACKEND 'fork' (default): the authors' own TextAttack fork (TextDefender/textattack), with the attack built exactly
# like TextDefender/utils/textattack_utils.py::build_english_attacker and results counted like its SimplifidResult.
# This is the code behind the paper's Table 1.
# BACKEND 'pip': our earlier rebuild of the same constraints on pip textattack. Kept for the record only: on the same
# model, InfoGAN and 200 test examples it gave TMD/PWWS 47.0, while the fork gives 66.0 (paper: 70.0).
BACKEND = 'fork'


def results_path(lab, tag, n=1000):
    """Results of one victim at one N_EVAL: <prefix><tag>_n<n>.json, so runs of different sizes never overwrite each
    other. Results saved before files were split by n (<prefix><tag>.json) are migrated on first use."""
    prefix = 'results_fork_' if BACKEND == 'fork' else 'results_'
    path = lab._path(f'{prefix}{tag}_n{n}.json')
    legacy = lab._path(f'{prefix}{tag}.json')
    if not os.path.exists(path) and os.path.exists(legacy):
        old = json.load(open(legacy))
        keep = {k: v for k, v in old.items()
                if k == 'clean' or (isinstance(v, dict) and v.get('n_requested', v.get('n')) == n)}
        if len(keep) > 1:
            json.dump(keep, open(path, 'w'), indent=1)
    return path


def _fork_wrapper(victim):
    from textattack.models.wrappers import ModelWrapper

    class VictimWrapper(ModelWrapper):
        def __init__(self, victim):
            self.victim = victim
            self.model = victim.lab.model

        def __call__(self, text_list):
            out = []
            with torch.no_grad():
                for i in range(0, len(text_list), 32):  # TextDefender's wrapper batch size
                    out.append(self.victim.scores(list(text_list[i:i + 32])).float().cpu())
            return torch.cat(out)

        def get_grad(self, text_input):
            raise NotImplementedError

    return VictimWrapper(victim)


def _build_attack_fork(name, victim, rho, K=50, sim=0.840845057):
    """build_english_attacker (TextDefender/utils/textattack_utils.py) for pwws, textfooler and bae (BERT-Attack)."""
    import textattack
    from textattack.attack_recipes import BERTAttackLi2020, PWWSRen2019, TextFoolerJin2019
    from textattack.constraints.overlap import MaxWordsPerturbed
    from textattack.constraints.pre_transformation import InputColumnModification
    from textattack.constraints.semantics import WordEmbeddingDistance
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.goal_functions import UntargetedClassification
    from textattack.shared.attack import Attack
    from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
    assert 'TextDefender' in textattack.__file__, f'expected the authors\' fork, got {textattack.__file__}'

    wrapper = _fork_wrapper(victim)
    if name == 'pwws':
        attacker = PWWSRen2019.build(wrapper)
    elif name == 'textfooler':
        attacker = TextFoolerJin2019.build(wrapper)
    else:
        attacker = BERTAttackLi2020.build(wrapper)
        attacker.transformation = WordSwapMaskedLM(method='bert-attack', max_candidates=K)
    if name in ['textfooler', 'pwws']:
        attacker.transformation = WordSwapEmbedding(max_candidates=K)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
    attacker.constraints.append(MaxWordsPerturbed(max_percent=rho))
    attacker.constraints.append(UniversalSentenceEncoder(threshold=sim, metric='angular', compare_against_original=False,
                                                         window_size=15, skip_text_shorter_than_window=True))
    attacker.pre_transformation_constraints.append(InputColumnModification(['premise', 'hypothesis'], {'premise'}))
    attacker.goal_function = UntargetedClassification(wrapper, query_budget=K)  # fork: budget = K x number of words
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def examples_path(lab, tag, n, attack):
    """Per-example log of one attack run (fork backend): one JSON line per example, in sample order."""
    return lab._path(f'examples_fork_{tag}_n{n}_{attack}.jsonl')


def _run_attack_fork(lab, victim, name, pairs, log_path, chunk=50):
    """Attack `pairs` one by one with the fork, appending one JSON line per example to log_path (resumes from it).
    status: S succeeded, F failed (= robust), K skipped (misclassified before the attack), D dropped (the attack raised
    an exception; the fork silently drops these). Accuracy under attack is counted like TextDefender's
    SimplifidResult: F / (S + F + K), dropped examples excluded."""
    import faulthandler
    import gc
    from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
    attack = _build_attack_fork(name, victim, lab.cfg['rho'], K=50)
    global _STACK_FILE  # watchdog: if a chunk takes > 3 min, dump every thread's stack to /content/stack_dump.txt
    _STACK_FILE = open('/content/stack_dump.txt' if os.path.isdir('/content') else 'stack_dump.txt', 'w')
    rows = [json.loads(line) for line in open(log_path)] if os.path.exists(log_path) else []
    rows = [r for r in rows if r['i'] < len(pairs) and r['text'] == pairs[r['i']][0]]
    if rows:
        print(f'    {name}: resuming at example {len(rows)}', flush=True)
    with open(log_path, 'w') as f:  # rewrite the kept rows, then append
        for r in rows:
            f.write(json.dumps(r) + '\n')
    t_run = time.time()
    with open(log_path, 'a') as f:
        for i in range(len(rows), len(pairs)):
            if i % chunk == 0:
                faulthandler.dump_traceback_later(180, repeat=True, file=_STACK_FILE)
            text, label = pairs[i]
            t0 = time.time()
            res = list(attack.attack_dataset([(text, label)]))
            row = dict(i=i, label=label, text=text, seconds=round(time.time() - t0, 2), status='D')
            if res:
                r = res[0]
                row['status'] = ('S' if isinstance(r, SuccessfulAttackResult) else 'F' if isinstance(r, FailedAttackResult)
                                 else 'K' if isinstance(r, SkippedAttackResult) else 'O')
                row['queries'] = int(r.perturbed_result.num_queries)
                if row['status'] == 'S':
                    adv = r.perturbed_result.attacked_text
                    row['adv_text'] = adv.text
                    row['pct_changed'] = round(100 * len(adv.all_words_diff(r.original_result.attacked_text))
                                               / max(1, len(r.original_result.attacked_text.words)), 2)
            rows.append(row)
            f.write(json.dumps(row) + '\n'); f.flush()
            if (i + 1) % chunk == 0 or i + 1 == len(pairs):
                faulthandler.cancel_dump_traceback_later()
                try:
                    attack.clear_cache()
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                s_ = summarize_rows(rows)
                print(f'    {name}: {i + 1}/{len(pairs)} done, acc under attack {s_["acc"]:.1f}% '
                      f'(dropped: {s_["dropped"]}), {(time.time() - t_run) / 60:.1f} min this session', flush=True)
    return summarize_rows(rows)


def summarize_rows(rows):
    cnt = {k: sum(r['status'] == k for r in rows) for k in 'SFKD'}
    counted = cnt['S'] + cnt['F'] + cnt['K']
    succ = [r for r in rows if r['status'] == 'S']
    attacked = [r for r in rows if r['status'] in 'SF']
    return dict(acc=100 * cnt['F'] / max(1, counted), acc_counting_dropped_as_wrong=100 * cnt['F'] / max(1, len(rows)),
                success_rate=100 * cnt['S'] / max(1, cnt['S'] + cnt['F']), orig_acc=100 * (cnt['S'] + cnt['F']) / max(1, counted),
                n=counted, n_requested=len(rows), dropped=cnt['D'],
                avg_queries=float(np.mean([r['queries'] for r in attacked])) if attacked else 0.0,
                avg_pct_changed=float(np.mean([r['pct_changed'] for r in succ])) if succ else 0.0,
                minutes=sum(r['seconds'] for r in rows) / 60)


def adversarial_pairs(lab, victim, victim_name, attack, n, seed):
    """Run `attack` on n TRAIN texts (seeded sample) against `victim`; successful attacks become training pairs.
    Returns (adversarial embeddings, clean embeddings, labels) under the current model. Resumable and cached."""
    cache = lab._path(f'advpairs_{victim_name}_{attack}_n{n}_s{seed}.pt')
    if os.path.exists(cache):
        return torch.load(cache)
    pairs = lab.attack_pairs(n, split='train', seed=seed)
    log = lab._path(f'advpairs_{victim_name}_{attack}_n{n}_s{seed}.jsonl')
    print(f'[pairs] {attack} vs {victim_name} on {n} train texts (seed {seed})', flush=True)
    _run_attack_fork(lab, victim, attack, pairs, log)
    rows = [json.loads(line) for line in open(log)]
    rows = [r for r in rows if r['status'] == 'S']
    emb = lambda texts: torch.cat([lab.pooled(texts[i:i + 128]).cpu() for i in range(0, len(texts), 128)])
    out = (emb([r['adv_text'] for r in rows]), emb([r['text'] for r in rows]), torch.tensor([r['label'] for r in rows]))
    torch.save(out, cache)
    print(f'[pairs] {len(rows)} successful attacks -> training pairs', flush=True)
    return out


def build_attack(name, victim, rho, K=50, sim=0.840845057):
    return (_build_attack_fork if BACKEND == 'fork' else _build_attack_pip)(name, victim, rho, K, sim)


def run_attack(lab, victim, name, pairs, state=None, chunk=50, on_chunk=None, log_path=None):
    if BACKEND == 'fork':
        return _run_attack_fork(lab, victim, name, pairs, log_path or lab._path(f'examples_fork_adhoc_{name}.jsonl'), chunk)
    return _run_attack_pip(lab, victim, name, pairs, state, chunk, on_chunk)


# ----- pip backend (kept for the record) -----
def _wrapper_class():
    from textattack.models.wrappers import ModelWrapper

    class VictimWrapper(ModelWrapper):
        def __init__(self, victim):
            self.victim = victim
            self.model = victim.lab.model

        @torch.no_grad()
        def __call__(self, text_input_list):
            texts, out = list(text_input_list), []
            for i in range(0, len(texts), 64):
                out.append(self.victim.scores(texts[i:i + 64]).float().cpu())
            return torch.cat(out)

    return VictimWrapper


def _pwws_search_class():
    """Port of TextDefender's GreedyWordSwapWIRPWWS (TextDefender/textattack/search_methods/greedy_word_swap_wir_pwws.py).

    The authors' PWWS recipe does not use TextAttack's stock weighted-saliency search: it ranks positions by
    softmax(saliency) * score of the best substitute, then applies each position's best substitute in that order
    (checking the constraints) until the label flips. The stock pip recipe searches differently, and is a
    noticeably different (stronger) attack, so we port the authors' version.
    """
    from textattack.goal_function_results import GoalFunctionResultStatus
    from textattack.search_methods import SearchMethod
    from textattack.shared.validators import transformation_consists_of_word_swaps_and_deletions
    from torch.nn.functional import softmax

    class PWWSSearch(SearchMethod):
        def __init__(self, transformation):
            self.transformation = transformation

        def _get_index_order(self, initial_text):
            len_text = len(initial_text.words)
            leave_one = [initial_text.replace_word_at_index(i, '[UNK]') for i in range(len_text)]
            results, search_over = self.get_goal_results(leave_one)
            saliency = np.array([r.score for r in results])
            soft = softmax(torch.Tensor(saliency), dim=0).numpy()
            index_and_word = []
            for idx in range(len_text):
                item = dict(idx=idx, word=None, saliency_score=0.0)
                cands = self.get_transformations(initial_text, original_text=initial_text, indices_to_modify=[idx])
                if cands:
                    swap_results, _ = self.get_goal_results(cands)
                    best = max(range(len(swap_results)), key=lambda i: swap_results[i].score)
                    item['word'] = cands[best].words[idx]
                    item['saliency_score'] = swap_results[best].score * soft[idx]
                index_and_word.append(item)
            index_and_word.sort(key=lambda x: x['saliency_score'], reverse=True)
            return index_and_word, search_over

        def perform_search(self, initial_result):  # pip textattack names it perform_search; the fork _perform_search
            cur_text = initial_result.attacked_text
            index_and_word, search_over = self._get_index_order(cur_text)
            i, cur_result = 0, initial_result
            while i < len(index_and_word) and not search_over:
                if index_and_word[i]['word'] is None:
                    break
                cand = cur_text.replace_word_at_index(index=index_and_word[i]['idx'],
                                                      new_word=index_and_word[i]['word'])
                i += 1
                cand.attack_attrs['last_transformation'] = self.transformation
                filtered = self.filter_transformations([cand], cur_text, initial_result.attacked_text)
                if len(filtered) == 0:
                    continue
                cur_text = filtered[0]
                result, search_over = self.get_goal_results([cur_text])
                cur_result = result[0]
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    break
            return cur_result

        _perform_search = perform_search

        def check_transformation_compatibility(self, transformation):
            return transformation_consists_of_word_swaps_and_deletions(transformation)

        @property
        def is_black_box(self):
            return True

        def extra_repr_keys(self):
            return []

    return PWWSSearch


def _build_attack_pip(name, victim, rho, K=50, sim=0.840845057):
    """Paper settings (Li et al. 2021 / TextDefender): K candidates, <= rho of words modified,
    USE angular similarity >= 0.8408 (window 15), query budget K * (number of words)."""
    from textattack import Attack
    from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, BERTAttackLi2020
    from textattack.constraints.overlap import MaxWordsPerturbed
    from textattack.constraints.semantics import WordEmbeddingDistance
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.goal_functions import UntargetedClassification
    from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM

    class BudgetGoal(UntargetedClassification):
        def __init__(self, model_wrapper, per_word_budget, **kw):
            super().__init__(model_wrapper, **kw)
            self.per_word_budget = per_word_budget

        def init_attack_example(self, attacked_text, ground_truth_output):
            self.query_budget = self.per_word_budget * attacked_text.num_words
            return super().init_attack_example(attacked_text, ground_truth_output)

    wrapper = _wrapper_class()(victim)
    recipe = {'pwws': PWWSRen2019, 'textfooler': TextFoolerJin2019, 'bertattack': BERTAttackLi2020}[name].build(wrapper)
    if name == 'bertattack':
        transformation = recipe.transformation  # reuse the recipe's masked LM instead of loading a second copy
        transformation.max_candidates = K
        constraints = list(recipe.constraints)
    else:
        transformation = WordSwapEmbedding(max_candidates=K)
        constraints = [c for c in recipe.constraints if not isinstance(c, WordEmbeddingDistance)]
    constraints.append(MaxWordsPerturbed(max_percent=rho))
    constraints.append(UniversalSentenceEncoder(threshold=sim, metric='angular', compare_against_original=False,
                                                window_size=15, skip_text_shorter_than_window=True))
    goal = BudgetGoal(wrapper, K, model_batch_size=64)
    search = _pwws_search_class()(transformation) if name == 'pwws' else recipe.search_method
    return Attack(goal, constraints + list(recipe.pre_transformation_constraints), transformation, search)


def _run_attack_pip(lab, victim, name, pairs, state=None, chunk=50, on_chunk=None):
    """Attack `pairs` in chunks. After each chunk the running counts are passed to on_chunk(state) (the caller
    saves them), and `state` lets a later call resume where an interrupted one stopped. Chunking also bounds
    memory: TextAttack's caches are cleared between chunks."""
    import faulthandler
    import gc
    from textattack import Attacker, AttackArgs
    from textattack.datasets import Dataset
    attack = _build_attack_pip(name, victim, lab.cfg['rho'], K=50)
    global _STACK_FILE  # watchdog: if a chunk takes > 3 min, dump every thread's stack to /content/stack_dump.txt
    _STACK_FILE = open('/content/stack_dump.txt', 'w')
    st = dict(done=0, failed=0, skipped=0, succ=0, queries=0.0, seconds=0.0)
    if state:
        st.update(state)
    while st['done'] < len(pairs):
        part = pairs[st['done']:st['done'] + chunk]
        args = AttackArgs(num_examples=len(part), random_seed=lab.seed, shuffle=False, silent=True,
                          disable_stdout=True, parallel=False)
        t0 = time.time()
        faulthandler.dump_traceback_later(180, repeat=True, file=_STACK_FILE)
        res = Attacker(attack, Dataset(part), args).attack_dataset()
        faulthandler.cancel_dump_traceback_later()
        kinds = [type(r).__name__ for r in res]
        st['failed'] += kinds.count('FailedAttackResult')
        st['skipped'] += kinds.count('SkippedAttackResult')
        st['succ'] += kinds.count('SuccessfulAttackResult')
        st['queries'] += float(sum(r.num_queries for r in res))
        st['seconds'] += time.time() - t0
        st['done'] += len(part)
        try:
            attack.clear_cache()
        except Exception:
            pass
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'    {name}: {st["done"]}/{len(pairs)} done, running acc under attack '
              f'{100 * st["failed"] / st["done"]:.1f}%, {st["seconds"] / 60:.1f} min', flush=True)
        if on_chunk:
            on_chunk(dict(st))
    n = st['done']
    return dict(acc=100 * st['failed'] / n, success_rate=100 * st['succ'] / max(1, n - st['skipped']),
                orig_acc=100 * (n - st['skipped']) / n, avg_queries=st['queries'] / n,
                minutes=st['seconds'] / 60, n=n)


def evaluate(lab, victim, tag, E_te, Y_te, attacks=('pwws', 'textfooler'), n=1000, chunk=50, need_log=False):
    """Clean accuracy on the full test set + accuracy under each attack (add 'bertattack' explicitly: it is the
    slowest). Cached per (dataset, tag, n); the fork backend also logs every example (examples_path), which McNemar
    tests and transfer tests need: need_log=True re-runs an attack whose cached result has no such log."""
    path = results_path(lab, tag, n)
    results = json.load(open(path)) if os.path.exists(path) else {}
    if 'clean' not in results:
        results['clean'] = dict(acc=100 * victim.clean_accuracy(E_te, Y_te), n=len(E_te))
        json.dump(results, open(path, 'w'), indent=1)
    print(f'[{tag}] clean acc {results["clean"]["acc"]:.2f}')
    pairs = lab.attack_pairs(n)
    for name in attacks:
        if name in results and results[name].get('n_requested', results[name].get('n')) == len(pairs):
            if not (need_log and BACKEND == 'fork' and not os.path.exists(examples_path(lab, tag, n, name))):
                continue
        partial = results.get(f'_partial_{name}')
        if partial and partial.get('done', 0) > 0:
            print(f'[{tag}] resuming {name} at example {partial["done"]}', flush=True)
        else:
            partial = None
        print(f'[{tag}] attacking with {name} on {len(pairs)} examples ...', flush=True)

        def save(st, name=name):
            results[f'_partial_{name}'] = st
            json.dump(results, open(path, 'w'), indent=1)

        results[name] = run_attack(lab, victim, name, pairs, state=partial, chunk=chunk, on_chunk=save,
                                   log_path=examples_path(lab, tag, n, name))
        results.pop(f'_partial_{name}', None)
        print('   ', {k: round(v, 2) for k, v in results[name].items()}, flush=True)
        json.dump(results, open(path, 'w'), indent=1)
    return results


def compare(lab, tag, baseline_tag=None, n=1000):
    """Table of our results vs the paper's TMD and vanilla numbers (and vs our own baseline run at the same N_EVAL,
    if present)."""
    import pandas as pd
    res = json.load(open(results_path(lab, tag, n)))
    base = None
    if baseline_tag:
        p = results_path(lab, baseline_tag, n)
        base = json.load(open(p)) if os.path.exists(p) else None
    paper = PAPER[lab.dataset]
    rows = []
    for i, m in enumerate(METRICS):
        if m not in res:
            continue
        ours, n = res[m]['acc'], res[m]['n']
        p = ours / 100
        ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-4) / n) * 100
        paper_n = len(lab.test_ds) if m == 'clean' else 1000
        pt = paper['TMD'][i] / 100
        ci_paper = 1.96 * math.sqrt(pt * (1 - pt) / paper_n) * 100
        diff = ours - paper['TMD'][i]
        tol = math.sqrt(ci ** 2 + ci_paper ** 2)
        verdict = 'BEATS' if diff > tol else ('worse' if diff < -tol else 'tie (within noise)')
        row = {'metric': METRIC_LABEL[m], 'paper vanilla': paper['vanilla'][i], 'paper TMD': paper['TMD'][i],
               'best baseline in paper': paper['best_baseline'][i], tag: round(ours, 1),
               'vs paper TMD': round(diff, 1), '95% noise': round(tol, 1), 'verdict': verdict}
        if base and m in base:
            row[f'our {baseline_tag}'] = round(base[m]['acc'], 1)
            row[f'vs our {baseline_tag}'] = round(ours - base[m]['acc'], 1)
        rows.append(row)
    return pd.DataFrame(rows)
