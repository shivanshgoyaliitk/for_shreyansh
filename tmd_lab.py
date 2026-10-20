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
import time

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

# Table 6 (BERT column), max_len from Appendix A.1, modify ratio from Sec. 4.1
CFG = {
    'ag_news': dict(max_len=128, n_cls=4, K=50, d=20, g_lr=1e-4, d_lr=1e-4, p_lr=0.0, dg=1, rho=0.3),
    'imdb': dict(max_len=256, n_cls=2, K=50, d=40, g_lr=1e-4, d_lr=1e-4, p_lr=0.0, dg=1, rho=0.1),
    'yelp_polarity': dict(max_len=256, n_cls=2, K=100, d=20, g_lr=1e-4, d_lr=9e-5, p_lr=0.0, dg=1, rho=0.1),
}

# Recent `datasets` / `huggingface_hub` versions require namespaced repo ids
HF_ID = {'ag_news': 'fancyzhx/ag_news', 'imdb': 'stanfordnlp/imdb', 'yelp_polarity': 'fancyzhx/yelp_polarity'}

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_env():
    """Keep TensorFlow (used only by TextAttack's sentence encoder) off the GPU."""
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
    torch.backends.cudnn.benchmark = True


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
    gan = InfoGAN(cfg['d'], cfg['K']).to(DEV)
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
        elif mode == 'encoder':
            z = enc(x, c)[0][:, None]
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

    def finetune(self, tag='base', epochs=3, lr=3e-5, bs=32, augment_fn=None, aug_prob=0.5):
        """Fine-tune (or load) the classifier. augment_fn(text)->text is applied to a fraction aug_prob of examples."""
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        path = self._path(f'finetuned_{tag}')
        if os.path.exists(os.path.join(path, 'config.json')):
            print(f'[{tag}] loading fine-tuned model')
            return self.use_model(tag)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.cfg['n_cls']).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        steps = epochs * math.ceil(len(self.train_ds) / bs)
        sched = get_linear_schedule_with_warmup(opt, 0, steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.amp == torch.float16))
        best = 0.0
        for ep in range(epochs):
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
            if va > best:
                best = va
                model.save_pretrained(path)
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
        path = self._path(f'infogan_{tag}.pt')
        gan = InfoGAN(self.cfg['d'], self.cfg['K']).to(DEV)
        if os.path.exists(path):
            gan.load_state_dict(torch.load(path))
            print(f'[{tag}] loaded InfoGAN')
            return gan.eval()
        gan = train_gan(E_tr, E_va, self.cfg, epochs=epochs, bs=bs, seed=self.seed)
        torch.save(gan.state_dict(), path)
        return gan

    def get_encoder(self, name, gan, E_tr, E_va, force=False, **kw):
        path = self._path(f'encoder_{name}.pt')
        if os.path.exists(path) and not force:
            enc = LatentEncoder(gan.latent_dim, gan.code_dim).to(DEV)
            enc.load_state_dict(torch.load(path))
            print(f'[{name}] loaded encoder')
            return enc.eval()
        enc = train_encoder(gan, E_tr, E_va, **kw)
        torch.save(enc.state_dict(), path)
        return enc

    # ----- attack protocol
    def attack_pairs(self, n):
        """TextDefender: keep test texts with more than 4 words, sample n without replacement (seeded)."""
        texts, labels = self.test_ds['text'], self.test_ds['label']
        idx = [i for i, t in enumerate(texts) if len(t.split(' ')) > 4]
        sel = np.random.RandomState(self.seed).choice(idx, size=min(n, len(idx)), replace=False)
        return [(texts[i], int(labels[i])) for i in sel]


# --------------------------------------------------------------------------
# Attacks with the paper's constraints (TextDefender build_english_attacker)
# --------------------------------------------------------------------------
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


def build_attack(name, victim, rho, K=50, sim=0.840845057):
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
        transformation = WordSwapMaskedLM(method='bert-attack', max_candidates=K)
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


def run_attack(lab, victim, name, pairs):
    from textattack import Attacker, AttackArgs
    from textattack.datasets import Dataset
    attack = build_attack(name, victim, lab.cfg['rho'], K=50)
    args = AttackArgs(num_examples=len(pairs), random_seed=lab.seed, shuffle=False, silent=True,
                      disable_stdout=True, parallel=False)
    t0 = time.time()
    res = Attacker(attack, Dataset(pairs), args).attack_dataset()
    kinds = [type(r).__name__ for r in res]
    n = len(kinds)
    failed, skipped, succ = kinds.count('FailedAttackResult'), kinds.count('SkippedAttackResult'), kinds.count('SuccessfulAttackResult')
    return dict(acc=100 * failed / n, success_rate=100 * succ / max(1, n - skipped), orig_acc=100 * (n - skipped) / n,
                avg_queries=float(np.mean([r.num_queries for r in res])), minutes=(time.time() - t0) / 60, n=n)


def evaluate(lab, victim, tag, E_te, Y_te, attacks=('pwws', 'textfooler', 'bertattack'), n=1000):
    """Clean accuracy on the full test set + accuracy under each attack. Cached per (dataset, tag)."""
    path = lab._path(f'results_{tag}.json')
    results = json.load(open(path)) if os.path.exists(path) else {}
    if 'clean' not in results:
        results['clean'] = dict(acc=100 * victim.clean_accuracy(E_te, Y_te), n=len(E_te))
        json.dump(results, open(path, 'w'), indent=1)
    print(f'[{tag}] clean acc {results["clean"]["acc"]:.2f}')
    pairs = lab.attack_pairs(n)
    for name in attacks:
        if name in results and results[name].get('n') == len(pairs):
            continue
        print(f'[{tag}] attacking with {name} on {len(pairs)} examples ...', flush=True)
        results[name] = run_attack(lab, victim, name, pairs)
        print('   ', {k: round(v, 2) for k, v in results[name].items()}, flush=True)
        json.dump(results, open(path, 'w'), indent=1)
    return results


def compare(lab, tag, baseline_tag=None):
    """Table of our results vs the paper's TMD and vanilla numbers (and vs our own baseline run, if present)."""
    import pandas as pd
    res = json.load(open(lab._path(f'results_{tag}.json')))
    base = None
    if baseline_tag:
        p = lab._path(f'results_{baseline_tag}.json')
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
