# RSNA Knee Abnormality Detection

Kaggle competition, closes **2026-10-22**. Predict 12 binary findings per knee MRI
study. Metric: **macro ROC-AUC** over the 12 columns.

**Best model: 3-slot EfficientNet-B0, 8 epochs — gold AUC 0.8766, public LB 0.874.**
Local gold tracks the LB closely (0.877 vs 0.874).

## Where the frontier actually is (verified 2026-09-11)

| | public macro AUC |
|---|---|
| our submitted model | 0.874 (rank ~2160 of ~3390) |
| public baseline notebook, heavily forked | ~0.891 |
| dozens of public blend notebooks | 0.936 – 0.941 |
| **top of leaderboard** | **0.954** |

We are **8 AUC points off the frontier and below the most-forked public baseline.**
An earlier version of this file framed "+0.005 is worth several hundred places" —
true, and misleading. That density is a crowded band far below the frontier, and
optimising against it is optimising the wrong target. Incremental work on our own
pipeline cannot close this gap; the public state of the art does several structural
things we do not (see below).

Rules §2.6 permits external data and models that are public and free to all
entrants, so public weights, the public LLM-label dataset and the public 384px
corpus are all usable.

**State as of 2026-09-10**
- 3-slot cache + b0: **0.8766 gold / 0.874 LB.** Current best; the submitted model.
- 6-slot cache + b0: **0.8652 gold / 0.8572 pseudo.** A null (see below). Built and
  kept, but not the model to submit.
- Labels: multi-vendor ensemble **paused** — daily AI quota exhausted. ~1,300 of
  4,349 banked for member 1; the JSONL cache resumes, nothing is lost.
- Submission notebook: four edits written for 6-slot compatibility, **not yet
  applied**. Irrelevant while the 3-slot model is the one being submitted.

## Read this before proposing an experiment: what the validation can and cannot see

Minimum detectable macro-AUC gain (95%, assuming rho=0.5 between two runs):

| validation set | smallest gain visible |
|---|---|
| gold, n=58 | **0.034** |
| pseudo, n=347 | **0.014** |

A **+0.005** gain is worth several hundred LB places and **neither set can see it.**
Every local "no difference" result therefore means *no large effect*, never *no
effect*. Do not read a null here as proof that an idea is worthless — and do not
run experiments whose expected effect is below these thresholds hoping to measure
them. The way out is a bigger validation set (folds over 4,349 studies), not more
careful reading of 58.

## The defining constraint

**Only 58 of 4,407 training studies carry human labels.** The other 4,349 have a
free-text radiology report in ~12 languages. Labels must be derived from those
reports with an LLM. That is the competition, not an optimisation.

Consequences that shape everything:

- 58 studies gives ~±0.10 per-finding and ±0.03 macro resolution. **Never tune
  against it.** A change of <0.1 on one finding is noise; one flipped study moves
  F1 by 0.04–0.08.
- The gold 58 are positive-enriched (Fracture 31% vs ~7% in the wild). Fine for
  AUC, misleading for prevalence.
- Gold is **held out, never trained on**. It is the only human-verified signal.

## Pipeline

| file | stage | notes |
|---|---|---|
| *(on Kaggle)* | 1 · labels | LLM extraction from reports → `pseudo_labels.csv`. The local `compi-label-generation.xpynb` is **0 bytes and misnamed** — re-download from Kaggle. |
| `model.ipynb` | 2 · cache | 3-slot cache, 16 slices, 224px, 10.6 GB. **Use this one.** |
| `knee_mri_step2b_cache6.ipynb` | 2b · cache | 6 slots, header-derived sequence typing, 12 slices, 15.9 GB. Built; a null as a single model. Keep for ensemble diversity. |
| `training.ipynb` | 3 · train | Reads cache + labels → `best.pt`. One line switches slot count (`PLANES` from `slots.json`). |
| `submission-notebook.ipynb` | 4 · submit | Decodes test DICOMs live, writes `submission.csv`. Currently 3-slot, matching the best model. **Needs 4 edits before any 6-slot submission — see below.** |
| `high_scoring_someone_elses.ipynb` | reference | A public 0.940. Inference-only blend of ~30 checkpoints from 8 users; trains nothing. |

## Settled experiment: six slots is a null (2026-09-10)

Six slots = one series per plane for fluid-sensitive+fatsat, plus sagittal
non-fatsat, plus coronal and sagittal T1. The old cache kept three and discarded
the T1 entirely.

| | 3-slot | 6-slot | delta | 95% CI |
|---|---|---|---|---|
| macro gold (n=58) | 0.8766 | 0.8652 | −0.0114 | ±0.045 |
| macro pseudo (n=347) | 0.8592 | 0.8572 | −0.0020 | tighter |

6 findings improved, 6 worsened; no per-finding delta clears its error bar.

**The experiment was fair — this is not a coverage artifact:**

- mean **4.45 of 6** slots present per study; 86.4% of studies had ≥4
- the model received **+48% more imaging per study** than the 3-slot run
- attention on the three new slots, **conditioned on presence, was 28.4%**
  (the raw 18.9% understates it; masked slots contribute exactly 0, so
  `E[a | present] = mean / coverage`)
- the allocation is clinically sensible and nobody encoded it: structural slots
  take **52.7% / 56.1% / 44.2%** for Medial OA / Lateral OA / PF OA, versus 7.7%
  for Fracture. T1 shows bone and cartilage; the model found that itself.

**Conclusion: T1 and non-fat-sat sequences are largely redundant with the
fluid-sensitive ones for these twelve findings, in a single model.**

**Training also overfits from epoch 5** — loss falls 0.34 → 0.27 across epochs 5–7
while both validation curves stay flat (gold 0.8652 → 0.8607 → 0.8647). More epochs
hurt. Early-stop at 6.

**What this rules out**
- More *input* is not the lever. +48% imaging bought nothing.
- More *capacity* is not the lever either — the model already overfits at epoch 5,
  so b3/convnext would overfit sooner. (Different *pretraining*, e.g. RadImageNet,
  is a separate question and still open.)
- The clinical attention prior is **not** worth doing. Its premise was that learned
  attention wasn't finding the anatomy. Conditioned on presence, it substantially is.

**What it does NOT rule out** — see the power table above. A +0.005 gain is invisible
here. And note the 0.940 reference notebook also uses six slots, though it contains
no ablation and is a 30-checkpoint blend, so nothing there attributes its score to
the slot scheme. Six slots may still pay via **ensemble diversity** (a 6-slot model
attends to T1 that a 3-slot model cannot see at all), or with cleaner labels, or with
much stronger backbones. Keep the cache; test it as an ensemble member by rank-
correlating its predictions against a 3-slot model — ~0.95 means real diversity,
0.99 means one vote counted twice.

## The real ceiling

Overfitting at epoch 5 on 4,002 studies, plus a null from +48% more input, both point
the same way: **the binding constraint is label noise (~25% disagreement) on a 4k
training set**, not architecture, input volume or capacity. Levers that attack that:
folds + ensembling (averages out noise-fitting), better labels, stronger
regularisation. Levers that don't: bigger models, more epochs, more slices.

## BLOCKING for any 6-slot submission: the submission notebook does not match

Switching training to six slots is **one line** (`PLANES` from `slots.json`; every
other reference goes through `len(PLANES)` and adapts). The submission notebook is
not so lucky, and it fails **silently**:

- It reads `PLANES = cfg["planes"]`, which is now `["SAG_FLUID_FS", ...]`, then
  matches `g["Anatomical_Plane"] == plane`. That column only holds
  `Sagittal`/`Coronal`/`Axial`, so **nothing matches** — every test study gets an
  all-zero tensor, predictions go near-constant, and every assertion still passes.
  A valid-looking submission scoring ~0.5.
- Even with slot names fixed, it selects on the provided `Fluid_Sensitive` flag and
  has no header scan, while the cache was built with header-derived typing. Train
  and test would sort series by different rules.

**Rule: test preprocessing must reproduce the cache-build logic exactly.** Port
`annotate()` and the six-slot `choose()` from `knee_mri_step2b_cache6.ipynb` before
any six-slot submission.

The four edits are written and syntax-checked but **not applied** (the current best
model is 3-slot, so they aren't needed yet): add `re` to the cell-1 imports; append
the `SLOTS` spec plus an `assert [s[0] for s in SLOTS] == list(PLANES)` — which turns
the silent failure loud; add `SLICE_BAND` to the `load_series` index calculation; and
replace cell 6 with the header scan + `annotate()` + six-slot `choose()`.

## Hard-won facts — each of these cost a wasted run

**`nn.DataParallel` is 13× slower than one GPU here.** 8.72 s/step vs 0.66 on the
same batch (28 img/s vs 364). It re-replicates the module and broadcasts every
BatchNorm buffer over PCIe each forward, and Kaggle's two T4s have no NVLink. Use
one GPU. If both are ever needed, DDP via `torchrun` — never DataParallel.

**`autocast` is thread-local.** Entering it in the training loop does not reach a
threaded forward. Enter it *inside* `forward`.

**One T4 saturates at batch 4.** Batch 8 gives 364 img/s vs 369, and 11.2 GB vs 5.6.

**protobuf needs both bounds: `>=5.29.6,<6`.** Lower clears the gencode mismatch;
upper is required by kaggle-benchmarks itself plus grpcio-tools, ydf and the
google-cloud stack. Unbounded resolves to 7.x and breaks everything. Needs a
session restart.

**Use `kaggle benchmarks init`, not `auth`.** `auth` writes only the proxy URL/key;
`init` also writes `LLM_DEFAULT` and `LLMS_AVAILABLE`. Without `LLM_DEFAULT` the
import of `kaggle_benchmarks` raises `KeyError`, and `kbench.llms` is `{}`.

**Proxy tokens expire in ~1 hour.** Auto-refresh must **fail closed**: the CLI
writes `expiry_time.isoformat() + "Z"`, which for an aware datetime yields
`...+00:00Z` and breaks `fromisoformat`. Treating an unparseable expiry as
"refresh now" caused an auth call per report and got the endpoint rate-limited.
Unknown expiry → fall back to a timer; always enforce a hard minimum interval.

**LLM pricing is flat per token across vendors.** claude-sonnet-5 $0.000586/report,
gemini-3-flash $0.000586, flash-lite $0.000586 — identical to within tokenizer
drift. Model choice is free; pick for quality and *speed*. Wall-clock varies 17×
(flash-lite 0.27 h per 4,349 reports, sonnet-5 1.96 h, deepseek-r1 4.6 h — and r1
fails structured-output parsing).

**Save & Run All starts with an empty `/kaggle/working`.** Every resumable cache is
lost. Attach the previous output and seed from it, or re-pay.

**Housekeeping must never be able to drop real work.** The refresh check ran
*before* each extraction call and was not wrapped, so an exception in it discarded
the report. Any pre-flight/side-channel check inside a work loop belongs in a
`try/except` that warns once and continues.

**Beware notebook-global shadowing.** `import datetime as dt` was silently
clobbered by `dt = time.time() - t0` in a later cell, and the failure only surfaced
minutes in, when a 5-minute circuit breaker first let the code path run. Use
underscore-prefixed aliases (`_dtmod`) for anything a notebook cell might reuse.

**Backbone throughput on one T4** at 6 slots x 10 slices x batch 4 (240 img/step,
1000 steps/epoch). EfficientNets are memory-bound, so wall-clock does not track
FLOPs — ResNet-50 is 10x b0's FLOPs but only 1.4x the time:

| backbone | img/s | min/epoch | 8 epochs |
|---|---|---|---|
| efficientnet_b0 | 365 | 11 | 1.5 h |
| efficientnet_b3 | 180 | 22 | 3.0 h |
| resnet50 (RadImageNet) | 250 | 16 | 2.1 h |
| convnext_small | 70 | 57 | 7.6 h — not affordable |

~25 h GPU/week, and a 5-fold run is 5x the above.

## Domain rules that are not negotiable

**Two rules about mirroring, and they point opposite ways. Do not merge them.**

1. **Laterality normalisation is REQUIRED, at cache-build time.** A right knee is
   the mirror of a left knee, so without it "medial" sits on opposite sides of the
   image depending on side and the model must infer the side then apply a mirrored
   rule. Five of twelve labels depend on that axis. +x is the patient's left:
   mirror columns for coronal/axial right knees, reverse the *file order* for
   sagittal (reversing the finished volume instead shifts the sampled band by a
   slice). Also canonicalise scanner handedness first via the sign of
   `ImageOrientationPatient[0]`. Implemented and unit-tested in step 2c.
2. **Random flip augmentation stays BANNED.** A random mirror during training undoes
   the normalisation and scrambles medial/lateral. Rotation, scale, translate and
   intensity only.

An earlier version of this file stated only "no flips", which would have blocked
rule 1. That was a real error — it is the largest single gap we had against the
public frontier.

**Resolution matters more than field of view.** Resizing a whole ~150 mm FOV to
224 px is ~0.67 mm/px, coarser than Nyquist for a 1 mm meniscal tear, and spends
most of the pixel budget on air. Crop a fixed *physical* size from `PixelSpacing`
(130 mm) and resize that: 130 mm / 336 px = 0.387 mm/px.

**Canonicalise slice direction.** The stack normal's sign depends on how the scanner
wrote `ImageOrientationPatient`; force its dominant component positive or identical
anatomy arrives reversed between studies.

**Percentiles over the whole volume,** not `out[out > 0]` — otherwise a negative
`RescaleIntercept` silently changes the normalisation.

**Anatomy per finding:** sagittal carries ACL and menisci; coronal the collaterals
and tibiofemoral compartments; axial the patellofemoral joint, effusion and
synovitis. A single shared attention over planes collapsed axial to 0.001 — use
per-label attention (`LabelAttnPool`).

## Conventions

- `NaN` in a label means *not observed*; masked per-cell via `masked_bce`, never
  filled with 0.
- Geometry is read from the cache (`cache.shape`, `slots.json`), never from
  constants, so a rebuilt cache can't silently corrupt training.
- Fuse in **rank space** (`rank(pct=True)`), not probability space — AUC only sees
  ordering.
- Submission notebooks must **never crash**. Every study is wrapped; failures fall
  back to the mean prediction; `submission.csv` is always written.
- Ties are the least visible AUC loss. Studies that fail to decode all receive the
  same constant — check the largest tie block per finding.

## Next

**Change one thing at a time and measure against 0.8766 gold / 0.874 LB.** The
leaderboard is too dense, and the gold set too small, to attribute a gain to one of
several stacked changes. And check the power table first: if an idea's expected
effect is under ~0.014, this setup cannot measure it — so either make the
validation bigger or accept the change on principle rather than on evidence.

1. **5-fold CV on the 3-slot cache + rank-space ensembling.** The largest reliable
   gain left, ~7.5 h. It attacks label noise by averaging, and — more importantly —
   replaces the 58-study validation with a 4,349-study one, which ends the
   guesswork the ±0.045 error bars keep forcing. Do this first; it makes every
   later experiment measurable.
2. **Finish the multi-vendor label ensemble** when quota resets → consensus targets
   in {0, ⅓, ⅔, 1}. Attacks the noise ceiling at its source. Cost is flat across
   vendors (~$2.55/pass); pick for quality and speed.
3. **Stronger regularisation + early stop at 6 epochs.** Free, and directly targets
   the epoch-5 overfitting.
4. **RadImageNet ResNet-50.** Better *features*, not more capacity — the one
   backbone idea the overfitting evidence doesn't rule out. ~2.1 h at 3 slots.
5. **TTA at submission: slice-window offsets and multi-scale, never flips.**
   Inference is decode-bound, so extra forwards on an already-decoded volume are
   nearly free.
6. **Ensemble a 6-slot member** once folds exist, gated on the rank-correlation
   check (see the six-slot section).
7. **Richer report extraction**: ~30 auxiliary findings as extra heads dropped at
   inference, plus a normalised English caption embedding. The 12 labels throw away
   most of each report. Highest ceiling, highest variance.

**Do not** spend time on: more epochs, more slices, or the clinical attention prior.
Each is measured-null or contradicted by the epoch-5 overfitting.

**Retracted (2026-09-11):** an earlier version of this file also said "do not use
bigger backbones". That generalised an *input-volume* null plus an overfitting curve
into a ban on *capacity*, which the power table says we cannot test. The frontier
uses DINOv2/v3-small, CoAtNet and RadImageNet — self-supervised and domain
pretraining are different objects from ImageNet-supervised b0. Backbones are open.

**Also note:** the six-slot null was measured *without* laterality normalisation and
*without* the physical crop. It says nothing about six slots under the corrected
recipe, and should be re-tested once 2c is in.
