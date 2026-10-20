> **Starting from nothing? Follow [START_HERE.md](START_HERE.md) and use the ordered notebooks in [RUN_ON_KAGGLE/](RUN_ON_KAGGLE/).** The runbook below documents the underlying reference notebooks.

# Knee MRI pipeline — audited 2026-09-12

The workspace contains a Kaggle MRI pipeline, not a trained model deployment. The prior notes report **0.8766 gold AUC / 0.874 public leaderboard**. No model weights, competition scans, cache arrays, or label CSVs are present locally, so those scores have not been independently reproduced. Current leaderboard standings could not be retrieved from Kaggle's dynamically rendered leaderboard.

The updated pipeline implements a stronger experimental foundation, with DINOv3-small available as a pretrained encoder. **No new competitive score or state-of-the-art result is claimed.** Changes require a new cache and real Kaggle training.

## What changed

| Area | Before | Now |
|---|---|---|
| Validation | Best epoch chosen by the 58 human-labeled studies | Five pseudo-label folds; soft-label validation BCE selects epochs; gold only audits the selected model |
| Splits | One random 8% pseudo holdout | Multilabel stratification when groups are unique; patient-group folds when patient IDs exist; gold patients excluded from pseudo training |
| Geometry | Coordinate side fallback used the image corner; one spacing for both crop axes | Actual image-center calculation, explicit side tags take priority, separate row/column spacings |
| Decode failures | Failed slices stayed zero and contaminated normalization | Nearest decoded slice replaces missing slices; all-failed series stays absent |
| Cache recovery | Completion persisted separately from slot presence | Atomic combined progress, flushed pixels, recipe and manifest fingerprint checks |
| Train/test contract | New crop/side/slot recipe absent from submission | Same preprocessing source embedded in both notebooks; versioned recipe checked at inference |
| Model | EfficientNet with hardcoded grayscale normalization | EfficientNet baseline or DINOv3-small; pretrained per-channel normalization; common architecture source |
| GPU memory | Full inference image batch | Chunked encoder inference; DINO microbatch 1 with four-step gradient accumulation |
| Reproducibility | Single best checkpoint | Per-fold best and resume checkpoints, histories, fold assignments, dataset identity, package versions, OOF exports |
| Submission | Single checkpoint; model ran on all-zero studies | Decode once to scratch storage; sequential fold inference; rank-space ensemble; explicit blank fallback and tie diagnostics |
| Recipe (2026-10-07) | b0, three fluid slots, 12 neighbour triplets per slot, one label table | Two self-trained families: a 2.5D ConvNeXt-T study reader (adjacent-slice windows, slot/position tokens, transformer, per-finding attention, EMA) and a DINOv2-S slot model; six header-typed slots, 8 windows × 3 slices at 240 px in a two-shard cache (`knee-2c-v3`); soft targets averaged over public and own report-label readers; report-grouped folds; gold trained at weight 3. See `METHOD_NOTES.md` |

The original notebooks and notes are retained in `backups/original/`. Label generation, the original three-plane cache, and the historical six-slot cache were not changed. The label-generation notebook exists and is substantial; the previous note calling it a zero-byte `.xpynb` was stale.

## Run on Kaggle

1. Run `knee_mri_step2c_cache.ipynb` with the competition attached and Internet enabled for decoder installation. Inspect the left/right preview and `side_src` distribution. Rebuild in an empty working directory: old cache pixels are incompatible with this recipe. The default 3 × 12 × 336 × 336 cache is about 17.9 GB for 4,407 studies.
2. Attach that output and the label CSV to `training.ipynb`. Set `CACHE_DIR` and `LABELS` to the actual mounted paths. Prefer the consensus `train_targets.csv` when available; NaNs remain masked and fractional targets remain fractional. The training notebook installs the small split helper if missing. Keep Kaggle's CUDA-compatible torch/torchvision pair; use a timm build supporting your selected model (local test version is in `requirements-tested.txt`).
3. Start with `FOLDS_TO_RUN = [0]` to measure actual T4 memory and epoch time at 336px. Default EfficientNet uses batch 4. Set `BACKBONE = "vit_small_patch16_dinov3.lvd1689m"` for the stronger-pretraining comparison; its default microbatch is 1, with accumulation 4. Pretrained weights require Internet on the training run. No GPU speed or memory figure has yet been measured for this revised recipe.
4. Complete folds 0–4 in the same output directory. On a fresh Kaggle session, copy earlier fold outputs into `OUT` first. To resume interrupted training, copy both `fold_N.pt` and `last_fold_N.pt` and set `RESUME = True`. Resume continues at the next epoch; in-progress epoch work is replayed. Use separate directories for different backbones or recipes.
5. Inspect per-fold histories, `oof_predictions.csv`, and `gold_ensemble.csv`. Check missing/one-class findings and patient grouping before interpreting AUC. Pseudo OOF is a noisy-label proxy and epoch selection makes it optimistic. The historical gold set has already influenced previous experiments, so it is not a pristine final test set. Larger pseudo validation does not remove label bias.
6. Attach only the intended `fold_0.pt` … `fold_4.pt` output set to `submission-notebook.ipynb`; point `CKPT_DIR` at that directory. With no fold checkpoints it falls back to the historical `best.pt` path. New fold recipes must match exactly. All new inference runs use `pretrained=False` and require the matching timm architecture and decoder packages already available offline.
7. Run an offline Kaggle test before submitting. Inspect `inference_diagnostics.json`: completed models, blank studies, tie sizes, and elapsed minutes. Scratch space must cover `N_test × 3 × 12 × 336²` bytes plus 1 GB; the scratch cache is removed after writing the CSV. Decoder/runtime failures can still degrade ranking. Invalid recipes or insufficient disk fail preflight visibly.

The original three-plane `best.pt` remains supported with its legacy preprocessing and normalization. Historical six-slot checkpoints intentionally fail an explicit compatibility check: their header selection and slice-band recipe is different. Use their exact old inference implementation or retrain on the new recipe; never relabel six-slot names as anatomical planes.

## What remains empirical

- Compare the corrected EfficientNet and DINOv3 runs on identical folds, then measure Kaggle inference time and leaderboard AUC. Improvements are hypotheses until measured.
- Finish the already resumable label ensemble when your Kaggle quota permits. No external API calls or quota-consuming training runs were launched here. Changing labels requires a new training output directory.
- Coordinate-derived knee side is still a heuristic: patient-coordinate origin alone does not establish anatomical laterality in every scan. Inspect these studies; tags are stronger evidence. The orientation code canonicalizes the established column/stack axes, not arbitrary oblique 3D volumes through full resampling.
- Anisotropic cropping respects both spacings but clips to the acquired image; scans with a smaller field of view do not magically gain 130 mm coverage. Missing spacing falls back to the full image.
- Spatial/slice-window TTA, six-slot model diversity, external public weights, and additional self-supervised models remain separate experiments. AUC is not improved merely by adding them. Rank ensembling across models is a design choice to validate, not a mathematical guarantee.
- This is competition research code, with no clinical deployment validation.

## Local verification and maintenance

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_notebooks.py
```

Edit `preprocessing.py`, `model_core.py`, `label_builder.py`, `validation.py`, `cache_runtime.py`, `train_config.py`, `train_data.py`, `dataset_runtime.py`, `train_runtime.py`, and `inference_runtime.py`, then rebuild the notebooks and launchers. The generator embeds the sources, so Kaggle notebooks need no companion Python files. Change the preprocessing version when changing image semantics and rebuild the cache. Review generated configuration too: the current cache recipe is fixed at 240px / 8 windows × 3 slices / 6 slots, two shards (`knee-2c-v3`).

Tests cover synthetic geometry and corruption, empty studies, cache resume and recipe mismatch, gold/group isolation, masked gradients, rank transforms, historical model compatibility, both real backbone architectures with random weights, and a small synthetic fold-training/resume/submission run. They do not establish real MRI accuracy, pretrained weight availability, compressed-DICOM backend coverage, CUDA throughput, or Kaggle platform compatibility. The local package snapshot is for reproducibility on macOS, not a command to replace Kaggle's GPU stack.

## Primary references checked

- [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html): image position, orientation, row/column spacing, and patient axes underpin the geometry correction.
- [timm DINOv3-small model card](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m): model identifier, pretrained feature extraction, model-specific normalization, and license link. Model-card benchmarks do not establish knee-MRI performance.
- [RSNA competition announcement](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri): competition context.
- [Competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules): external model/data access conditions. Recheck the current rules and each model license for any additional external ensemble.
