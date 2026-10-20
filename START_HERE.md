# Start here — run everything on Kaggle

**Quick path:** open `KAGGLE_RUN/README.md`. That folder holds only the files to upload or run, grouped as `step1` … `step5`, with a checklist. The details below explain each step.

Use the notebooks in **KAGGLE_RUN/** (the same files as **RUN_ON_KAGGLE/**, grouped in run order). They are self-contained: upload each `.ipynb` into a separate Kaggle notebook; no local Python files need uploading. Run on Kaggle, not on your Mac.

Accept the competition rules and attach **RSNA Knee Abnormality Detection** through **Add Input** to every notebook. Import the indicated file, choose the settings below, then **Save Version → Save & Run All**. Wait for that saved version to finish successfully before attaching its output downstream. Pin the same Kaggle environment for every training and submission run: checkpoints record package versions.

| Step | File | Accelerator | Internet | Additional inputs |
|---|---|---|---|---|
| 1 | Upload `reader_labels/` as a private dataset | — | — | — |
| 1b | Upload `knee_wheelhouse/` (5 wheels, 10 MB) as private dataset `knee-wheelhouse` | — | — | — |
| 2 | `02_cache_shard0.ipynb` and `02_cache_shard1.ipynb` | None / CPU | ON | None (run both, in parallel) |
| 2b | Save both cache outputs as private datasets (`knee-cache-v3-shard0/1`) | — | — | — |
| 3 | `03_gpu_preflight.ipynb` | GPU T4 | ON | None |
| 4a | `04_input_check.ipynb` | None / CPU | ON | Session C's inputs (below) |
| 4 | `04_session_A_reader_f0_f1`, `04_session_B_reader_f2_f3`, `04_session_C_reader_f4_slot_rad` | GPU **T4 x2** | ON | Both cache datasets + label datasets (below); session C also the RadImageNet ResNet50.pt |
| 5 | `05_submit.ipynb` | GPU T4 | **OFF** | `knee-wheelhouse` dataset (offline packages) + the three session outputs |

Three self-trained families: the ConvNeXt 2.5D reader (primary), the DINOv2 slot model, and RadImageNet heads. `optional/single_runs/` keeps one-fold-per-notebook versions of every job for reruns.

Attach the saved output of your earlier `01_labels` Kaggle run (`compi-label-generation`) to every training session. Training reads its Gemini 3.1 Flash-Lite member file (`labels/pseudo_google_gemini_3_1_flash_lite_preview.jsonl`) and ignores its consensus `train_targets.csv`. There's no need to rerun `01_labels`.

## 1. Label datasets

Training averages every report-label table it finds under `/kaggle/input` (`label_builder.py`). Attach to every training session:

- the four public tables: **pilkwang/rsna-knee-llm-labels**, **stevenleehans/rsna-knee-llm-report-labels**, **lixin73/rsna-knee-llm-report-labels-sol56**, **laymond/rsna-knee-abnormality-qwen3-8b-weak-labels**;
- your own readers: run `python scripts/export_reader_labels.py` locally, upload the resulting `reader_labels/` folder (GPT Sol and Sonnet tables) as one private dataset, and attach it;
- the saved output of your `01_labels` run, for the Flash-Lite labels.

The log prints every reader it used, its coverage, and its gold AUC as a diagnostic. It does not choose readers by gold. Attach each table once; duplicates stop the run, and a missing required reader stops it before training. The merged targets are saved as `targets_used.csv`.

## 2. Build the image cache (two shards)

Upload both `02_cache_shard0.ipynb` and `02_cache_shard1.ipynb`; they may run at the same time. Each writes **cache.npy**, **cache_index.csv**, **cache_meta.json** and progress files; shard 0 also writes **wheelhouse/** for the offline submission. Each shard is about 18 GB.

The recipe must print `knee-2c-v3`. It covers 6 header-typed slots × 8 windows of 3 adjacent slices × 240 px over a 130 mm crop, and each shard holds every second study. Check the left/right preview and `Studies with each slot`. The three fat-suppressed fluid slots should cover nearly every study; the non-suppressed sagittal and two T1 slots are sparser. Absent slots are masked, never filled with another sequence.

Each shard notebook writes its cache as 18 files of about 1 GB (`cache_piece_00/` …), each with its own index and metadata; training discovers them like shards. On 2026-10-07 Kaggle could not attach the earlier single 17 GiB cache file as an input, on GPU or CPU machines (`ERRORED_MOUNTING_DATASET`). Then open each shard notebook's **Output → ⋮ → New dataset** and save it as a private dataset (`knee-cache-v3-shard0`, `knee-cache-v3-shard1`; use New dataset version if the output is already linked); their latest version must hold the `cache_piece_*` folders. Attach these datasets from here on. Every session first copies the attached data (~34 GiB) to local disk, about 15 minutes before any code runs.

## 3. Check the GPU

Upload `03_gpu_preflight.ipynb`. It times training and evaluation steps of the ConvNeXt reader and the DINOv2 slot model at the real cache geometry. It writes **gpu_preflight.json** with peak memory and projected GPU minutes per fold (an upper bound, with every slot present). Every entry must have no `error`; if a model runs out of memory, lower `BATCH` and raise `ACCUM_STEPS` in `train_config.py`.

## 4. Train: three two-GPU sessions

First run `04_input_check.ipynb` on CPU (no GPU quota) with session C's inputs attached. It runs the jobs' own data setup, times random cache reads and verifies the RadImageNet weights; its last line must say `INPUT CHECK PASSED`.

Kaggle bills GPU quota per session hour, and its T4 sessions have two GPUs. Each session (`session_runner.py`) therefore runs two jobs side by side, one process per GPU, each with the exact single-GPU recipe:

- **A:** ConvNeXt reader folds 0 and 1;
- **B:** ConvNeXt reader folds 2 and 3;
- **C:** reader fold 4 on GPU 0; DINOv2 slot folds 0–4, then the RadImageNet heads, on GPU 1.

Each session takes roughly 2.5–3 h. It prints the label readers, the recipe and each epoch's losses every two minutes, writes one folder per family, and fails loudly if any job fails.

**2.5D reader (`reader_convnext`).** ConvNeXt-T on windows of three adjacent slices; slot and stack-position tokens, two transformer layers and per-finding attention pooling. 14 epochs at 4×2 studies per step, encoder learning rate 1e-4 and head 5e-4, one warmup epoch then cosine decay, EMA weights at 0.998. Pretrained ImageNet weights download from Hugging Face, so Internet must be ON. Folds group duplicate reports together. The 58 gold studies train in every fold at weight 3 (`GOLD_WEIGHT`) and never validate; their printed audit is therefore marked in-sample.

**DINOv2 slot model (`slot_dinov2`).** DINOv2-S with its last six blocks fine-tuned, one spread 3-slice image per slot, and a per-finding query head.

**RadImageNet heads (`radimagenet`).** The frozen official RadImageNet ResNet-50 (**marwanmath/resnet-50-radimagenet-marwan**, `ResNet50.pt`, verified by SHA-256) reads the centre slice of every window once. One query head per fold then trains: AdamW 2e-4, batch 48, at most 24 epochs, stopping after 5 epochs without a pseudo-validation gain.

Expected outputs per family folder: **fold_N.pt** (EMA weights for the reader), **last_fold_N.pt**, **history_fold_N.csv**, **oof_fold_N.csv**, **gold_fold_N.csv**, **folds.csv**, **targets_used.csv**, **run_config.json**.

If a session is interrupted, attach its saved output to a fresh run of the same session and set `RESUME_DIR` to that output. Folds with both `fold_N.pt` and `last_fold_N.pt` continue at their next epoch; resume rejects changed recipes or package versions.

CoAtNet-RMLP-2 @384 (`reader_coatnet`) is not part of the plan: it ran out of memory on a T4 in the preflight.

## 5. Offline submission

Upload `05_submit.ipynb`. Attach the competition, the `knee-wheelhouse` dataset (offline packages; not the cache), and every session output. Enable GPU, disable Internet.

- **Members.** Every attached `fold_N.pt` is a member, and each is rebuilt from its own checkpoint recipe.
- **Checks.** Members must share preprocessing, slots and labels, and a duplicated family-fold stops the run.
- **Blending.** Members are rank-averaged within their family, then families are averaged with equal weight (`FAMILY_WEIGHTS`, fixed before scoring). A five-fold family therefore does not outvote a two-fold one.

After success, inspect **submission.csv** and **inference_diagnostics.json**: members, families, blank studies, tie sizes. For an early check, attach only session A's output.

## 6. Noisy students (2026-10-08) and test-time self-training

- **Teacher.** `07_teacher_targets.ipynb` (CPU) averages the out-of-fold predictions of the reader sessions A–F into `teacher_targets.csv`. Student sessions can also build the same teacher in-session when the A–F outputs are attached instead.
- **Students.** Sessions M, N and O train `reader_student` folds 0–4 plus a second seed on fold 0. Each target cell mixes the report consensus with the teacher; the teacher share runs from 0.5 where the report readers agree to 0.9 where they split. Whole MRI series are dropped at random during training (slot dropout 0.2).
- **Submissions.** `05c_submit_student.ipynb` scores the students alone. `05d_submit_student_ttst.ipynb` adds transductive self-training on the unlabeled test volumes (`TTST_EPOCHS`). Details and attribution are in `METHOD_NOTES.md`.
- **Without the browser.** `python scripts/kaggle_push.py SLUG RUN_ON_KAGGLE/NOTEBOOK.ipynb --gpu --competition --datasets knee-wheelhouse --kernels SESSION,...` pushes and starts a notebook with the pinned Docker image (needs `~/.kaggle/kaggle.json`). Kaggle allows two GPU batch sessions at a time; a push beyond that is refused, so retry it later.

## Local maintenance only

```bash
cd /Users/shreyanshgoyal/Documents/compi
.venv/bin/python scripts/build_launchers.py
.venv/bin/python -m pytest -q
```

Local tests and notebook runs use synthetic data. Real Kaggle data, pretrained downloads, GPU throughput and leaderboard scores must be checked by the runs above.
