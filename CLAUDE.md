# RSNA Knee Abnormality Detection — current project instructions

Read `START_HERE.md` for the from-scratch Kaggle run order. Read `README.md` for the audit, limitations, and source references.
`RUN_ON_KAGGLE/` contains generated launch notebooks and `KAGGLE_RUN/` the same notebooks grouped in run order with the upload dataset; rebuild both with `scripts/build_launchers.py` after source changes.
The historical notes are preserved unchanged in `backups/original/CLAUDE.md`.
They contain outdated paths, conflicting capacity advice, and unverified leaderboard snapshots; do not treat them as current instructions.

## Current pipeline

- Labels: `label_builder.py` averages report-label readers found under /kaggle/input: four public tables, our `extra_labels_*.csv` (GPT Sol, Sonnet) from `scripts/export_reader_labels.py`, and the Gemini 3.1 Flash-Lite member file (`pseudo_google_gemini*flash*.jsonl`) from the `compi-label-generation.ipynb` output. That output's consensus `train_targets.csv` is deliberately not read. Gemini 3.1 Pro labels are not used.
- `knee_mri_step2c_cache.ipynb`: v3 cache, six typed slots × 8 windows of 3 adjacent slices, 240px, 130mm crop; built as two shards, each written as 18 ~1 GB pieces (`PIECES`) and saved as a private dataset, because Kaggle could not attach 17 GiB single files as inputs (v2 three-slot stays decodable for old checkpoints). `04_input_check.ipynb` runs the jobs' data setup on CPU before GPU sessions.
- `training.ipynb`: five-fold training with report-grouped folds; `PRESET` picks the family: `reader_convnext` (default, `model_core.Reader25D`, EMA), `reader_coatnet` (same reader, CoAtNet-RMLP-2 @384, gradient checkpointing), `slot_dinov2` (`SlotQueryNet`), `radimagenet` (`RadImageNetHead`, trained by `rad_runtime.py`, all folds in one run), `kneenet_b0` (legacy).
- Training on Kaggle runs as three two-GPU sessions (`session_runner.py`): one process per T4, same single-GPU recipe, one output folder per family. CoAtNet-RMLP-2 @384 (`reader_coatnet`) ran out of memory on a T4 in the preflight and has no launchers.
- Policy (user, 2026-10-07): train our own models wherever the full architecture and recipe are public; use other competitors' trained weights only for components whose recipe is unknown. `METHOD_NOTES.md` records sources and attribution.
- `submission-notebook.ipynb`: recipe-driven preprocessing from the checkpoint; every attached fold checkpoint is a member; ranks are averaged within each family, then across families; members must share preprocessing; supports historical three-plane best.pt too.
- Shared Python sources (`preprocessing.py`, `model_core.py`, `label_builder.py`, `train_config.py`, `train_data.py`, `dataset_runtime.py`, `train_runtime.py`, `rad_runtime.py`, `session_runner.py`, `inference_runtime.py`, `cache_runtime.py`, `validation.py`) are embedded by `scripts/build_notebooks.py`. Edit sources and rebuild notebooks together.

## Invariants

- NaN targets are unobserved; mask them, never label them negative.
- Gold studies may train at elevated weight (`GOLD_WEIGHT`, user decision 2026-10-07) but never validate: never use gold AUC to select epochs, label sources or blend weights. Once gold trains, its audit is in-sample and must be reported as such. If patient IDs are present, keep patients within folds.
- No random mirror augmentation. Deterministic side normalization is a separate preprocessing operation.
- Header laterality outranks coordinate heuristics. Review inferred/unknown sides; origin-based guesses are not ground truth.
- Train and test preprocessing and architecture must agree. Version semantic preprocessing changes; require cache rebuilds.
- Preserve slot identity and geometry in checkpoints. Do not silently reinterpret old slot labels.
- Save cache completion and presence together, atomically, after pixels are flushed.
- Avoid DataParallel; historical benchmarks here favored one T4. DDP would need a separate measured implementation.
- Do not call pseudo OOF a human-verified validation set, or infer backbone limits from underpowered ablations.
- Do not claim SOTA or improved AUC without completed validation and competition evidence.

## Verification

Run `.venv/bin/python -m pytest -q` after changes. Test real Kaggle data and accelerator runtime before submission. The local synthetic tests are not a replacement for those checks.
