# TMD improvement notebooks (Colab Pro)

Goal: each notebook tries one improvement to the paper's Textual Manifold-based Defense and checks it against the benchmark the authors report (Table 1, BERT rows), using the paper's own attack protocol.

**Run order:** `00` first, then any of `01`-`04` in any order; `05` → `06` → `07` → `07b` build on each other. They share one cache folder on Google Drive (`/MyDrive/tmd_run_paperft/<dataset>/`; the earlier 3-epoch runs stay in `/MyDrive/tmd_run/`): the fine-tuned model, embeddings, InfoGAN and results, so nothing is trained twice.

| Notebook | Improvement | Changes vs the paper | Main risk |
|---|---|---|---|
| `00_reproduce_tmd.ipynb` | none: reproduces the paper's TMD and vanilla rows | authors' fine-tuning recipe (10 epochs, batch 64, final epoch), InfoGAN search, attacks on the authors' own TextAttack fork | if this does not land near Table 1, the other notebooks mean little |
| `01_ensemble_projection.ipynb` | average R projections per query | inference only | R times the projection compute per query |
| `02_projection_aware_head.ipynb` | fine-tune the classifier head on projected embeddings | extra head-only training stage | mostly helps clean accuracy |
| `03_encoder_projection.ipynb` | (superseded by 05) learned encoder predicts the latent z, replacing the random search (plus a hybrid) | extra encoder training stage; denoising uses synonym-perturbed training embeddings | PWWS/TextFooler gains are partly 'trained on similar perturbations'; BERT-Attack is the clean check |
| `04_augmented_encoder_plus_tmd.ipynb` | BERT fine-tuned with synonym augmentation under TMD | language-model fine-tuning | a stronger defense stack, not a like-for-like manifold improvement; also reports the augmented model alone |
| `05_adversarial_encoder.ipynb` | **the team's adversarially trained encoder (enc_adv3)**, ported to the reproduced setup: rounds enc_den -> enc_adv -> enc_adv2 -> enc_adv3 on adversarial pairs regenerated against our model with the fork | encoder replaces TMD's random latent search (methods 9/10/11) | BERT-Attack is the held-out attack; includes McNemar tests, transfer tests and a head-only adversarial-training control |
| `06_beat_table1.ipynb` | push the encoder past Table 1 on TextFooler / BERT-Attack: noise around the encoder output, two more adversarial rounds with best-epoch selection | variants chosen on validation sentences only; one final test run | noise gains may be masking (transfer test included); BERT-Attack is no longer held out for the new rounds |
| `07_attack_agnostic_tmd.ipynb` | **a better TMD with nothing learned from attacks**: BERT trained to ignore random word masking, deletion and random-word replacement (consistency training), then the paper's TMD on top with the paper's Table 6 GAN; also a reproduction check of TMD with the Table 6 GAN | BERT continues training for 2 epochs on generic noise; no adversarial examples, synonym lists or attack code | same footing as the paper's TMD, so gains may be modest; the TMD variant keeps TMD's randomness (transfer test included) |
| `07b_tmd_plus_adversarial_training.ipynb` | **the combination**: BERT adversarially trained on attacked training sentences (attacks used as black boxes), a learned projection on top, one adaptive round, weight-averaged encoder | adversarial training, the family of the paper's ASCC and DNE baselines; starts from 07's BERT if it helped | must be reported as an adversarially trained defense, compared with TMD and with the paper's best baseline |

`tmd_lab.py` (repo root) holds everything shared: data, fine-tuning, the InfoGAN, projection variants, the encoder, the attack protocol and the comparison table.

## How "beats the benchmark" is decided
`compare()` prints our number next to the paper's TMD number (and the paper's vanilla and best-baseline numbers), plus our own reproduction from notebook 00. A metric is marked **BEATS** only if our number exceeds the paper's TMD by more than the combined 95% sampling noise (about +-3 points per attack column at 1000 examples). Otherwise it reads `tie (within noise)` or `worse`. A single run at one seed cannot establish a small gain: repeat with other seeds before claiming one.

## Protocol (TextDefender / Li et al. 2021, as in `TextDefender/utils/textattack_utils.py`)
All attacks run on the authors' vendored TextAttack fork (`TextDefender/textattack`), built like their `build_english_attacker` and counted like their `SimplifidResult` (`tmd_lab.setup_env()` selects it; results go to `results_fork_<tag>.json`). K=50 candidate substitutes; at most 30% (AG News) or 10% (IMDB, Yelp) of words modified; sentence similarity (USE, angular) >= 0.8408 with window 15; query budget K x number of words; 1000 random test texts with more than 4 words; accuracy under attack counts originally misclassified texts as wrong; clean accuracy on the full test set.

## Known gaps, stated up front
- **Nothing here has been run by its author.** Expect small API fixes (TextAttack and Colab versions move).
- An earlier rebuild of the same attacks on pip `textattack` (kept as `setup_env('pip')`, results in `results_<tag>.json`) matched vanilla and TMD/TextFooler but not TMD/PWWS (47.0 vs the fork's 66.0 on the same model, GAN and examples). Use the fork for every reported number.
- The authors chose a learning rate by grid search; we use one (2e-5). The InfoGAN search uses 8 trials, theirs was larger.
- The fork's BERT-Attack never substitutes words that BERT splits into several sub-word pieces (that branch is commented out in `TextDefender/textattack/transformations/word_swap_masked_lm.py` and returns no candidates). That is the paper's setting; pip TextAttack's version enumerates all sub-word combinations and is a different, stronger and much slower attack.
- BERT-base only. RoBERTa/XLNet use different embeddings in the paper's code and are not supported here.
- Cost: with `N_EVAL=1000` the three attacks are slow, especially on TMD (every query runs the projection). Start with `N_EVAL=200` to smoke-test, then raise it. Results are cached per attack, so a disconnected session resumes where it stopped.
- `TMD_colab.ipynb` (repo root) is an earlier, simpler notebook with stock TextAttack settings; it does not follow the paper's protocol and is superseded by this folder.
