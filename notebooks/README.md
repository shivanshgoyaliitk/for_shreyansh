# TMD improvement notebooks (Colab Pro)

Goal: each notebook tries one improvement to the paper's Textual Manifold-based Defense and checks it against the benchmark the authors report (Table 1, BERT rows), using the paper's own attack protocol.

**Run order:** `00` first, then any of `01`-`04` in any order. They share one cache folder on Google Drive (`/MyDrive/tmd_run/<dataset>/`): the fine-tuned model, embeddings, InfoGAN and results, so nothing is trained twice.

| Notebook | Improvement | Changes vs the paper | Main risk |
|---|---|---|---|
| `00_reproduce_tmd.ipynb` | none: reproduces the paper's TMD and vanilla rows | rebuilt attack constraints on pip `textattack`, single InfoGAN, 3 fine-tuning epochs | if this does not land near Table 1, the other notebooks mean little |
| `01_ensemble_projection.ipynb` | average R projections per query | inference only | R times the projection compute per query |
| `02_projection_aware_head.ipynb` | fine-tune the classifier head on projected embeddings | extra head-only training stage | mostly helps clean accuracy |
| `03_encoder_projection.ipynb` | learned encoder predicts the latent z, replacing the random search (plus a hybrid) | extra encoder training stage; denoising uses synonym-perturbed training embeddings | PWWS/TextFooler gains are partly 'trained on similar perturbations'; BERT-Attack is the clean check |
| `04_augmented_encoder_plus_tmd.ipynb` | BERT fine-tuned with synonym augmentation under TMD | language-model fine-tuning | a stronger defense stack, not a like-for-like manifold improvement; also reports the augmented model alone |

`tmd_lab.py` (repo root) holds everything shared: data, fine-tuning, the InfoGAN, projection variants, the encoder, the attack protocol and the comparison table.

## How "beats the benchmark" is decided
`compare()` prints our number next to the paper's TMD number (and the paper's vanilla and best-baseline numbers), plus our own reproduction from notebook 00. A metric is marked **BEATS** only if our number exceeds the paper's TMD by more than the combined 95% sampling noise (about +-3 points per attack column at 1000 examples). Otherwise it reads `tie (within noise)` or `worse`. A single run at one seed cannot establish a small gain: repeat with other seeds before claiming one.

## Protocol (TextDefender / Li et al. 2021, as in `TextDefender/utils/textattack_utils.py`)
K=50 candidate substitutes; at most 30% (AG News) or 10% (IMDB, Yelp) of words modified; sentence similarity (USE, angular) >= 0.8408 with window 15; query budget K x number of words; 1000 random test texts with more than 4 words; accuracy under attack counts originally misclassified texts as wrong; clean accuracy on the full test set.

## Known gaps, stated up front
- **Nothing here has been run by its author.** Expect small API fixes (TextAttack and Colab versions move).
- Pip `textattack` is used, not the authors' vendored fork, so attack behaviour can differ slightly from theirs.
- The authors kept the best InfoGAN from a random hyperparameter search; notebook 00 trains one.
- BERT-base only. RoBERTa/XLNet use different embeddings in the paper's code and are not supported here.
- Cost: with `N_EVAL=1000` the three attacks are slow, especially on TMD (every query runs the projection). Start with `N_EVAL=200` to smoke-test, then raise it. Results are cached per attack, so a disconnected session resumes where it stopped.
- `TMD_colab.ipynb` (repo root) is an earlier, simpler notebook with stock TextAttack settings; it does not follow the paper's protocol and is superseded by this folder.
