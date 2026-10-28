# Study guide: every decision, the code, and likely questions

This is for the team to understand and explain the project. **It is not report text.** The course requires the report to be in your own words (about 85%), so read this, then write in your own words.

## 1. The pipeline in one paragraph

AG News sentence → fine-tuned BERT-base → sentence embedding *t* (the 768-dimensional [CLS] pooler output) → **projection** onto a learned manifold of clean embeddings → BERT's own classification layer → label.

- **TMD (the paper)** projects by random search. A helper network Q(c|t) picks the piece *c* of the manifold, 30 random latent vectors *z* are drawn, the InfoGAN generator makes G(z, c) for each, and the one closest to *t* is kept.
- **Our method** replaces the random search with a learned encoder E(t, c) → z. That makes the projection one deterministic pass, trained so that embeddings of attacked sentences map back to their clean versions.

Robustness is measured with the authors' attack code: accuracy under attack on 1000 test sentences.

## 2. Decisions, in the order they were made

| # | Decision | Why | Evidence / where |
|---|---|---|---|
| 1 | Build on the authors' repository (dangne/tmd) | The course's second project type allows reusing a 2019+ paper's code; TMD is EMNLP 2022 | repo root, `src/`, `TextDefender/` |
| 2 | AG News + BERT-base only | Short texts (about 43 words) make attacks affordable. IMDB/Yelp texts are 4-7x longer and the query budget grows with length | Table 1, BERT row |
| 3 | Use the authors' own TextAttack fork for every attack, never pip TextAttack | The attack implementation changes the numbers: on the same model, GAN and sentences, TMD's PWWS was 47.0 with pip and 66.0 with the fork, while vanilla BERT barely moved. The fork is what the paper used | notebook 00; `tmd_lab._build_attack_fork` |
| 4 | The paper's protocol exactly | Needed to compare with Table 1: 50 candidate words, at most 30% of words changed, sentence similarity (USE, angular) ≥ 0.8408 with window 15, query budget 50 × number of words, 1000 random test sentences with more than 4 words, accuracy under attack = robust / (robust + broken + already misclassified) | paper §4.1; `_build_attack_fork`, `summarize_rows` |
| 5 | In this protocol PWWS uses the same 50 nearest counter-fitted word vectors as TextFooler (not WordNet), with the authors' own PWWS search | That is how the authors' code builds it; we copy it only for evaluation, never for training | `_build_attack_fork` |
| 6 | Fine-tuning: 10 epochs, batch 64, keep the last epoch, learning rate 2e-5 | The authors' sweep configs (`src/finetune/*.yaml`). They also grid-searched the learning rate (1e-6 to 9e-4); we used one value for compute. This likely explains part of our vanilla gap on PWWS (35.3 vs 39.2) | appendix A.2; `PAPER_FT` |
| 7 | InfoGAN architecture of the paper's Tables 7/8; hyperparameters by random search over Table 4's ranges (8 trials), choosing the GAN with the lowest reconstruction error on **validation** data | Faithful architecture; selection that never sees test data. Winner: G lr 5e-4, D lr 9e-5, prior lr 1e-3, K = 100, d = 30, 5 D steps per G step | `search_gans`; appendix A.3 |
| 8 | Also tried the paper's Table 6 hyperparameters (07) | To check whether our GAN explains the PWWS gap. It did worse on validation (mean 48.25 vs 53.75), so our GAN is not a lowered baseline | notebook 07, `tmd_t6` |
| 9 | K = latent dimension, d = code dimension | We first had them swapped; fixed after reading the authors' configs | `CFG` |
| 10 | BERT-Attack in the fork skips words that BERT splits into several sub-words | That branch is commented out in the authors' code; it is the paper's setting, so kept | README "Known gaps" |
| 11 | Screen on 200 validation sentences, report on 1000 test sentences, results cached per size, attacks resumable | Cost (Colab disconnects); the 1000-sentence test set is the paper's size | `evaluate`, `_run_attack_fork` |
| 12 | The projection encoder (designed by Cengiz): MLP 768 + code → 1024 → 512 → K, LayerNorm, LeakyReLU(0.2) | Amortised GAN inversion: predict the best z directly instead of searching | `ProjEncoder` |
| 13 | Encoder losses, generator frozen | **cycle** E(G(z,c),c) ≈ z keeps E consistent with G; **reconstruction** G(E(t)) ≈ t keeps clean sentences intact; **denoising** maps perturbed embeddings to clean ones; **typical set** keeps z where G was trained (norm ≈ √K); **adversarial** maps attacked training sentences to their clean embedding; **classification** keeps the label through BERT's frozen classifier | `train_proj_encoder` docstring |
| 14 | Adversarial rounds: attack the current defense on training sentences (black box), train the next encoder on everything collected so far | A defense must hold against attacks on *itself*; each round adds what still fools it | notebooks 05-06; `adversarial_pairs` |
| 15 | Choose variants and epochs only on validation sentences, with the rule fixed before seeing results; attack the chosen one once on test | Choosing on test would inflate the result | notebooks 06, 07, 07b |
| 16 | Control: the same attack examples used to train only BERT's classification layer (no GAN, no encoder) | To show the gain is not just "more training data". It did worse than vanilla (TextFooler 19.4, PWWS 28.6, BERT-Attack 29.2) | notebook 05 |
| 17 | Transfer test: texts that fooled vanilla BERT, fed once to each defense | The attack never queries the defense, so randomness cannot confuse it. TMD keeps 10.4% (PWWS) and 23.7% (TextFooler); enc_adv4 keeps 60.0% and 69.9% | notebook 06 |
| 18 | Noise around the encoder output (06) was rejected | It collapsed (6.5% or less): each query is randomly misclassified ~10% of the time, so over hundreds of queries the attack almost surely sees one wrong answer and counts a success | notebook 06 screening |
| 19 | No attack internals in training (team rule) | Copying the attacks' candidate word lists into training would tailor the defense to the test. Attacks are only called as black boxes | discussion before 07 |
| 20 | 07: attack-free robustness (BERT trained to ignore random masking/deletion/random words) + TMD | To see whether a like-for-like improvement of TMD exists without attack knowledge. It did not (mean 50.75 vs 53.75 on validation) | notebook 07 |
| 21 | 07b: adversarially train BERT itself, then the encoder; generic noise instead of synonym lists; weight averaging | In 05-06 BERT stayed frozen and the rounds plateaued (enc_adv5 added nothing) | notebook 07b |
| 22 | 08: second seed, ablations, worst case, randomness-aware attack; 09: tables | What a grader asks for; the paper's own analyses redone | notebooks 08, 09 |

## 3. Statistics we use, and why

- **95% noise margin vs the paper** (`compare`): each accuracy has sampling error 1.96·√(p(1−p)/n). Our 1000 sentences and the paper's 1000 are different random samples, so the margin is √(ours² + paper²), about ±4.3 points per attack. "BEATS" means a gain above that.
- **Exact McNemar test vs our TMD** (`mcnemar`): both defenses were attacked on the *same* 1000 sentences, so we count sentences only one of them survives and test whether the split is 50/50 (binomial). Paired tests have much more power, which is why BERT-Attack is significant there (p = 0.0005) but not against the paper.
- **Worst case** (08): a sentence is robust only if it survives the adaptive attack *and* its transferred text.
- **Randomness-aware attack** (08, Athalye et al. 2018): TMD averages 5 projections per query, so the attacker sees TMD's expected output.

## 4. Code map (`tmd_lab.py`)

| Part | Functions | What they do |
|---|---|---|
| Setup | `setup_env`, `_use_fork` | NLTK data, the authors' TextAttack fork on the path, GPU settings |
| Paper numbers | `PAPER`, `PAPER_FT`, `CFG` | Table 1 numbers, the fine-tuning recipe, per-dataset settings |
| Data and BERT | `Lab`, `Lab.finetune`, `Lab.finetune_robust`, `Lab.use_model`, `Lab.pooled`, `Lab.embeddings`, `Lab.attack_pairs` | load AG News (10% of train as validation), fine-tune, switch models, sentence embeddings, the seeded 1000-sentence sample |
| GAN | `InfoGAN`, `train_gan`, `search_gans`, `Lab.get_gan` | the paper's InfoGAN, training, random search, loading |
| Projection | `project` | TMD's random search (`sample`), the encoder (`encoder`), and variants |
| Encoder | `ProjEncoder`, `train_proj_encoder`, `typical_set_penalty` | our learned projection and its losses |
| Defenses | `Victim` | BERT + optional projection; what the attacks query |
| Attacks | `_build_attack_fork`, `_fork_wrapper`, `_run_attack_fork`, `summarize_rows` | the paper's attacks, one sentence at a time, logged and resumable |
| Training material | `adversarial_pairs`, `adversarial_texts`, `embed_texts`, `SynonymAugmenter`, `NoiseAugmenter` | attacks on training/validation sentences, re-reading them, perturbations |
| Evaluation | `evaluate`, `screen`, `compare`, `mcnemar`, `transfer_eval`, `train_head_adv` | clean + attacked accuracy, tables, significance, transfer, the control |

Notebooks: 00 reproduction; 01-04 early ideas (not run); 05 encoder rounds 1-3; 06 rounds 4-5 and the first win; 07 attack-free attempt (negative); 07b adversarially trained BERT + encoder; 08 checks; 09 tables and figures.

## 5. Questions to expect, with short answers

- **Why is your TMD's PWWS 63 and not 70?** Same attack code and recipe. Our vanilla BERT is also 4 points lower on PWWS (one learning rate instead of the authors' grid). The paper's own GAN settings did worse, not better. TMD's PWWS also varies by several points from run to run because it is random.
- **Isn't adversarial training unfair against TMD?** TMD never sees attacks; ours learns from attacks on training sentences. We say so, and also compare with the paper's adversarial-training baselines (ASCC, DNE), which we beat on every attack. The held-out check: enc_adv3 never saw BERT-Attack and still matched TMD on it (53.8 vs 52.2).
- **How do you know the gain comes from the encoder?** The same attack examples given to the classifier alone made it worse than vanilla BERT. 08 adds a nonlinear classifier with the encoder's size and no GAN, and the encoder without attack examples.
- **Why is TMD better on PWWS?** Its random projection makes the attack's word ranking noisy. Evidence: on transferred texts TMD keeps 10% and ours 60%. 08 attacks TMD's average output to test this directly.
- **Why not optimise z by gradient descent (Defense-GAN style)?** The paper (§4.6) found random sampling better than optimisation. A learned encoder is a third option: fast, deterministic and trainable on attack examples.
- **Is the result a lucky checkpoint?** 08 retrains the whole encoder chain with a second seed and reports both seeds and their mean.
- **What would you do with more compute?** The authors' learning-rate grid, more training seeds, IMDB and Yelp, and attacks the defense never saw.
