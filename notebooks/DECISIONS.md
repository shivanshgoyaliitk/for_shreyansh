# Study guide: every decision, the code, and likely questions

This is for the team to understand and explain the project. **It is not report text.** The course requires the report to be in your own words (about 85%), so read this, then write in your own words.

## 1. The pipeline in one paragraph

AG News sentence → fine-tuned BERT-base → sentence embedding *t* (the 768-dimensional [CLS] pooler output) → **projection** onto a learned manifold of clean embeddings → BERT's own classification layer → label.

- **TMD (the paper)** projects by random search. A helper network Q(c|t) picks the piece *c* of the manifold, 30 random latent vectors *z* are drawn, the InfoGAN generator makes G(z, c) for each, and the one closest to *t* is kept.
- **Our method** makes two changes. BERT is also trained on attacked training sentences, and the random search is replaced by a learned encoder E(t, c) → z whose output goes through the same frozen generator, t̂ = G(z, c). The projection becomes one deterministic pass, trained so that embeddings of attacked sentences map back to their clean versions.

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
| 23 | The final defense: adversarially trained BERT + learned encoder (`advbert2_enc1`), chosen on validation, attacked once on test | It won the validation screening by a wide margin | test: clean 94.66, PWWS 81.4, TextFooler 80.7, BERT-Attack 79.1 vs the paper's TMD 94.29 / 70.0 / 50.0 / 55.2; a second encoder seed: 94.80 / 73.2 / 71.2 (TextFooler, BERT-Attack) |
| 24 | Clean-accuracy check for every encoder | Encoder training sometimes falls into a stuck state (about 70% clean accuracy, one topic mislabeled). Encoders whose clean validation accuracy is more than 1 point below BERT's own are rejected and retrained with the next seed; decided from clean data only | notebook 10 (its first encoder was rejected) |
| 25 | 10: the final recipe trained on TextFooler examples only, tested on all three attacks | Answers "you trained on the same attacks you test on". No BERT-Attack or PWWS example is used anywhere, including TextFooler examples made against models that had seen them | test: 94.68 / PWWS 72.3 / TextFooler 70.1 / BERT-Attack 65.4 |
| 26 | What 08 found | the frozen-BERT encoder is not stable across seeds (52.5 / 58.1 vs 37.0 / 45.0); without the GAN the same training fails (22.5 / 27.0, validation); the encoder without attack examples is below TMD (40.0 / 49.5) although it reconstructs more closely (error 4.15 vs 6.13); inference: TMD 2.08x vanilla (paper 2.07x), ours 1.10x; averaging 5 TMD projections per query lowers its PWWS only from 65.3 to 62.7 | notebook 08 |

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

Notebooks: 00 reproduction; 01-04 early ideas (not run); 05 encoder rounds 1-3; 06 rounds 4-5 (a win that did not survive a second seed); 07 attack-free attempt (negative); 07b adversarially trained BERT + encoder (the final defense); 08 checks; 09 tables and figures; 10 the final recipe trained on TextFooler only, tested on unseen attacks.

## 5. Questions to expect, with short answers

- **Why is your TMD's PWWS 63 and not 70?** Same attack code and recipe. Our vanilla BERT is also 4 points lower on PWWS (one learning rate instead of the authors' grid). The paper's own GAN settings did worse, not better. TMD's PWWS also varies by several points from run to run because it is random.
- **Isn't adversarial training unfair against TMD?** TMD never sees attacks; ours learns from attacks on training sentences, and we say so. But training on attacks alone is not enough: our adversarially trained BERT alone scores 55.5 / 47.5 (TextFooler / BERT-Attack, validation), and the paper's own adversarial-training defenses (ASCC, DNE) are far below TMD. With the attack training held fixed, the projection decides: 55.5 / 47.5 without one, 66.0 / 66.0 with TMD's random search, 80.5 / 80.0 with our encoder.
- **Does it work on attacks it never saw?** Trained on TextFooler only (notebook 10): BERT-Attack 65.4 vs the paper's TMD 55.2 (beyond the noise), PWWS 72.3 vs 70.0 (a tie). In the authors' protocol PWWS uses the same candidate words as TextFooler with a different search, so BERT-Attack is the truly unseen one.
- **How do you know the gain comes from the projection?** The same attack examples given to the classifier alone made it worse than vanilla BERT; a network with the encoder's size trained on them without the GAN scores 22.5 / 27.0 (validation). The encoder trained without attack examples scores 40.0 / 49.5, below TMD. So the gain needs both: attack examples and the manifold projection.
- **Is TMD's PWWS score real?** It depends on the attack. Against the attack that queries it, TMD keeps 63%; on texts already crafted against plain BERT it keeps 11%. An attacker averaging 5 of TMD's projections per query lowered it only from 65.3 to 62.7 (300 sentences), so we do not claim it is just randomness; the solid evidence is the transfer gap. Our final defense keeps 81.4% under attack and 93% on transferred texts.
- **Why not optimise z by gradient descent (Defense-GAN style)?** The paper (§4.6) found random sampling better than optimisation. A learned encoder is a third option: fast, deterministic and trainable on attack examples.
- **Is the result a lucky checkpoint?** For the frozen-BERT encoder of notebook 06, yes: its second seed scored 37.0 / 45.0 against 52.5 / 58.1. For the final defense, no: two encoder seeds give 80.7 / 79.1 and 73.2 / 71.2 (mean 77.0 / 75.2), both far above TMD's 50.0 / 55.2. Only the encoder's seed was varied; BERT's adversarial training was not repeated.
- **Why did one encoder get 70% clean accuracy?** Encoder training sometimes gets stuck mislabeling one of the four topics. The best-epoch rule cannot help if every epoch is stuck, so we added a check on clean validation accuracy and retrain with another seed; it uses no attack results.
- **What would you do with more compute?** Repeat BERT's adversarial training with more seeds, a lower encoder learning rate (against the stuck state), the authors' learning-rate grid, IMDB and Yelp, and other models.
