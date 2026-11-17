# Report draft (AAAI format, max 6 pages excluding references)

Draft sentences and tables only. Rewrite every sentence in the team's own words before submitting. Numbers are from notebook 09's run; LaTeX versions of the tables are in `report_assets/tables.tex` on Drive.

---

## Title

A Deterministic, Attack-Aware Projection for Textual Manifold-based Defense

## Abstract

- Textual Manifold-based Defense (TMD) protects a text classifier by projecting each sentence embedding onto a manifold learned by an InfoGAN, using a random search over 30 generated points.
- We replace the random search with a learned encoder that predicts the latent point in one deterministic step, and we adversarially train BERT underneath it on attacked training sentences.
- Because our defense learns from attacks, we compare it first with the paper's adversarial-training defenses (ASCC, DNE), and second with TMD.
- On AG News with BERT, under the paper's protocol and the authors' attack code, it reaches 81.4 / 80.7 / 79.1% accuracy under PWWS / TextFooler / BERT-Attack, against 34.0 / 33.6 / 52.3% for the better of ASCC and DNE and 70.0 / 50.0 / 55.2% for TMD, at level clean accuracy.
- A second encoder seed is lower (72.5 / 73.2 / 71.2) and still far above ASCC/DNE; against TMD it ties on PWWS.
- It also holds on attacks it never saw, including a character-level attack (DeepWordBug: 80.8% vs TMD's 54.7%), and it runs at 1.10x the cost of plain BERT, against 2.08x for TMD.

## 1 Introduction

- Word-substitution attacks change a few words of a sentence so that a classifier changes its label while a human reads the same meaning.
- TMD (Nguyen and Luu 2022) assumes adversarial embeddings lie off the manifold of natural embeddings and projects them back before classification.
- Its projection is a random search, so its output changes from query to query, and it never learns from attacks.
- We ask two questions: can a learned projection do better than random search, and does the defense hold on attacks it was not trained on?
- Contributions:
  1. a learned, deterministic projection encoder through TMD's frozen generator;
  2. an adversarially trained BERT underneath it, using attacks only as black boxes on training sentences;
  3. an evaluation under the paper's exact protocol, with a reproduction of TMD, paired significance tests, ablations, two encoder seeds, held-out attacks and a transfer test.

## 2 Background: TMD

- A fine-tuned BERT maps a sentence x to its pooled [CLS] embedding t.
- An InfoGAN is trained on the embeddings of clean training sentences: a generator G(z, c) with a Gaussian latent z and a categorical code c, a discriminator, and an auxiliary network Q(c | t) that recovers the code.
- At inference, Q predicts the code c of the input embedding t. TMD draws k = 30 latent samples z_i, keeps the generated point G(z_i, c) closest to t, and classifies it with BERT's own classification layer.
- The paper reports the strongest robustness among its baselines on AG News with BERT (Table 1 below).

## 3 Method

**Adversarially trained BERT.**
- We continue fine-tuning BERT on clean sentences, randomly perturbed sentences (word masking, deletion, random words) and attacked training sentences.
- The loss is cross-entropy plus a KL term and a cosine term that pull a perturbed or attacked sentence's prediction and embedding towards those of its clean version.
- Attacked sentences come from running TextFooler, BERT-Attack and PWWS as black boxes on training sentences, in two rounds: against plain BERT, then against the first robust BERT. No attack internals (candidate lists, synonym tables) are used.

**Manifold.**
- An InfoGAN with the paper's Table 6 hyperparameters is trained on the new BERT's clean embeddings.

**Learned projection.**
- An encoder E(t, c) → z (MLP 768 + code → 1024 → 512 → latent, LayerNorm, LeakyReLU) predicts the latent point. The projection is G(E(t, c), c) with G frozen, so the output always lies on the generator's manifold. Only one forward pass, and no randomness.
- Training losses:

| Loss | What it does |
|---|---|
| latent cycle | E(G(z, c), c) ≈ z keeps E consistent with G |
| reconstruction | G(E(t, c), c) ≈ t keeps clean sentences intact |
| denoising | maps perturbed embeddings to their clean embedding |
| typical set | keeps ‖z‖ near √K, where G was trained |
| adversarial | maps attacked training embeddings to their clean embedding |
| classification | keeps the correct label through BERT's frozen classifier |

- The encoder is first trained without attack examples, then with them. Its epoch is chosen on attacked validation sentences.
- **Clean-accuracy check:** encoder training sometimes sticks at about 70% clean accuracy (one topic mislabeled). An encoder more than 1 point below BERT's own clean validation accuracy is rejected and retrained with another seed. This uses no attack results.

## 4 Experimental setup

- AG News (4 topics), BERT-base, 10% of the training set held out for validation.
- The paper's protocol, with the authors' own attack code (TextDefender and its TextAttack fork): 50 candidates per word, at most 30% of words changed, sentence similarity (USE) ≥ 0.8408 with window 15, query budget 50 × number of words.
- 1,000 random test sentences; clean accuracy on the full 7,600-sentence test set.
- Accuracy under attack = sentences still correctly classified after the attack / all attacked sentences.
- All choices were made on 200 validation sentences; each final defense was attacked once on test.
- Statistics: against the paper's numbers, a gain counts only if it exceeds the combined 95% sampling noise (about ±4 points). Against our own reproduction, which was attacked on the same sentences, we use an exact McNemar test.

## 5 Results

### 5.1 Reproduction of TMD

| | Paper vanilla | Our vanilla | Paper TMD | Our TMD |
|---|---|---|---|---|
| Clean | 94.39 | 94.05 | 94.29 | 93.84 |
| PWWS | 39.2 | 35.3 | 70.0 | 63.2 |
| TextFooler | 27.9 | 28.8 | 50.0 | 49.8 |
| BERT-Attack | 39.0 | 35.9 | 55.2 | 52.2 |

- TMD's gain over vanilla is reproduced closely: +27.9 / +21.0 / +16.3 against the paper's +30.8 / +22.1 / +16.2.
- Our PWWS numbers are lower for both vanilla and TMD. We used one fine-tuning learning rate instead of the authors' grid search. The paper's own GAN hyperparameters did not help (48.0 / 48.5 vs 50.5 / 57.0 on validation), so the gap is not caused by a weaker GAN.

### 5.2 Main results

| Defense | Learns from attacks | Clean | PWWS | TextFooler | BERT-Attack |
|---|---|---|---|---|---|
| ASCC (paper) | yes (synonym hulls) | 91.57 | 32.8 | 31.4 | 32.1 |
| DNE (paper) | yes (synonym hulls) | 94.09 | 34.0 | 33.6 | 52.3 |
| SAFER (paper) | no (certified) | 94.42 | 39.3 | 35.5 | 42.3 |
| TMD (paper) | no | 94.29 | 70.0 | 50.0 | 55.2 |
| TMD (our reproduction) | no | 93.84 | 63.2 | 49.8 | 52.2 |
| **Ours, seed 1** | yes | **94.66** | **81.4** | **80.7** | **79.1** |
| Ours, seed 2 | yes | 94.80 | 72.5 | 73.2 | 71.2 |
| Ours, mean of 2 seeds | yes | 94.73 | 77.0 | 77.0 | 75.2 |

- **Against defenses that also learn from attacks (the like-for-like comparison):** both seeds beat the better of ASCC and DNE on every attack, by +38.5 to +47.4 on PWWS and TextFooler and +18.9 to +26.8 on BERT-Attack. Clean accuracy is level.
- ASCC and DNE train on convex hulls of word synonyms. Ours trains on attacked sentences from the same attack algorithms used at test. Section 5.4 tests attacks that were held out from training.
- **Against TMD (not like-for-like, since TMD never sees attacks):** both seeds beat the paper's TMD on TextFooler (+30.7, +23.2) and BERT-Attack (+23.9, +16.0). On PWWS, seed 1 beats it (+11.4) and seed 2 ties (+2.5, noise ±4.0).
- **Against our TMD reproduction, on the same sentences:** both seeds win on all three attacks with p < 0.0001 (for example PWWS, sentences only one defense survives: 27 vs 209 for seed 1, 51 vs 144 for seed 2).
- The two encoder seeds differ by about 8 points, so encoder training is sensitive to the seed.

### 5.3 Where the gain comes from (ablation, 200 validation sentences)

TextFooler / BERT-Attack accuracy under attack (%):

| BERT | No projection | TMD random search | Our encoder |
|---|---|---|---|
| Plain BERT | 24.5 / 33.5 | 50.5 / 57.0 | 57.5 / 62.0 |
| Adversarially trained BERT | 55.5 / 47.5 | 66.0 / 66.0 | **80.5 / 80.0** |

| Control | TextFooler / BERT-Attack |
|---|---|
| Adversarially trained BERT + our encoder trained without attack examples | 70.0 / 63.0 |
| A network of the encoder's size trained on the same attack examples, no GAN | 22.5 / 27.0 |
| BERT trained on generic noise only + TMD (no attack examples) | 50.0 / 51.5 |

- Adversarial training alone is not enough: the robust BERT without a projection scores 55.5 / 47.5.
- With the attack training held fixed, the projection decides: 66.0 / 66.0 with TMD's random search, 80.5 / 80.0 with our encoder.
- The encoder needs attack examples: without them it only ties random search (70.0 / 63.0).
- The manifold is needed: the same attack examples without the GAN give 22.5 / 27.0.

### 5.4 Attacks never seen in training

| Trained on | TextFooler | BERT-Attack | PWWS |
|---|---|---|---|
| TMD, paper (no attacks) | 50.0 | 55.2 | 70.0 |
| TextFooler only | 70.1 | *65.4* | *72.3* |
| BERT-Attack only | *68.6* | 70.7 | *72.8* |
| TextFooler + BERT-Attack | 72.3 | 70.6 | *76.4* |
| All three (final) | 80.7 | 79.1 | 81.4 |

*Italic: an attack the defense never saw in training or epoch selection.*

- Trained on one attack, the defense beats the paper's TMD on a different, unseen attack: BERT-Attack 65.4 (+10.2) and TextFooler 68.6 (+18.6).
- On unseen PWWS it ties TMD after one attack (72.3, 72.8) and is above it after two (76.4, +6.4, one seed). PWWS uses TextFooler's candidate words in this protocol, so it is a weak held-out test.

**A new kind of attack.** DeepWordBug (Gao et al. 2018) misspells words instead of replacing them. No defense was trained on it. It was built like the authors' attacks, with the same limits.

| Defense | Attacks seen in training | Clean | DeepWordBug | Only TMD / only this survives | p vs TMD |
|---|---|---|---|---|---|
| Vanilla BERT | 0 | 94.05 | 38.2 | | |
| TMD (our reproduction) | 0 | 93.84 | 54.7 | | |
| Ours, TextFooler only | 1 | 94.68 | 72.3 | 66 / 242 | < 0.0001 |
| Ours, BERT-Attack only | 1 | 94.74 | 71.6 | 65 / 234 | < 0.0001 |
| Ours, TextFooler + BERT-Attack | 2 | 94.66 | 74.3 | 41 / 237 | < 0.0001 |
| Ours, all three | 3 | 94.66 | 80.8 | 26 / 287 | < 0.0001 |

- Every version of our defense beats TMD on this unseen, character-level attack. The robustness is not specific to word substitutions.
- More attacks seen goes with higher accuracy (54.7 → 72.3 → 74.3 → 80.8). However, the three-attack defense also had more training data, so only the step from one to two attacks is like-for-like, and it is small.

### 5.5 Is TMD's robustness real? Transfer and worst case

| Attack | Defense | Adaptive attack | Transfer from vanilla | Worst case |
|---|---|---|---|---|
| PWWS | TMD (ours) | 63.2 | 11.4 | 38.3 |
| PWWS | Ours | 81.4 | 92.9 | 81.2 |
| TextFooler | TMD (ours) | 49.8 | 23.0 | 28.0 |
| TextFooler | Ours | 80.7 | 93.1 | 80.3 |
| BERT-Attack | TMD (ours) | 52.2 | 25.9 | 34.3 |
| BERT-Attack | Ours | 79.1 | 92.0 | 78.5 |

- Transfer: adversarial texts that fooled plain BERT are fed once to each defense. TMD keeps only 11.4% on PWWS, although it keeps 63.2% against the attack that queries it. Ours keeps 92.9%.
- Worst case: a sentence counts as robust only if it survives both. Ours drops by at most 0.6 points; TMD drops to 28.0–38.3%.
- An attacker that sees TMD's average output over 5 projections lowered its PWWS accuracy only from 65.3 to 62.7 (300 sentences). So TMD's PWWS score is not explained by randomness alone; the transfer gap is the stronger evidence.

### 5.6 Cost

| Defense | Time (s, notebook 08) | × vanilla |
|---|---|---|
| Vanilla BERT | 0.71 | 1.00 |
| TMD (30 samples) | 1.48 | 2.08 (paper: 2.07) |
| Ours | 0.78 | 1.10 |

- Our projection is one forward pass of a small MLP and the generator, and it is deterministic.
- Effort: 101 evaluation attack runs (56,300 sentences, 19.0 GPU hours) and 23 attack runs on training and validation sentences (26,500 sentences, 7.9 GPU hours), on one L4 GPU.

## 6 Limitations

- Our defense learns from attacked sentences; TMD does not. ASCC and DNE also learn from perturbations, but from synonym sets rather than attack outputs. The baseline numbers are the paper's, not re-run by us.
- One dataset (AG News) and one model (BERT-base).
- Two encoder seeds, about 8 points apart; BERT's adversarial training was run once.
- Our TMD reproduction is below the paper on PWWS (63.2 vs 70.0).
- Encoder training can get stuck at about 70% clean accuracy; the clean-accuracy check catches it.
- No DeepWordBug number exists in the paper, so that comparison is with our reproduction only.

## 7 Conclusion

- A learned, deterministic projection on TMD's manifold, on top of an adversarially trained BERT, is far more robust than the paper's adversarial-training defenses. It is also more robust than TMD on TextFooler and BERT-Attack across two seeds.
- The gain needs both parts: attack examples alone, or the encoder without attack examples, do not reach it.
- It holds on attacks it never saw, including character-level typos, and it costs 1.10x plain BERT.
- Future work: more seeds, IMDB and Yelp, other models, and a stabler encoder training.

## AI use (required disclosure)

- The code was written with the help of an AI coding assistant (Claude). All design decisions, code and results were checked and can be explained by the team.

## References (check details before submitting)

- Nguyen, D. M.; and Luu, A. T. 2022. Textual Manifold-based Defense Against Natural Language Adversarial Examples. EMNLP.
- Li, Z.; et al. 2021. Searching for an Effective Defender: Benchmarking Defense against Adversarial Word Substitution. EMNLP.
- Morris, J.; et al. 2020. TextAttack. EMNLP (Demos).
- Dong, X.; et al. 2021. Towards Robustness Against Natural Language Word Substitutions (ASCC). ICLR.
- Zhou, Y.; et al. 2021. Defense against Synonym Substitution-based Adversarial Attacks via Dirichlet Neighborhood Ensemble (DNE). ACL.
- Ye, M.; Gong, C.; and Liu, Q. 2020. SAFER. ACL.
- Gao, J.; et al. 2018. Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers (DeepWordBug). IEEE S&P Workshops.
- Jin, D.; et al. 2020. TextFooler. AAAI. · Li, L.; et al. 2020. BERT-Attack. EMNLP. · Ren, S.; et al. 2019. PWWS. ACL.
- Chen, X.; et al. 2016. InfoGAN. NeurIPS.
- Athalye, A.; Carlini, N.; and Wagner, D. 2018. Obfuscated Gradients Give a False Sense of Security. ICML.
