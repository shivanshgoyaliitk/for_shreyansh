# Handover: deterministic, attack-aware TMD (NTU AI6103 group project)

This is a briefing for an agent or person picking up the project. Read it in full before changing anything. `DECISIONS.md` (same folder) has the reasoning behind every choice and the answers to likely questions. `main_results.xlsx` has every final number.

## 1. Goal and status

- **Course:** NTU AI6103 Deep Learning. Group of 5. Deadline: week 14 of the semester. AI use is allowed as long as the team can explain every decision and every line of code. The report must be in the team's own words.
- **Task:** improve the EMNLP 2022 paper *Textual Manifold-based Defense* (TMD; Nguyen and Luu) and beat a number it reports, under the paper's exact protocol, using the authors' attack code.
- **Status: the experiments are finished, and the goal is met.** What is left: the written report, the video (a deck exists, see §7), and pushing the final notebooks with their outputs to the public repository.

## 2. Rules the user set (do not break)

1. **No reverse-engineering of the attacks.** No attack internals (candidate word lists, synonym tables, search order) are used in training. Attacks are only called as black boxes on *training/validation* sentences. The user called anything else "gaming the system".
2. **Evaluate only with the authors' attack code** (the TextAttack fork vendored in `TextDefender/`), with the paper's protocol (§4). DeepWordBug was added at the user's request as a "completely novel attack", built the same way as the authors' `build_english_attacker`.
3. Selection on validation only, then **one** test run. Never choose anything on test results.
4. Nothing runs on the user's Mac. Everything runs in Google Colab (an L4 GPU) with Google Drive.
5. Keep the `runtime.unassign()` (disconnect) cells at the end of the Colab notebooks.
6. Do not push to `main` or open pull requests unless asked. Never ask for tokens in chat. Keep the user's email out of everything.
7. No people's names in figures. Credits go in the README.

## 3. Where things are

| What | Where |
|---|---|
| Working repository (full history, all notebooks incl. dead ends 01-04) | `shivanshgoyaliitk/for_shreyansh`, branch `claude/magical-pasteur-ptp2k2` |
| Public, cleaned repository (no history, no comments or markdown) | `github.com/shreyanshgoyalntu/deterministic-tmd`, branch `main` |
| All cached models, embeddings, GANs, encoders, attack logs, results | Google Drive: `MyDrive/tmd_run_paperft/ag_news/` |
| Report tables and figures written by notebook 09 | Drive: `MyDrive/tmd_run_paperft/ag_news/report_assets/` (`tables.tex`, `results_summary.md`, `fig_*.png`) |
| Shared code | `tmd_lab.py` (repo root) |
| Notebook generators (source of truth for notebooks 07-14 and 09) | the previous agent's scratchpad (`nbXX.py` + `make_nb.py`); the generated `.ipynb` files in `notebooks/` are what counts |

**Drive file names** (all in the cache folder; `tag` names a defense or model):

| Pattern | Content |
|---|---|
| `finetuned_<model>/` | a fine-tuned BERT (`base`, `advbert1`, `advbert2`, `tfbert1/2`, `babert1/2`, `mixbert1/2`, ...) |
| `emb_<model>_<split>.pt` | pooled BERT embeddings (train / val / test / train_noise) |
| `infogan_<model>...pt` | the InfoGAN for that BERT |
| `projenc_<name>_L<latent>C<code>.pt` | a trained projection encoder |
| `advpairs_<source>_<attack>_n<N>_s<seed>.jsonl` | attacked training/validation sentences (training material) |
| `results_fork_<tag>_n1000.json` | test results; `results_fork_val_<tag>_n200.json` for validation |
| `examples_fork_<tag>_n1000_<attack>.jsonl` | per-sentence attack logs (used by the McNemar tests) |
| `nb08_results.json` | notebook 08's analyses (worst case, transfer, timing, EOT) |

## 4. Method and protocol (short)

- **TMD (baseline):** BERT pooled embedding t → InfoGAN; the code c is predicted by Q(c|t); 30 random latent points G(z, c) are drawn and the one closest to t is classified by BERT's head. Random at inference.
- **Ours:** (1) BERT adversarially trained (`finetune_robust`: cross-entropy + KL + cosine consistency on noisy views and on attacked training sentences), (2) InfoGAN with the paper's Table 6 hyperparameters on the new embeddings, (3) a learned **ProjEncoder** E(t, c) → z, passed through the frozen generator: deterministic, one step. Encoder design by Cengiz (MLP 768 + code → 1024 → 512 → latent; LayerNorm; LeakyReLU; losses: latent cycle, reconstruction, denoising, typical set, adversarial, classification).
- **Clean-accuracy guard:** an encoder whose clean validation accuracy is more than 1 point below its BERT's is rejected and retrained with the next seed (42, 1, 2, 3). Encoder training sometimes sticks at about 70% (one topic mislabeled).
- **Protocol:** AG News, BERT-base, TextDefender fork; K = 50 candidates; at most 30% of words changed; USE similarity ≥ 0.8408 with window 15; query budget 50 × words; 1,000 random test sentences (seeded); clean accuracy on all 7,600 test sentences.
- **Statistics:** "BEATS the paper" means the gain exceeds the combined 95% sampling noise (about ±4 points). Exact McNemar test against our TMD reproduction on the same 1,000 sentences.

## 5. Notebooks (run order; each reads what earlier ones saved)

| Notebook | What it does | Tags it produces |
|---|---|---|
| 00 | fine-tune BERT, train the GAN, reproduce TMD and vanilla | `vanilla`, `baseline` |
| 05 | encoder rounds 1-3 on frozen BERT, transfer test, control | `enc_adv*` |
| 06 | encoder rounds 4-5, screening, first test run (a lucky seed, see 08) | `enc_adv4_m9` ... |
| 07 | attack-free robustness (negative result) | `nrb_*` |
| 07b | **the final defense**: adversarially trained BERT + encoder; second encoder seed | `advbert2_enc1`, `advbert2_tmd`, `advbert2` |
| 08 | ablations, second seed of 06, timing, worst case, EOT attack | `nb08_results.json` |
| 10 | final recipe trained on TextFooler only | `tf_only_enc1_s1` (guarded) |
| 11 | final recipe trained on BERT-Attack only | `ba_only_enc1*` |
| 12 | encoder without attack examples on 07b's BERT (validation) | `advbert2_encden` |
| 13 | final recipe trained on TextFooler + BERT-Attack; PWWS held out | `tf_ba_enc1*` |
| 14 | every defense attacked with DeepWordBug (unseen by all) | adds `deepwordbug` to existing result files |
| 09 | collects everything into tables and figures (CPU is enough) | `report_assets/` |

Notebook 09 has been extended for 13 and 14 (Table 2b now has the two-attack row; new Table 2c for DeepWordBug) but **has not been re-run** since. Re-running it in Colab is the next step (§8).

## 6. Results (all final)

Test, 1,000 sentences, accuracy under attack (%):

| Defense | Clean | PWWS | TextFooler | BERT-Attack |
|---|---|---|---|---|
| Paper's TMD | 94.29 | 70.0 | 50.0 | 55.2 |
| Our TMD reproduction | 93.84 | 63.2 | 49.8 | 52.2 |
| **Ours (final, 07b)** | **94.66** | **81.4** | **80.7** | **79.1** |
| Ours, second encoder seed | 94.80 | not run | 73.2 | 71.2 |
| Trained on TextFooler only (10) | 94.68 | 72.3 | 70.1 | 65.4 |
| Trained on BERT-Attack only (11) | 94.74 | 72.8 | 68.6 | 70.7 |
| Trained on TextFooler + BERT-Attack (13) | 94.66 | 76.4 | 72.3 | 70.6 |

DeepWordBug (14), never seen by any defense: vanilla 38.2, TMD 54.7, TextFooler-only 72.3, BERT-Attack-only 71.6, TextFooler + BERT-Attack 74.3, final 80.8. McNemar vs TMD ("only TMD survives / only this survives"): 66/242, 65/234, 41/237, 26/287, all p < 0.0001.

Other findings:
- Transfer (texts that fooled plain BERT, fed to each defense, PWWS): TMD 11.4%, ours 92.9%.
- Inference time vs plain BERT: TMD 2.08x (paper 2.07x), ours 1.10x.
- Ablation grid (validation, TextFooler / BERT-Attack): plain BERT 24.5/33.5, + TMD 50.5/57.0, + encoder 57.5/62.0; adversarially trained BERT 55.5/47.5, + TMD 66.0/66.0, + encoder 80.5/80.0; encoder without attack examples on that BERT 70.0/63.0 (ties TMD); no-GAN control 22.5/27.0.
- EOT attack (TMD's output averaged over 5 projections): PWWS only 65.3 → 62.7, so TMD's PWWS score is not "just randomness". The transfer gap is the solid evidence about TMD.
- Effort (before 13/14): 86 evaluation attack runs (42,900 sentences, 14.8 GPU h) + 19 training attack runs (19,500 sentences, 6.1 GPU h). Notebook 09 recomputes this, now including DeepWordBug.

Caveats that must appear wherever the results do:
- Ours learns from attacked training sentences; TMD does not. Compare also with the paper's attack-aware baselines (ASCC, DNE).
- Our TMD reproduction is below the paper on PWWS (63.2 vs 70.0).
- The three-attack defense (07b) also had more and different training data than 10/11/13, so the curve over the number of attacks seen mixes two effects; the 1→2 step is the fair one and is about the size of the noise.
- PWWS uses TextFooler's candidate words in this protocol, so it is a weak "unseen" attack. DeepWordBug is the strong one.
- The paper reports no DeepWordBug number; that comparison is with our reproduction.
- One dataset, one model; BERT's adversarial training ran with one seed.

## 7. Deliverables and their state

| Deliverable | State |
|---|---|
| `notebooks/DECISIONS.md` | up to date (decisions 1-30, Q&A) |
| `notebooks/main_results.xlsx` | up to date (7 sheets, incl. "Novel attack (DeepWordBug)") |
| Pipeline figure | `notebooks/figures/pipeline_tmd_vs_ours.{svg,png}`, no names |
| Video deck | a private claude.ai artifact of the user's, 10 slides, with a "typos" (DeepWordBug) slide; speaker notes on each slide |
| Public repo | 00-08, 10, 11 pushed with outputs; README, `tmd_lab.py`, 09, 12-14 and the workbook must be (re)pushed by the user (see §8) |
| Report | **not written yet**: the team writes it in their own words |
| Email to the professor | drafted earlier; the user decides whether to send it. Contribution issues go in the peer review, not the email |

## 8. Next steps

1. In Colab, re-run notebook 09 (CPU runtime) so `report_assets/` includes the two-attack row, the DeepWordBug table and the updated effort count. Download it with outputs.
2. Push to the public repo: the updated `README.md`, `results/main_results.xlsx`, `tmd_lab.py` (has DeepWordBug), notebooks 12-14 with their Colab outputs, and 09 after step 1. Clean any notebook the same way as the others (no comments, no markdown, outputs kept; noisy download/warning lines dropped).
3. Write the report from `results_summary.md`, `tables.tex` and `DECISIONS.md`.

## 9. Known problems

- **Encoder instability:** training can stick at ~70% clean accuracy. The guard handles it; mention it in the report.
- **Colab clone of the public repo fails (exit 128)** while the repo is private. Make it public, or add a `GITHUB_TOKEN` Colab secret and clone with it.
- **GitHub accounts:** the agent session's GitHub access is the `shivanshgoyaliitk` account; it cannot push to `shreyanshgoyalntu/deterministic-tmd`. The user pushes there themselves (or installs the Claude GitHub App on that account and adds the repo to the session).
- **Colab sessions disconnect.** Every notebook caches to Drive and resumes; re-run from the top.
- In notebook 14 the defense specs rebuild each model from Drive; a defense whose files are missing is skipped, not an error.

## 10. Credits

TMD: Nguyen and Luu (EMNLP 2022, MIT licence). Evaluation protocol and attack code: TextDefender (Li et al., EMNLP 2021), with a TextAttack fork (Morris et al., 2020). Encoder design, round-based adversarial training and the masking hypothesis: Cengiz. Learned-encoder idea and using a larger model: the user. Adversarially training BERT under the encoder: suggested by Claude, which wrote the code.
