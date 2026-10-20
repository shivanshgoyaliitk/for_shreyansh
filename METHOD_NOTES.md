# Method notes and attribution

Notes for the write-up. They separate what this pipeline does itself from the public work it adapts. Public-LB scores of our own submissions are listed under "Results"; other scores are as measured by the cited authors.

## Policy

Two final submissions are kept apart:

- **Pure-own:** every model is trained in our own Kaggle notebooks from ImageNet / self-supervised initialisations available through `timm`. No other competitor's trained competition weights are loaded. Public work enters only as (a) published architectures and recipes we re-implemented and (b) public report-label tables, which the competition rules allow as external data.
- **Hybrid:** the pure-own ensemble rank-blended with public third-party checkpoints. These are used only where their recipe is unpublished, or not reproducible on Kaggle in the time left, and where we can explain the model (user decisions 2026-10-08 and 2026-10-11). Items 8 and 9 below describe them.

## Data and preprocessing (ours)

`preprocessing.py`, cache recipe `knee-2c-v3`:

- **Series typing.** Series are typed from DICOM headers into six slots: fat-suppressed fluid-sensitive sagittal, coronal and axial; non-suppressed sagittal fluid; coronal and sagittal T1. The six-slot scheme follows the public DINOv2 baseline. An absent sequence stays absent; no other series is substituted.
- **Slice order.** Slices are ordered by projecting ImagePositionPatient onto the slice normal.
- **Laterality.** Normalised deterministically: right knees are mirrored in plane (coronal/axial) or stack-reversed (sagittal). Header laterality is used first, then the image-centre patient-x coordinate. Random flips are never used.
- **Crop and resolution.** A physical 130 mm crop using both pixel spacings, at 240 px (0.54 mm/px).
- **Slices per slot.** 8 windows of 3 adjacent slices, with centres spread over 6–94% of the stack, and a per-series 1–99 percentile intensity window.

Recipe `knee-2c-v4` (ours, 2026-10-11), the six v3 slots pixel-identical, plus two **anchored local slots**:

- **Knee-joint anchor from header geometry alone.** Left-right comes from the middle of the sagittal stack, anterior-posterior from the middle of the coronal stack, and superior-inferior from the mean of both field centres. Each stack's centre is the mean of its slices' field centres, from ImagePositionPatient, ImageOrientationPatient, PixelSpacing and the image size. No segmentation model and no external data are involved.
- **Local slots.** `SAG_LOCAL` and `COR_LOCAL` re-read the sagittal and coronal fluid-sensitive series (falling back to the non-suppressed sagittal and coronal T1 series). Each slice is cropped to 90 mm around the anchor's projection into that slice, a zero-padded square that is never distorted, at 240 px (0.375 mm/px instead of 0.54). If the projection falls outside the central 15–85% of the image, the centred crop is used.
- **Check on 60 training studies.** The anchor was found in 60/60 and used for 100% of sagittal and 98% of coronal local crops. It lies 9.5 mm (sagittal) and 4.1 mm (coronal) from the image centre on average, at most 21 mm. Montages confirmed the joint is centred. Source: `scripts/anchor_check.py`.
- **Origin of the idea.** Anchored local views were adopted from heliosli's public compact model (item 9), which locates the joint with a cartilage segmenter. Ours needs only the DICOM headers.

Recipe `knee-2c-v5` (ours, 2026-10-12): v4 at 288 px (0.45 mm/px full field, 0.31 mm/px local) with 10 windows per series (30 slices), so less of a 30–40-slice stack is skipped. It is about 88 GB, built as five CPU shard notebooks. The reader takes any number of windows unchanged. Submission notebooks decode the test set once in the recipe of their members; a v3 member can read the first six slots of a v4 decode, which are pixel-identical.
## Labels

`label_builder.py`:

- **Soft targets.** Each study and finding gets the weighted mean of report-label readers, clipped to [0.01, 0.99]. Missing readers are unobserved, not negative.
- **Public readers.** pilkwang `report_labels_v2.csv`, stevenleehans `llm_labels_v4_blend.csv`, lixin73 `labels_llm_gpt56sol.csv`, laymond `qwen_knee_weak_labels.csv`. The same four-table mean is used by the public 2.5D reader below, which reports 0.896 macro AUC for it on the 58 gold studies.
- **Our readers.** Three report-extraction runs; a fourth (Gemini 3.1 Pro) was stopped unfinished and is not used.
  - Gemini 3.1 Flash-Lite (`google/gemini-3.1-flash-lite-preview` through Kaggle's model proxy): the per-model file `pseudo_google_gemini_3_1_flash_lite_preview.jsonl` written by our `01_labels` notebook (legacy protocol), read directly in training. That notebook's consensus `train_targets.csv` is not read, since it already mixes Flash-Lite with Sonnet; its GPT-5.4-nano member file is not used.
  - GPT Sol: protocol `knee-report-labels-v2`, GPT-5.6 then GPT-6.1. A drifted window of the GPT-5.6 run (records 400–700) is excluded, leaving 4,049 of 4,349 studies.
  - Sonnet: legacy protocol.
  - Sol and Sonnet are converted by `scripts/export_reader_labels.py`. All three use the fixed state→probability maps in `label_builder.STATE_P`, set before any training:
    - legacy (the `01_labels` notebook's own map): definite 1.0, probable 0.75, possible 0.35, absent 0; unsure unobserved;
    - v2: definite 1.0, unsure 0.5, absent 0.
- **Gold.** The 58 official labels replace the soft targets and train in every fold at weight 3. They never select epochs, readers or blend weights.
- **Folds.** Five folds, grouped by normalised report text so duplicate reports never straddle a split.

## Models

1. **2.5D study reader (`model_core.Reader25D`).**
   - *Adapted from:* goodpjw2008, "RSNA Knee | Stack + 2.5D ConvNeXt MIL" (Kaggle notebook, Apache 2.0) and its dataset's `knee.py` / `train.py`. Reported 0.929 public LB alone with 3 folds.
   - *Architecture:* ConvNeXt-T per 3-adjacent-slice window, then tokens with slot and stack-position embeddings, then a 2-layer transformer, then attention pooling per finding.
   - *Recipe:* AdamW (encoder 1e-4, head 5e-4, weight decay 0.02), one warmup epoch then cosine decay, EMA 0.998, gradient clipping at 3, 14 epochs.
   - *Our differences:* our preprocessing (typed slots, laterality normalisation, physical crop, fixed cached windows with a random 6-of-8 window subset per step), our reader set, gold training, fp16 on T4.
2. **DINOv2 slot model (`model_core.SlotQueryNet`).**
   - *Adapted from:* pilkwang, "RSNA Knee baseline v1" (0.891 public LB).
   - *Architecture:* DINOv2-S with its last six blocks fine-tuned, and CLS plus mean-patch features per slot.
   - *Head:* per-finding query attention over slots.
   - *Recipe:* encoder 8e-6, head 1e-3, weight decay 0.02, 10 epochs.
   - *Our difference:* the input is one spread 3-slice image drawn from our windowed cache, averaged over all spread triplets at test time.
3. **CoAtNet reader (planned, not trained).** In the T4 preflight (2026-10-07) it ran out of memory at 2 studies per step, even with gradient checkpointing; the public Raptor models were trained off Kaggle. Kept for reference only:
   **CoAtNet reader (`Reader25D` with `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`).**
   - *Backbone from:* dreaddevelopment's Raptor models ("Knee MRI: twelve findings from a single model"; datasets raptor-knee-maxspan / native384, 0.924–0.928 public LB as single models).
   - *What they publish:* architecture, preprocessing and inference only; the training hyperparameters are not published.
   - *Ours:* we train this backbone inside our 2.5D reader with the reader recipe above (10 epochs at 384 px, gradient checkpointing). It is an adaptation of the backbone choice, not a reproduction of their training.
4. **RadImageNet heads (`model_core.RadImageNetHead`, `rad_runtime.py`).**
   - *Adapted from:* the public RadImageNet arm (E9/E11) in prvsiyan's "RSNA Knee: read the report, then the knee" and its derivative sofiaanjenje/rsna-knee-e11-train (Apache 2.0).
   - *Encoder:* the frozen official RadImageNet ResNet-50 (Mei et al., 2022; uploaded by marwanmath). Window-centre slices are scaled to [-1, 1], giving 2048-d tokens with slot and position embeddings.
   - *Head:* per-finding query cross-attention fused with the mean token.
   - *Recipe:* AdamW 2e-4, weight decay 3e-3, batch 48, at most 24 epochs, early stop after 5 epochs on pseudo-validation AUC.
   - *Our differences:* our six typed slots and cache, our label readers, gold training.
5. **Noisy student (`reader_student`, ours, 2026-10-08).**
   - *Motivation:* the test labels (like the 58 gold studies) were read from the images blind to the reports, and reports often contradict them. Competitors at 0.950–0.955 report that teacher-to-student pseudo-labels, not larger backbones, close the gap. Our own observation: image models average out report noise.
   - *Teacher:* `scripts/teacher_targets.py` (`07_teacher_targets.ipynb`, CPU) averages the out-of-fold predictions of our own ConvNeXt readers: seeds first, then families. Every value is out-of-fold, so each study's teacher value comes from models that never trained on it. No third-party checkpoint is a teacher, which keeps the student entirely our own work and free of the OAI question that hangs over Raptor. Gold studies have no OOF rows and keep their gold labels.
   - *Agreement-gated mixing (our design):* target = (1 − a) × report consensus + a × teacher, with a = 0.5 + 0.4 × (1 − |2 × consensus − 1|). Cells where the 7 report readers agree confidently keep more of the report; split or uncertain cells lean on the image teacher (up to 0.9). Cells no reader labelled take the teacher value at weight 0.9. Gold is never used to set these numbers.
   - *Slot dropout (our design):* during training each present MRI series is hidden with probability 0.2 (a study keeps at least one), so every finding must be readable from more than one sequence. The test comes from many international sites with different protocols. Inference uses every series.
   - *Recipe:* otherwise identical to the ConvNeXt reader, so the leaderboard comparison student vs teacher on the same folds isolates the label and dropout changes.
6. **Transductive test-time self-training (TTST, ours, 2026-10-09; optional, `05d_submit_student_ttst.ipynb`).**
   - *What:* inside the scoring run, after the first pass, each reader member is fine-tuned for one epoch on the unlabeled test volumes. Its target is the weighted ensemble's mean probability, with the member's usual training augmentations: affine/intensity jitter, random window subsets and slot dropout. It then predicts again.
   - *Why:* the test studies come from many international sites and scanners. Fitting the ensemble's consensus under augmentation adapts each member's features and normalisation statistics to the test distribution, and distils the ensemble into each member.
   - *Safeguards:* no label of any kind is used. The head learning rate is 1e-4 (the encoder trains at 0.2×). A member that cannot finish inside the time budget keeps its first-pass prediction. The diagnostics record every member's steps and losses.
   - *Status:* to be judged on the public leaderboard against the same submission without TTST (05c).
7. **Ensemble.** Fold and seed models are rank-averaged within each family (macro AUC reads only per-finding order). Families are then averaged with fixed weights (`FAMILY_WEIGHTS`). Out-of-fold evidence favoured reader 0.7 / slot 0.1 / RadImageNet 0.2 (below), but on the public LB the 17-model equal-weight blend (0.917) lost to two reader folds alone (0.923), so the current default is reader only (slot and RadImageNet at weight 0, skipped at inference).
   - *How the weights were chosen (2026-10-08):* by `scripts/oof_blend_check.py` (`06_oof_blend_check.ipynb`, CPU). It scores a 0.1-step grid of family weights on the pooled out-of-fold predictions of the seed-42 runs, against the pseudo (report-label) targets. Gold studies have no OOF rows, so gold never entered the choice; neither did the leaderboard.
   - *OOF macro pseudo-AUC (4,349 studies):* reader 0.889, RadImageNet 0.859, slot 0.847. Equal family weights scored 0.886, below the reader alone; 0.7 / 0.1 / 0.2 scored 0.893.
   - *Seeds:* on the 1,721 studies that are out-of-fold for both seed 42 and seed 7, the two readers rank-averaged reached 0.892, against 0.886 and 0.887 alone. This is why sessions D–F, H and I train extra reader seeds.
   - *Caveat:* pseudo AUC is a noisy-label proxy, and the per-epoch checkpoint selection used the same validation folds.
8. **Third-party member: public Raptor CoAtNet checkpoint (hybrid submissions only).**
   - *What:* `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` from the public dataset `nartaa/rsna-knee-publication-swa-weights-20261007` (CoAtNet-RMLP-2 @384, SWA; 0.949 public LB alone), run by its public inference notebook `nartaa/rsna-knee-0949-anatomical-mirror`, built on dreaddevelopment's Raptor CoAtNet/MIL recipe.
   - *Why it is allowed here:* its training recipe (external OAI images, blended labels) is not published, so under the project policy its trained weights may be used. We do not train it, and it cannot be trained on a T4 at 384 px.
   - *How:* `third_party/raptor_swa_inference.py` holds that notebook's code unmodified except for its output path, with attribution. `05b_submit_raptor_blend_w*.ipynb` runs it first in its own process (checkpoint verified by SHA-256), then our ensemble, then rank-blends per finding: (1 − w) × Raptor + w × ours, with w fixed before scoring. First runs (2026-10-08) used w = 0.35 and 0.5 with the equal-family 17-model ensemble (w = 0.5 scored 0.940 public, below Raptor alone at 0.949); later runs use the reader-only ensemble at w = 0.2 or 0.3. If the Raptor member fails, the submission is our ensemble alone.
   - *Approved by the user on 2026-10-08* after Claude Code's safety check flagged the third-party code integration.

9. **Third-party member: heliosli's compact ConvNeXt-Small model "S6" (hybrid submission 05k only).**
   - *What:* bundle `209a32c8b50a6b1c` in the public dataset `heliosli/rsna-knee-compact-v1` (four fold models `E_lrprior_s0_f{0..3}.pt`, a side classifier, `manifest.json`, `state_spec.json`), run by heliosli's public notebook `rsna-knee-blend-gold-dataset-restored` (public LB 0.954), which rank-blends it per finding with the Raptor checkpoint of item 8 (Raptor 0.7, compact 0.3, weights fixed before scoring). The model's code ships inside that notebook; a copy is kept in `third_party/heliosli_blend_gold/` (notebook, `sources/`, bundle `manifest.json`).
   - *Why it is allowed here (user decision 2026-10-11):* its training code is public, but reproducing it needs a 384 px multi-view pixel cache of roughly 200 GB and a knee-cartilage segmenter trained on the external SKM-TEA dataset, neither feasible on Kaggle before the deadline. The user allowed its weights on the condition that the report explains how the model works; the description below is from its published code and manifest.
   - *Input pipeline (`px_cache.py`, `knee_anchor_bank_v2.py`, `compact_infer.py`):* each DICOM series is rendered into a native cache in a canonical laterality (a small side classifier re-renders studies whose image orientation contradicts the header). A knee-joint anchor is found per study: the centroid of femoral/tibial cartilage and menisci segmented by an SKM-TEA V-Net on 7 central sagittal slices, checked against a header-geometry "slab prior" (left-right from the sagittal stack, anterior-posterior from the coronal stack, superior-inferior from both field centres) and replaced by that prior when implausible. Every slice is then rendered at 384 px as several views (set `v4`): the full field and 100 mm local views around the anchor (sagittal/axial: joint centre, patellofemoral region anterior-superior, posterior region; coronal: medial and lateral compartments 30 mm either side).
   - *Model (`compact_train.Net`):* 2.5D windows (three physically adjacent slices, spaced by ~3.5 mm of anatomy, as the three input channels), random 352 px crops in training and centre crops at inference; a ConvNeXt-Small encoder (ImageNet-22k pretrained) embeds each window; each window token gets learned embeddings for its series slot, view and geometry (4 numbers); per finding, a learned query attends over all window tokens of the study (tanh attention, softmax over windows), and the pooled finding token, after LayerNorm, gives that finding's logit. Training samples 176 windows per study; inference uses every slice as a window centre and averages the four fold models' logits.
   - *Training targets and loss (arm `E_lrprior`):* report-derived targets from their own LLM label pipeline (the targets file is named `probe_targets_opusv11b_rr3_plv21_cdr.csv`, suggesting Opus readings of the reports; the table itself is not published), expressed per study and finding as a likelihood ratio LR = P(report reading | finding present) / P(report reading | finding absent). The loss is the negative log-likelihood of the report under the image model's probability p: −log(p·LR + (1 − p)). The report is treated as noisy evidence about the image-defined truth rather than as the target itself, so a report that omits a finding the images show costs a bounded penalty. Gold studies are never trained on. 4 folds, 16 epochs, batch 2, AdamW (encoder lr 1e-4, head lr 1e-3, weight decay 0.05).
   - *How we run it:* `05k_submit_public954_own_students_w20.ipynb` runs heliosli's notebook cells unchanged except for its compact-stage deadline (8.3 h to 6.0 h, so our stage fits Kaggle's 9 h limit), keeps its submission, runs our student ensemble (05i), and rank-blends per finding: 0.8 × public stack + 0.2 × ours, fixed before scoring. The pure-own final pick contains none of this.
   - *What we took from it for our own models:* the anchored local views (our `knee-2c-v4` cache: two 90 mm slots around a header-geometry joint anchor, no external segmenter) and the likelihood-ratio view of report labels.

10. **Other backbones on the student recipe (ours).** All use the same round-1 teacher and targets:
    - ResNet-50 (`reader_student_r50`): far worse (pseudo-validation AUC 0.915 vs 0.963), dropped.
    - CAFormer-S18 (`reader_student_caf`): equal to ConvNeXt-T.
    - ConvNeXt-Base, ImageNet-22k (`reader_student_cnxb`, trained on TPU): equal on pseudo-validation, +0.002 on the public LB.

11. **TPU training (ours, `tpu_runtime.py`).** A single process drives all eight TPU v5e chips with torch_xla SPMD:
    - Each optimiser step's 8 studies (the GPU recipe's 4 × 2) are split across the chips along the batch axis, and the compiler inserts the gradient all-reduce.
    - Tensor shapes are fixed (`Reader25D.static_shapes`). Every batch holds exactly 8 studies: training drops the remainder, and prediction pads and discards it.
    - The per-series jitter runs in the CPU loader. bf16 autocast is used.
    - A time guard skips folds that would not finish inside the 9 h session.
    - One process per chip (`xmp.spawn`) failed on Kaggle's TPU VM, because its preset topology variables conflict with torch_xla's per-process setup.
    - A CPU rehearsal with torch_xla's CPU client (`09_tpu_preflight_cpu`) checked the code and the number of compiled graphs before any TPU queue slot was spent.
    - Measured speed: ConvNeXt-T on v4 ≈ 3.7 min/epoch; ConvNeXt-B on v3 ≈ 5.5 min/epoch. Checkpoints are interchangeable with the GPU ones.

12. **Tried and dropped (public-LB evidence, folds 0–1 unless noted).**
    - Round-2 noisy student, where the teacher includes round-1 students: 0.932 vs 0.935.
    - Report-grounded auxiliary loss, regressing a 32-dimensional PCA of multilingual-E5 report embeddings: 0.926 vs 0.935. Test labels are read from the images blind to the reports, so pulling features toward report content hurts.
    - Transductive test-time self-training: 0.932 vs 0.935.
    - ResNet-50 student: 0.882.
    - Adding the round-0 readers to the students: 0.933 vs 0.936.

13. **Compact-taught students (ours, teacher partly third-party; 2026-10-15).** The students can only be as good as their
    teacher, and ours is our first-generation readers (LB 0.923–0.928). heliosli's compact model (item 9) is stronger,
    and its out-of-fold predictions on the training studies can be rebuilt exactly: `compact_train.py` puts pool study
    u in fold `int(sha1(u)[:8], 16) % 4`, and fold model k never trains on fold k (gold, blanked and review-holdout
    studies are in no training set). `10_compact_oof_*` runs heliosli's `compact_infer.py` unchanged with
    `--split train`, once per fold on a folder that holds only that fold's model, and writes
    `reader_compact/oof_fold_<k>.csv`. Gold studies are skipped. Probe: 48 of 48 studies scored, about 8.5 s of T4 per
    study for one model, about 3 h per half of the training set on T4 ×2.
    - On 24 probe studies, compact and our teacher agree only moderately: Spearman 0.57–0.88 per finding.
    - Their scales differ where the label definitions differ: synovitis mean 0.54 vs 0.30, Baker's 0.09 vs 0.17.
    - `build_teacher(..., quantile_map=('reader_compact',))` keeps compact's per-finding ranking but gives it our
      teachers' value distribution before averaging.
    - Presets `reader_student_ct_v5` (ConvNeXt-T) and `reader_student_cnxb_ct_v5` (ConvNeXt-B) weight our ConvNeXt readers
      and compact 1.0 each, i.e. half and half. Only the ConvNeXt readers (sessions A-F) are attached to the student
      sessions, so the Z5/B5 teacher was ConvNeXt-only; a missing teacher family now stops the session (2026-10-19).
    - Everything else is the v5 student recipe.
    - Sessions K5 and L5 (folds 0–1) are compared on the LB with the same students taught by our readers alone (0.939
      and 0.940).

14. **External audit checks (2026-10-15).**
    - **Coordinate laterality fallback.** It sets the side for half the training studies (2,213 of 4,407; tags 2,183;
      default 11). `03b_laterality_check` ran on 600 training studies.
      - Where tags exist (306 studies), the first-file coordinate side agrees with the tag 98.7% of the time; the
        stack-centre variant agrees 99.0%.
      - On untagged studies the two methods disagree on 2 of 288.
      - Median |sagittal stack x| is 84 mm (5th percentile 41 mm).
      - Not worth a cache rebuild.
    - **Train/test window mismatch: real.** Readers encode 6 random windows per series in training and all of them at
      test time (10 on v5). `05l` (`WINDOW_TTA = 4`) averages the all-window pass with 4 random training-size subsets;
      LB: 0.940 with and without (ConvNeXt-T + ConvNeXt-B v5), so no gain; 05i stays the default.
    - **Undecoded studies.** They now take the median prediction, not the mean, which ranked them above most studies on
      rare findings.
    - **sol_v2 "unsure" rows: real but negligible.** 18 of 4,049 studies have six or more findings at 0.5.
    - **Reader base rates differ widely** (effusion: sol 0.20 vs Sonnet 0.59). Weighting readers would need a clean
      validation set, and gold may not be used for that.

## Results (our public-LB submissions)

| Submission | Public LB |
|---|---|
| ConvNeXt 2.5D readers (v3), 2 folds / 9 models | 0.923 / 0.928 |
| Noisy students (v3), folds 0–1 / ×6 | 0.935 / 0.936 |
| CAFormer students ×6; ConvNeXt + CAFormer ×12 | 0.935; 0.936 |
| ConvNeXt-Base students (v3) ×5 | 0.938 |
| Students on v4 (anchored local slots), folds 0–1 / ×10 | 0.938 / 0.938 |
| All four own families (v3 + v4) | 0.938 |
| Students on v5 (288 px, 10 windows), folds 0–1 / ×5 | 0.939 / 0.938 |
| ConvNeXt-Base students on v5, folds 0–1 | 0.939 |
| ConvNeXt-T + ConvNeXt-Base students on v5, folds 0–1 each | **0.940** |
| … + CAFormer-S18 students on v5 (7 epochs), folds 0–1 | 0.937 |
| ConvNeXt-T + ConvNeXt-Base on v5 with test-time window subsets (05l) | 0.940 |
| Public 0.954 stack ×0.8 + ConvNeXt-T + ConvNeXt-Base v5 ×0.2 | 0.953 |
| Raptor ×0.7 + own ×0.3 | 0.949 (Raptor alone 0.949) |
| Public 0.954 stack ×0.8 + own v3 / v4 students ×0.2 | 0.953 / 0.954 (stack alone 0.954) |
| Compact-taught ConvNeXt-T v5, folds 0–1 (K5) | 0.945 (readers-only teacher 0.939) |
| Compact-taught ConvNeXt-T + ConvNeXt-B v5, folds 0–1 each (K5 + L5) | **0.946** (readers-only teacher 0.940) |
| ConvNeXt-T + ConvNeXt-B v5, five folds each (readers-only teacher) | 0.939 |
| Public 0.9586 notebook (heliosli, our copy with the R6 mount fixed) | 0.958 |
| Public 0.9586 ×0.9 / ×0.8 + Z5 + B5 ×0.1 / ×0.2 | 0.957 / 0.957 |
| Public 0.9586 ×0.9 + K5 + L5 ×0.1 | 0.957 |

Public-LB differences of 0.001–0.003 are within its noise; the consistent direction across inputs (v3 → v4 → v5) is the evidence.

## Not used

The public 0.94x inference stacks and every competition-trained weight in them. Checked on 2026-10-07:

- **Re-implemented and trained by us** (architecture and recipe public): the DINOv2 ×20 family, the ConvNeXt reader, the RadImageNet heads.
- **Backbone adopted, training ours:** the Raptor CoAtNet; its training hyperparameters are unpublished.
- **Not reproducible:** mattiaangeli's CoAtNet variants (D4 depth-zone, Global96, residual-gated) and the A5 attention-pooling stage have no published description or training code. They are the only candidates for borrowed weights, and none are used.
