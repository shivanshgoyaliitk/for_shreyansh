import os, gc, copy, math, time, json, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ---- inputs ------------------------------------------------------------------
CACHE_DIR = os.environ.get("KNEE_CACHE_DIR", "/kaggle/input/notebooks/shreyanshgoyaliitb/knee-mri-step2c-cache")
LABEL_ROOT = os.environ.get("KNEE_LABEL_ROOT", "/kaggle/input")
# Editor runs mount the competition under /kaggle/input/competitions/, CLI-pushed runs under /kaggle/input/.
COMP_DIR  = os.environ.get("KNEE_COMP_DIR") or next(
    (d for d in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection")
     if os.path.isdir(d)), "/kaggle/input/competitions/rsna-knee-abnormality-detection")
OUT       = os.environ.get("KNEE_OUT", "/kaggle/working")   # one folder per family in a session
# -----------------------------------------------------------------------------
# Several cache shards may be attached; a comma-separated list is accepted too.
CACHE_DIRS = [d for d in CACHE_DIR.split(',') if d]

LABEL_COLS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
              "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

_metas = [json.load(open(os.path.join(d, 'cache_meta.json'))) for d in CACHE_DIRS]
_recipe_only = lambda m: {k: v for k, v in m.items() if k not in ('shard', 'n_shards')}
assert all(_recipe_only(m) == _recipe_only(_metas[0]) for m in _metas), 'Cache shards come from different recipes'
CACHE_META = _recipe_only(_metas[0])
_shards = sorted(m.get('shard', 0) for m in _metas)
assert _shards == list(range(_metas[0].get('n_shards', 1))), f'Attach every cache shard exactly once (found {_shards})'
assert CACHE_META.get('version') in ('knee-2c-v2', 'knee-2c-v3', 'knee-2c-v4', 'knee-2c-v5'), 'Rebuild the step-2c cache with current preprocessing'
PLANES = CACHE_META['slots']
CACHE_IMG = CACHE_META['img']
N_SLICES = CACHE_META['n_slices']

# ---- model recipe -------------------------------------------------------------
# Eight self-trained families, chosen by PRESET (one launcher set per family):
# reader_convnext: public 2.5D study reader (0.929 alone). ConvNeXt-T on windows of three
#   adjacent slices, slot and stack-position tokens, two transformer layers, attention
#   pooling per finding, EMA weights, warmup then cosine.
# reader_student: the ConvNeXt reader trained on report labels mixed with our teachers' OOF predictions.
# reader_student2: round 2, whose teacher also includes the round-1 students.
# reader_caformer: the same reader on CAFormer-S18 (ImageNet-22k), for backbone diversity.
# reader_coatnet: the same reader on CoAtNet-RMLP-2 at 384 px, the backbone of the public
#   Raptor single model (0.928 alone). Gradient checkpointing keeps it inside a T4.
# slot_dinov2: the public DINOv2 member (one spread 3-slice image per slot, last six
#   blocks fine-tuned, per-finding query head).
# radimagenet: frozen official RadImageNet ResNet-50 slice tokens with a per-finding query
#   head (public E9/E11 arm); encoded once, all folds in one run (rad_runtime.py).
# kneenet_b0: the earlier EfficientNet-b0 model, kept for comparison.
PRESET = os.environ.get("KNEE_PRESET", "reader_convnext")
PRESETS = {
    'reader_convnext': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                            epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                            pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                            clip=3., grad_checkpoint=False),
    # Noisy student: the same ConvNeXt reader recipe trained on targets mixed with the out-of-fold
    # predictions of our own readers (teacher_targets.csv). Agreement-gated: the teacher share runs
    # from 0.5 where the report readers agree confidently to 0.9 where they are split or silent.
    # Slot dropout hides whole MRI sequences in training (test sites differ in protocol).
    'reader_student': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                           epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                           pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                           clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # Round 2: the same student recipe, whose teacher also averages the round-1 students' OOF.
    # 11 epochs: pseudo-validation plateaus by epochs 10-12, and a two-fold session then fits
    # in under two GPU hours.
    'reader_student2': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                            epochs=11, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                            pct_start=1/11, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                            clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2,
                            teacher_families={'reader_convnext': 1.0, 'reader_caformer': 1.0, 'reader_student': 1.0}),
    # Same reader, a different backbone family for ensemble diversity (convolution stages, then
    # attention; ImageNet-22k). ~23% more activation memory per image than ConvNeXt-T, whose
    # batch 4 peaked at 10.9 GB on a T4: half the micro-batch, same effective batch.
    'reader_caformer': dict(arch='reader25d', backbone='caformer_s18.sail_in22k_ft_in1k', img=CACHE_IMG,
                            epochs=14, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                            pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                            clip=3., grad_checkpoint=False),
    # Round-1 student recipe and teacher on a ResNet-50 (batch-norm CNN) for ensemble diversity:
    # LB 2026-10-09/10: more ConvNeXt students/seeds added ~0.001; round-2 teachers did not help.
    # Half the micro-batch (activation memory), same effective batch.
    'reader_student_r50': dict(arch='reader25d', backbone='resnet50.a1_in1k', img=CACHE_IMG,
                               epochs=14, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                               pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                               clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # Report-grounded student: the round-1 student recipe and teacher plus an auxiliary loss that
    # regresses each study's report embedding (scripts/report_embeddings.py; attach the
    # knee-report-embeddings dataset) from the images, at this weight relative to the label BCE.
    'reader_student_rep': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                               epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                               pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                               clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2,
                               report_aux=.2),
    # Round-1 student recipe and teacher on CAFormer-S18 (in22k): its readers (session J) beat the
    # ConvNeXt readers on pseudo-validation (AUC 0.895/0.897 vs 0.893/0.891, folds 0/1); ResNet-50
    # students fell far behind (0.915 vs 0.963), so diversity comes from this family instead.
    'reader_student_caf': dict(arch='reader25d', backbone='caformer_s18.sail_in22k_ft_in1k', img=CACHE_IMG,
                               epochs=14, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                               pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                               clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # Round-1 student recipe and teacher on ConvNeXt-Base (ImageNet-22k), ~3x ConvNeXt-T compute:
    # trained on the TPU (08_tpu_student_cnxb), where the long run does not hold up GPU sessions.
    'reader_student_cnxb': dict(arch='reader25d', backbone='convnext_base.fb_in22k_ft_in1k', img=CACHE_IMG,
                                epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # The round-1 ConvNeXt student recipe and teacher on the knee-2c-v4 cache (six v3 slots + two
    # 90 mm local slots around the header-geometry joint anchor); a separate family name so its
    # LB comparison with the v3 students (0.935 on folds 0-1) stays clean.
    'reader_student_v4': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                              epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                              pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                              clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # ConvNeXt-Base student on knee-2c-v4 (TPU): LB 2026-10-12 showed both gains separately,
    # ConvNeXt-B v3 x5 = 0.938 vs ConvNeXt-T v3 x6 = 0.936, and v4 folds 0-1 = 0.938 vs v3 = 0.935.
    'reader_student_cnxb_v4': dict(arch='reader25d', backbone='convnext_base.fb_in22k_ft_in1k', img=CACHE_IMG,
                                   epochs=14, batch=4, accum=2, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                   pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                   clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # The ConvNeXt student on knee-2c-v5 (288 px, 10 windows per series; TPU).
    # 288 px x 8 slots: half the T4 micro-batch of v4 (activation memory), same effective batch of 8.
    'reader_student_v5': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                              epochs=14, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                              pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                              clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    # ConvNeXt-Base and CAFormer-S18 students on knee-2c-v5 (TPU / GPU after the 2026-10-17 quota reset).
    # On a T4: one study per micro-batch (48 windows at 288 px) with gradient checkpointing; the TPU
    # runtime ignores both (8 studies per step across the chips, no checkpointing). 8 epochs: at 50-60
    # min per epoch on a T4, 14 would exceed the 12 h session (2026-10-14); ConvNeXt-B at 288 px
    # reached val BCE 0.375 / AUC 0.954 by epoch 3.
    'reader_student_cnxb_v5': dict(arch='reader25d', backbone='convnext_base.fb_in22k_ft_in1k', img=CACHE_IMG,
                                   epochs=8, batch=1, accum=8, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                   pct_start=1/8, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                   clip=3., grad_checkpoint=True, teacher=(.5, .9), slot_dropout=.2),
    # v5 students taught partly by heliosli's compact model (its out-of-fold predictions on the training
    # studies, 10_compact_oof_*): teacher = our ConvNeXt readers (sessions A-F, the Z5/B5 teacher; no CAFormer
    # reader outputs are attached) and the compact model at 1.0 each, i.e. half ours and half compact, its
    # per-finding ranking put on our teachers' value distribution (its report targets use other label
    # definitions, e.g. synovitis mean 0.54 vs 0.30);
    # otherwise the ConvNeXt-T / -B v5 recipes.
    'reader_student_ct_v5': dict(arch='reader25d', backbone='convnext_tiny.in12k_ft_in1k', img=CACHE_IMG,
                                 epochs=14, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                 pct_start=1/14, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                 clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2,
                                 teacher_families={'reader_convnext': 1.0, 'reader_compact': 1.0},
                                 teacher_quantile_map=('reader_compact',)),
    'reader_student_cnxb_ct_v5': dict(arch='reader25d', backbone='convnext_base.fb_in22k_ft_in1k', img=CACHE_IMG,
                                      epochs=8, batch=1, accum=8, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                      pct_start=1/8, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                      clip=3., grad_checkpoint=True, teacher=(.5, .9), slot_dropout=.2,
                                      teacher_families={'reader_convnext': 1.0, 'reader_compact': 1.0},
                                      teacher_quantile_map=('reader_compact',)),
    'reader_student_caf_v5': dict(arch='reader25d', backbone='caformer_s18.sail_in22k_ft_in1k', img=CACHE_IMG,
                                  epochs=7, batch=1, accum=8, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                                  pct_start=1/7, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                                  clip=3., grad_checkpoint=False, teacher=(.5, .9), slot_dropout=.2),
    'reader_coatnet': dict(arch='reader25d', backbone='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', img=384,
                           epochs=10, batch=2, accum=4, lr=5e-4, encoder_lr_scale=.2, wd=.02,
                           pct_start=1/10, dropout=.2, unfreeze_last=None, head_hidden=384, ema=.998,
                           clip=3., grad_checkpoint=True),
    'slot_dinov2': dict(arch='slot_query', backbone='vit_small_patch14_dinov2.lvd142m', img=224, epochs=10,
                        batch=4, accum=2, lr=1e-3, encoder_lr_scale=8e-3, wd=.02, pct_start=.15, dropout=.2,
                        unfreeze_last=6, head_hidden=256, ema=0., clip=5., grad_checkpoint=False),
    'radimagenet': dict(arch='radhead', backbone='radimagenet_resnet50', img=224, epochs=24, batch=48,
                        accum=1, lr=2e-4, encoder_lr_scale=0., wd=3e-3, pct_start=0., dropout=.15,
                        unfreeze_last=None, head_hidden=512, ema=0., clip=1., grad_checkpoint=False),
    'kneenet_b0': dict(arch='kneenet', backbone='tf_efficientnet_b0.ns_jft_in1k', img=CACHE_IMG, epochs=6,
                       batch=4, accum=1, lr=3e-4, encoder_lr_scale=.3, wd=1e-2, pct_start=.1, dropout=.4,
                       unfreeze_last=None, head_hidden=None, ema=0., clip=5., grad_checkpoint=False),
}
_preset = PRESETS[PRESET]
_needs = {'_v4': 'knee-2c-v4', '_v5': 'knee-2c-v5'}.get(PRESET[-3:])
assert _needs is None or CACHE_META.get('version') == _needs, f'Attach the {_needs} cache outputs'
ARCH = _preset['arch']
BACKBONE = os.environ.get("KNEE_BACKBONE", _preset['backbone'])
IMG = _preset['img']              # model input size; the cache is resized on the GPU if different
EPOCHS = _preset['epochs']
BATCH = _preset['batch']          # studies per micro-batch
ACCUM_STEPS = _preset['accum']    # effective batch = BATCH * ACCUM_STEPS studies
LR = _preset['lr']                # head learning rate
ENCODER_LR_SCALE = _preset['encoder_lr_scale']   # encoder rate = LR * this
WD = _preset['wd']
PCT_START = _preset['pct_start']  # warmup share of the one-cycle schedule
DROPOUT = _preset['dropout']
UNFREEZE_LAST = _preset['unfreeze_last']
HEAD_HIDDEN = _preset['head_hidden']
EMA_DECAY = _preset['ema']        # 0 disables; EMA weights are validated and saved
CLIP = _preset['clip']
GRAD_CHECKPOINT = _preset['grad_checkpoint']
TEACHER_WEIGHT = _preset.get('teacher', 0.0)   # constant or (lo, hi) gate; attach teacher_targets.csv
SLOT_DROPOUT = _preset.get('slot_dropout', 0.0) # reader25d: chance of hiding each present series in training
TEACHER_FAMILY_WEIGHTS = _preset.get('teacher_families')   # None: label_builder.TEACHER_FAMILIES
TEACHER_QUANTILE_MAP = _preset.get('teacher_quantile_map', ())   # outside teachers put on our teachers' scale
REPORT_AUX = _preset.get('report_aux', 0.0)    # weight of the report-embedding auxiliary loss; 0 disables
TRAIN_WINDOWS = 6                 # reader25d: windows encoded per series per step (of 8)
# Gold studies train in every fold at this per-cell weight (decision 2026-10-07). Their
# audit AUC is then in-sample and is printed as such; selection never reads gold.
GOLD_WEIGHT = 3.0
READER_WEIGHTS = {}               # e.g. {'extra_labels_sol_v2.csv': 2.0}; default 1 each
# Every reader the run must find; a missing attachment stops before any GPU time is spent.
REQUIRED_READERS = ['report_labels_v2.csv', 'llm_labels_v4_blend.csv', 'labels_llm_gpt56sol.csv',
                    'qwen_knee_weak_labels.csv', 'extra_labels_sol_v2.csv',
                    'extra_labels_sonnet_legacy.csv', 'pseudo_google_gemini*flash*.jsonl']
NORMALIZATION = 'pretrained'
N_FOLDS = 5
FOLDS_TO_RUN = json.loads(os.environ.get("KNEE_FOLDS", "[0,1,2,3,4]"))  # one fold first on Kaggle
GROUP_COL = 'report_group'        # folds keep duplicate reports together
RESUME = False  # copy both fold_N.pt and last_fold_N.pt to OUT before enabling
T_TRAIN     = 10       # kneenet only: contiguous slices sampled per plane per step
SEED        = int(os.environ.get("KNEE_SEED", 42))   # sessions may train extra seeds of a family
NUM_WORKERS = int(os.environ.get("KNEE_WORKERS", 4))   # Kaggle gives 4 cores; two-GPU sessions use 2 per job

def write_atomic(path, content):
    # Two jobs in one session may write the same small file (folds, config); replace it whole.
    # content: text, or a function that writes to the temporary path it is given.
    temp = f'{path}.{os.getpid()}.tmp'
    if callable(content):
        content(temp)
    else:
        with open(temp, 'w') as handle: handle.write(content)
    os.replace(temp, path)


os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True          # shapes are fixed -> let cuDNN autotune

print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
if DEV == "cuda" and torch.cuda.device_count() > 1:
    print(f"note: {torch.cuda.device_count()} GPUs visible, using 1 by design "
          f"(DataParallel benchmarked 13x slower here)")
if ARCH == 'kneenet' and BACKBONE.startswith('vit_'):
    BATCH, ACCUM_STEPS = 1, 4  # effective batch 4; lower activation memory
print(f"recipe: {PRESET} ({ARCH}) | {BACKBONE} @ {IMG}px | {CACHE_META['version']} {len(PLANES)} slots x "
      f"{N_SLICES} slices x {CACHE_IMG}px in {len(CACHE_DIRS)} shard(s) | {EPOCHS} epochs, "
      f"{BATCH}x{ACCUM_STEPS} studies/step | gold weight {GOLD_WEIGHT}")
