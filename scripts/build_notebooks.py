"""Embed reviewed source into portable Kaggle notebooks; run from repository root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/name).read_text())
def source(n,i): return ''.join(n['cells'][i]['source'])
def put(n,i,s):
    n['cells'][i]['source'] = s.strip().splitlines(keepends=True)
    if n['cells'][i]['cell_type']=='code':
        n['cells'][i]['outputs']=[]; n['cells'][i]['execution_count']=None

def save(name,n):
    for i,c in enumerate(n['cells']):
        c.setdefault('id', f'cell-{i:03d}')
        if c['cell_type']=='code': c['outputs']=[]; c['execution_count']=None
    (ROOT/name).write_text(json.dumps(n,indent=1)+'\n')


# Kept as a function so tests can rebuild notebooks without launching training.
def build():
    pre = (ROOT/'preprocessing.py').read_text()
    core = (ROOT/'model_core.py').read_text()
    validation = (ROOT/'validation.py').read_text()
    cache = read('backups/original/knee_mri_step2c_cache.ipynb')
    put(cache,0,'# Step 2c v3 — six typed slots, shared versioned MRI preprocessing\n\nRebuild the cache for this recipe (knee-2c-v3): six header-typed series slots (fat-suppressed fluid in all three planes, non-suppressed sagittal fluid, coronal and sagittal T1), 6 slices per slot across the central 60% of each stack, 130 mm physical crop at 336 px. Both spacings determine the crop; laterality uses explicit tags first, then an image-center coordinate heuristic. Inspect coordinate-derived and unknown sides before a full build. Progress atomically stores both completion and presence. The same source is embedded in submission. Random mirror augmentation remains disabled.')
    config = source(cache,2)
    config = config.replace('    !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg',
        '    import subprocess\n    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"])')
    # The recipe lives only in preprocessing.py; this cell keeps paths and decoders.
    a = config.index('IMG        ='); b = config.index('CACHE = ', a)
    config = config[:a] + config[b:]
    a = config.index('n_std = '); b = config.index('try:', a)
    config = config[:a] + '\n' + config[b:]
    config += '''
# Large recipes are built as shards: run this notebook once per shard (0, 1, ...).
SHARD = int(os.environ.get("KNEE_SHARD", 0))
'''
    put(cache,2,config)
    put(cache,4,pre+'''\ntrain = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
N_SHARDS = recipe().get('n_shards', 1)
gb = len(SLOTS) * N_SLICES * IMG * IMG * len(train) / 1e9 / N_SHARDS
print(f"{PREPROCESS_VERSION}: {len(train):,} studies x {len(SLOTS)} slots x {N_SLICES} slices x {IMG}px"
      f" = {gb:.1f} GB per shard; building shard {SHARD} of {N_SHARDS}")
assert gb < 19, "over the 20 GB output cap"
train_series = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))
train_series['series_dir'] = [os.path.join(DATA_DIR,'train_series',str(st),str(se))
    for st,se in zip(train_series['StudyInstanceUID'],train_series['SeriesInstanceUID'])]
manifest = build_manifest(train,train_series)
''')
    for i in [6,8]: put(cache,i,'# Implemented above by the embedded preprocessing.py source.')
    put(cache,10,'# load_series is defined by the shared preprocessing source above.')
    preview = source(cache,11)
    for old, new in [('load_series(row["dir_1"], "Coronal", CANON_SIDE)', 'load_series(row["dir_1"], "Coronal", CANON_SIDE, **series_kwargs(recipe()))'),
                     ('load_series(row["dir_1"], "Coronal", row["study_side"])', 'load_series(row["dir_1"], "Coronal", row["study_side"], **series_kwargs(recipe()))')]:
        assert old in preview; preview = preview.replace(old, new)
    put(cache,11,preview)
    put(cache,9,'## Decoder\n\nShared with inference: canonical slice ordering, side normalization, anisotropic physical cropping, and nearest-valid replacement for failed slices.')
    put(cache,13,(ROOT/'cache_runtime.py').read_text())
    put(cache,15,"print(index.head().to_string(index=False))\nprint('Recipe:', meta)")
    put(cache,16,'## Train\n\nAttach this output to training.ipynb and set CACHE_DIR to its actual Kaggle path. The training notebook reads cache_meta.json and embeds the full recipe in every checkpoint. Old cache files must be rebuilt.')
    save('knee_mri_step2c_cache.ipynb',cache)
    train = read('backups/original/training.ipynb')
    for i in [2,3,4,18]: put(train,i,'# Run cells in order; no working-directory changes are needed.')
    put(train,2,'''# Training requires Internet for pretrained weights and this small split helper.
import importlib.util, subprocess, sys
if importlib.util.find_spec('iterstrat') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','iterative-stratification==0.1.9'])
''')
    config = source(train,5)
    config = config.replace('"/kaggle/input/notebooks/shreyanshgoyaliitb/knee-mri-step2b-cache6"','os.environ.get("KNEE_CACHE_DIR", "/kaggle/input/notebooks/shreyanshgoyaliitb/knee-mri-step2c-cache")')
    # Superseded below by train_config.py; the edits above document the historical cell.
    config=config.replace('LABELS    = "/kaggle/input/datasets/shreyanshgoyaliitb/pseudo-labels/pseudo_labels.csv"', 'LABELS = os.environ.get("KNEE_LABELS", "/kaggle/input/notebooks/shreyanshgoyaliitb/compi-label-generation/train_targets.csv")')
    a=config.index('_slots ='); b=config.index('BACKBONE',a)
    config=config[:a]+'''_meta_path = os.path.join(CACHE_DIR, 'cache_meta.json')
assert os.path.exists(_meta_path), 'Attach the rebuilt step-2c v2 output and set CACHE_DIR'
CACHE_META = json.load(open(_meta_path))
assert CACHE_META.get('version') == 'knee-2c-v2', 'Rebuild the step-2c cache with v2 preprocessing'
PLANES = CACHE_META['slots']
NORMALIZATION = 'pretrained'
DROPOUT = .4
N_FOLDS = 5
FOLDS_TO_RUN = json.loads(os.environ.get("KNEE_FOLDS", "[0,1,2,3,4]"))  # one fold first on Kaggle
GROUP_COL = None  # automatically populated below if patient IDs are supplied
RESUME = False  # copy both fold_N.pt and last_fold_N.pt to OUT before enabling
ENCODER_LR_SCALE = .3
''' + config[b:]
    config=config.replace('BACKBONE    = "tf_efficientnet_b0.ns_jft_in1k"','BACKBONE = os.environ.get("KNEE_BACKBONE", "tf_efficientnet_b0.ns_jft_in1k")\n# Stronger-pretraining experiment: vit_small_patch16_dinov3.lvd1689m; compare the same folds.')
    config=config.replace('EPOCHS      = 8','EPOCHS      = 6')
    put(train,5,(ROOT/'train_config.py').read_text())
    data=source(train,7)
    data += '''\nassert cache.dtype == np.uint8 and len(index) == len(cache)
assert len(PLANES) == N_PLANES and IMG == CACHE_META['img'] and N_SLICES == CACHE_META['n_slices']
assert index['StudyInstanceUID'].is_unique and index['row'].is_unique
assert index['row'].between(0,len(cache)-1).all()
assert df['source'].isin(['gold','pseudo']).all(), 'Unknown label source'
assert ((np.isnan(y_all)) | ((y_all>=0)&(y_all<=1))).all(), 'Targets outside [0,1]'
# Recover human labels from authoritative train.csv even when a source column was supplied.
original = pd.read_csv(os.path.join(COMP_DIR,'train.csv'))
original[LABEL_COLS] = original[LABEL_COLS].apply(pd.to_numeric,errors='coerce')
gold_table = original.loc[original[LABEL_COLS].notna().any(axis=1)].set_index('StudyInstanceUID')
mask_gold = df['StudyInstanceUID'].isin(gold_table.index)
df.loc[mask_gold,'source'] = 'gold'
df.loc[mask_gold,LABEL_COLS] = gold_table.loc[df.loc[mask_gold,'StudyInstanceUID'],LABEL_COLS].to_numpy()
assert not (df['source'].eq('gold') & ~mask_gold).any(), 'Gold rows absent from train.csv'
# If patient IDs exist, keep both knees and repeated studies in one fold; exclude gold patients.
for candidate in ('PatientID','patient_id','PatientId'):
    if candidate in original and original[candidate].notna().all():
        df = df.merge(original[['StudyInstanceUID',candidate]],on='StudyInstanceUID',validate='one_to_one')
        GROUP_COL = candidate
        gold_groups = set(df.loc[df['source'].eq('gold'),candidate])
        df = df[~(df['source'].eq('pseudo') & df[candidate].isin(gold_groups))].reset_index(drop=True)
        break
if GROUP_COL is None: print('No patient identifier available: folds are study-disjoint only.')
df = df[df[LABEL_COLS].notna().any(axis=1)].reset_index(drop=True)
'''
    put(train,7,data)
    # Superseded: labels now come from report-label readers merged by label_builder.py.
    put(train,7,(ROOT/'label_builder.py').read_text() + '\n' + (ROOT/'train_data.py').read_text())
    put(train,8,validation)
    put(train,9,(ROOT/'dataset_runtime.py').read_text())
    put(train,10,'## 2 · Model\n\n`slot_query` (default) follows the public DINOv2 members: each series slot is one 3-channel image whose channels are slices from the lower, middle and upper thirds of the stack. A DINOv2-S encoder has its last six blocks fine-tuned at 8e-6 while the head learns at 1e-3. CLS and mean patch tokens feed a head where every finding has its own query attending over slots. Evaluation averages logits over every spread triplet of the cached slices.\n\n`kneenet` is the earlier EfficientNet-b0 model with neighbour triplets, slice attention and per-label plane attention, kept for comparison and for older checkpoints.')
    put(train,11,core)
    put(train,13,'''def masked_bce(logits, targets, weights=None):
    # BCE over (B,12) soft targets that may contain NaN. NaN cells get exactly zero
    # gradient, so a study a reader abstained on still trains the other heads. Optional
    # per-cell weights up-weight gold rows; the mean is taken over observed weight.
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0
    w = mask.float() if weights is None else weights * mask
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(targets, nan=0.0), reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1e-6)''')
    put(train,15,'# Training, checkpoint selection, and fold exports are implemented below.')
    put(train,16,(ROOT/'train_runtime.py').read_text())
    for i in [19,20,22]: put(train,i,'# Results are saved per fold: histories, OOF predictions, and untouched gold audit predictions.')
    put(train,0,'# Step 3 — reproducible five-fold training\n\nAttach the step-2c cache (v3 six-slot, or v2 three-slot) and the label output. Install `iterative-stratification` in the Kaggle training environment. Select one fold first to measure GPU memory and runtime at 336px. Checkpoint selection uses pseudo-validation soft-label BCE; gold is evaluated only after selection. OOF AUC measures agreement with noisy labels and is affected by epoch selection; it is not a human-label generalization estimate. Default recipe: `slot_query` with DINOv2-S; set BACKBONE to vit_small_patch16_dinov3.lvd1689m for a second ensemble arm on the same folds.')
    put(train,23,'## Outputs\n\nAttach fold_0.pt through fold_4.pt to submission. Keep last_fold_N.pt for resuming training; inference ignores those files. Kaggle reruns start with empty working storage: copy prior outputs to OUT before setting RESUME=True. Do not mix recipes in one checkpoint directory.')
    save('training.ipynb',train)
    sub=read('backups/original/submission-notebook.ipynb')
    config=source(sub,1)
    # The competition mounts at /kaggle/input/competitions/<slug> in the editor but at
    # /kaggle/input/<slug> for notebooks pushed with the Kaggle CLI; accept either.
    old_comp = 'COMP_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"'
    assert config.count(old_comp) == 1
    config = config.replace(old_comp, '''COMP_DIR = next((d for d in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                                "/kaggle/input/rsna-knee-abnormality-detection") if os.path.isdir(d)),
                "/kaggle/input/competitions/rsna-knee-abnormality-detection")''')
    config=config.replace('ckpt_path = os.path.join(CKPT_DIR, "best.pt")','''CKPT_PATHS = sorted(glob.glob(os.path.join(CKPT_DIR, 'fold_*.pt')))
if not CKPT_PATHS: CKPT_PATHS = [os.path.join(CKPT_DIR, 'best.pt')]
ckpt_path = CKPT_PATHS[0]''')
    config += '''\nMODERN = cfg.get('preprocessing',{}).get('version') in ('knee-2c-v2', 'knee-2c-v3', 'knee-2c-v4', 'knee-2c-v5')
assert MODERN or PLANES == ['Sagittal','Coronal','Axial'], 'Unsupported legacy slot recipe: use its matching inference notebook or retrain on 2c v2/v3'
if MODERN:
    IMG = cfg['preprocessing']['img']   # decoded test volumes use the cache geometry; models resize
# Members are decoded once, so they must share pixels, slots and targets. Architecture,
# backbone and seed may differ: each member is rebuilt from its own checkpoint recipe.
SHARED = ('preprocessing', 'planes', 'label_cols', 'n_slices')
SLOT_PREFIX_KEYS = ('img', 'n_slices', 'crop_mm', 'slice_band', 'canon_side', 'min_slices', 'slot_rule', 'windows', 'window_size')
def reads_slot_prefix(member, reference):
    # A knee-2c-v3 member runs on knee-2c-v4 volumes: v4's first six slots are v3's slots with the same
    # geometry (pixel-identical by construction), so the member reads volumes[:, :6].
    a, b = member.get('preprocessing', {}), reference.get('preprocessing', {})
    return (a.get('version') == 'knee-2c-v3' and b.get('version') == 'knee-2c-v4'
            and all(a.get(k) == b.get(k) for k in SLOT_PREFIX_KEYS) and b['slots'][:len(a['slots'])] == a['slots']
            and reference['planes'][:len(member['planes'])] == member['planes']
            and member.get('label_cols') == reference.get('label_cols') and member.get('n_slices') == reference.get('n_slices'))
seen_members = {}
for path in CKPT_PATHS:
    other_ck = torch.load(path,map_location='cpu',weights_only=False)
    other = other_ck['cfg']
    assert all(other.get(k) == cfg.get(k) for k in SHARED) or reads_slot_prefix(other, cfg), \
        f'{path}: different preprocessing, slots or labels; do not combine'
    assert MODERN or other == cfg, 'Legacy checkpoints must share one recipe'
    # Members of one family may differ by seed; a family-fold-seed may appear only once.
    key = (other.get('preset', f"{other.get('arch', 'kneenet')}:{other['backbone']}"), other_ck.get('fold', path), other.get('seed'))
    assert key not in seen_members, f'{path} duplicates {seen_members.get(key)} ({key}); attach each training output once'
    seen_members[key] = path
    del other_ck
families = {}
for family, fold, seed in seen_members: families.setdefault(family, []).append(fold if seed in (None, 42) else f'{fold}(s{seed})')
print('ensemble members:', len(CKPT_PATHS), '| families and folds:', families)
'''
    put(sub,1,config)
    legacy_decoder=source(sub,5)
    # Execute only the matching decoder; the new recipe carries all parameters in its checkpoint.
    decoder='if MODERN:\n'+''.join('    '+line+'\n' for line in pre.splitlines())
    decoder+='''    pp = cfg['preprocessing']
    RECIPE = recipe(pp['version'])
    assert preprocessing_meta(pp['version']) == pp, 'Preprocessing recipe mismatch: rebuild this notebook from the matching source'
    SLOTS = RECIPE['slots']
    SERIES_KW = series_kwargs(RECIPE)
else:
'''+''.join('    '+line+'\n' for line in legacy_decoder.splitlines())
    put(sub,5,decoder)
    legacy_manifest=source(sub,6)
    modern="test = pd.read_csv(os.path.join(COMP_DIR,'test.csv'))\nmanifest = build_manifest(test,test_series,RECIPE)"
    put(sub,6,'if MODERN:\n'+''.join('    '+x+'\n' for x in modern.splitlines())+'else:\n'+''.join('    '+x+'\n' for x in legacy_manifest.splitlines()))
    put(sub,8,core+'''\ndef rank_blend(predictions):
    return np.mean([pd.DataFrame(p).rank(pct=True,method='average').to_numpy()
                    for p in predictions],axis=0)
''')
    put(sub,10,(ROOT/'inference_runtime.py').read_text())
    put(sub,11,'# Fallback coverage and tie diagnostics are written with submission.csv above.')
    put(sub,13,'print(pd.read_csv(OUT_CSV).head().to_string(index=False))')
    save('submission-notebook.ipynb',sub)

if __name__=='__main__': build()
