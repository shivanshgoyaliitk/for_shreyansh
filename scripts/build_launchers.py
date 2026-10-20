"""Build the ordered, self-contained Kaggle entry points: python scripts/build_launchers.py."""
import copy
import json
from pathlib import Path
from build_notebooks import build, read, source, put, ROOT

DEST = ROOT / 'RUN_ON_KAGGLE'


def write(name, notebook):
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    # Remove stale widget state, outputs, dataset attachments and accelerator settings.
    notebook['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                            'language_info': {'name': 'python'}}
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'cell-{i:03d}'
        c['metadata'] = {}
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    (DEST / name).write_text(json.dumps(notebook, indent=1) + '\n')


def main():
    build()
    DEST.mkdir(exist_ok=True)
    # Launchers superseded by the sharded v3 cache and the reader/slot families.
    for stale in ['02_cache.ipynb', '02b_split_cache.ipynb', *[f'04b_train_dinov3_fold{k}.ipynb' for k in range(5)],
                  *[f'04c_train_coatnet_fold{k}.ipynb' for k in range(5)]]:  # superseded
        (DEST / stale).unlink(missing_ok=True)
    discover = (ROOT / 'kaggle_inputs.py').read_text()
    labels = read('compi-label-generation.ipynb')
    put(labels, 0, '# 01 — Generate report labels\n\nCPU, Internet ON, competition attached. Run all cells. Uses Kaggle model quota. See START_HERE.md for saving progress and resuming. Keep the same completed label output attached to every training fold.')
    config = source(labels, 2)
    config += '''\n# Optional: attach a previous label run and set its labels/ directory here.
# Copy cached requests before any API calls; never silently overwrite current progress.
RESUME_LABEL_DIR = ""
if RESUME_LABEL_DIR:
    import shutil
    from pathlib import Path
    previous = Path(RESUME_LABEL_DIR)
    assert previous.is_dir(), previous
    for cached in previous.glob('*.jsonl'):
        target = Path(OUTDIR) / cached.name
        if not target.exists(): shutil.copy2(cached, target)
'''
    put(labels, 2, config)
    auth = source(labels, 7).replace('ENV_FILE = os.path.join(WORK, ".env")',
        'import tempfile\n_AUTH_DIR = tempfile.mkdtemp(prefix="knee-auth-")\nENV_FILE = os.path.join(_AUTH_DIR, ".env")')
    auth = auth.replace('os.path.join(WORK, "_kbench_example.py")', 'os.path.join(_AUTH_DIR, "example.py")')
    put(labels, 7, auth)
    ensemble = source(labels, 35)
    ensemble += "\nRUN_SONNET = False  # retain cached Sonnet labels; opt in explicitly to new paid requests\n"
    put(labels, 35, ensemble)
    loop = source(labels, 36).replace('for model in ENSEMBLE:', "for model in sorted(ENSEMBLE, key=lambda m: 'anthropic/' in m):\n    if 'anthropic/' in model and not RUN_SONNET:\n        print(f'{model}: new requests disabled; existing labels retained')\n        continue")
    put(labels, 36, loop)
    # An incomplete second model must not erase already extracted base labels.
    publish = source(labels, 39).replace('ens_part = ens_pseudo[["StudyInstanceUID", *LABEL_COLS]].copy(); ens_part["source"] = "pseudo"', '''consensus_targets = ens_pseudo.set_index("StudyInstanceUID")[LABEL_COLS]
base_targets = pseudo.set_index("StudyInstanceUID")[LABEL_COLS]
combined = consensus_targets.combine_first(base_targets)
ens_part = combined.reset_index(); ens_part["source"] = "pseudo"
assert ens_part[LABEL_COLS].notna().any(axis=1).sum() >= 5, "Too few labelled studies; resume label generation"
print("Consensus labels used where available; base labels fill missing consensus cells.")''')
    put(labels, 39, publish)
    # Drop the optional scan visualization, unrelated to report extraction.
    labels['cells'] = labels['cells'][:40]
    write('01_labels.ipynb', labels)

    cache = read('knee_mri_step2c_cache.ipynb')
    cache['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': '''# Bundle these small packages for the Internet-OFF submission environment.
import subprocess, sys
from pathlib import Path
wheelhouse = Path('/kaggle/working/wheelhouse')
wheelhouse.mkdir(exist_ok=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
    '--no-deps', '--dest', str(wheelhouse), 'timm==1.0.29', 'pydicom',
    'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'])
print('Offline packages saved:', [p.name for p in wheelhouse.glob('*.whl')])
'''.splitlines(keepends=True), 'outputs': [], 'execution_count': None})
    n_shards = 5                                   # knee-2c-v5 (recipe n_shards): ~17.5 GB per shard
    for shard in range(n_shards):
        n = copy.deepcopy(cache)
        cfg_cell = next(i for i, c in enumerate(n['cells']) if 'KNEE_SHARD' in ''.join(c['source']))
        put(n, cfg_cell, source(n, cfg_cell).replace('SHARD = int(os.environ.get("KNEE_SHARD", 0))',
            f'SHARD = {shard}\n# ~1 GB cache files: Kaggle could not mount the 17 GiB single-file shards (2026-10-07).\nPIECES = 18'))
        put(n, 0, f'# 02 — Build cache shard {shard} of {n_shards} (knee-2c-v5)\n\nCPU, Internet ON, competition attached. Run every shard notebook (they may run at the same time) and keep each saved output: training attaches all of them. v5 = v4 (six typed slots plus two 90 mm local slots around the header-geometry knee-joint centre) at 288 px with 10 windows per series. ' + ('This shard also saves the offline wheelhouse used by submission.' if shard == 0 else 'The wheelhouse is saved by shard 0.'))
        if shard:
            n['cells'] = n['cells'][:-1]                     # wheelhouse only in shard 0
        write(f'02_cache_shard{shard}.ipynb', n)
    # Built from the shared model source so the preflight measures the model that trains.
    core = (ROOT / 'model_core.py').read_text()
    preflight = {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': '# 03 — GPU preflight\n\nGPU T4, Internet ON. Times one training step and one evaluation pass of the 2.5D reader and the DINOv2 slot model on synthetic data at the real v3 cache geometry (6 slots x 24 slices x 240 px), and projects GPU minutes per fold. All slots are present here, so this is an upper bound. Run before the five training folds.'},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': '''import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'timm==1.0.29'])
import gc, json, math, time, torch, timm
assert torch.cuda.is_available(), 'Select a GPU before running this preflight'
print('torch',torch.__version__,'timm',timm.__version__,'GPU',torch.cuda.get_device_name(0))'''},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': core},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': '''# (name, arch, backbone, studies per step, slots, train slices, eval slices)
ARMS = [('reader_convnext_t', 'reader25d', 'convnext_tiny.in12k_ft_in1k', 4, 6, 24, 24),
        ('slot_dinov2_s', 'slot_query', 'vit_small_patch14_dinov2.lvd142m', 8, 6, 3, 24)]
IMGS = {'reader_convnext_t': 240, 'reader_coatnet_rmlp2_384': 384, 'slot_dinov2_s': 224}
TRAIN_STUDIES, VAL_STUDIES = 3540, 870   # one fold: 3,480 pseudo + 58 gold train studies
EPOCHS = {'reader_convnext_t': 14, 'reader_coatnet_rmlp2_384': 10, 'slot_dinov2_s': 10}
results = []
for name, arch, backbone, batch, slots, t_train, t_eval in ARMS:
    model = None
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        cfg = dict(arch=arch, backbone=backbone, img=IMGS[name], drop=.2,
                   normalization='pretrained', unfreeze_last=6 if arch == 'slot_query' else None,
                   head_hidden=256 if arch == 'slot_query' else 384, train_windows=6,
                   preprocessing={'slice_band': [0.06, 0.94]})
        model = build_model(cfg, 12, slots, pretrained=False).cuda().train()
        if 'coatnet' in backbone: model.enc.set_grad_checkpointing(True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        x = torch.randint(0, 256, (batch, slots, t_train, 240, 240), dtype=torch.uint8, device='cuda')
        present = torch.ones((batch, slots), dtype=torch.bool, device='cuda')
        target = torch.zeros((batch, 12), device='cuda')
        times = []
        for step in range(6):
            torch.cuda.synchronize(); started = time.time()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, present)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            torch.cuda.synchronize(); times.append(time.time() - started)
        train_step = sorted(times[2:])[len(times[2:]) // 2]
        model.eval()
        xe = torch.randint(0, 256, (batch, slots, t_eval, 240, 240), dtype=torch.uint8, device='cuda')
        with torch.inference_mode():
            model(xe, present); torch.cuda.synchronize(); started = time.time()
            for _ in range(3): model(xe, present)
            torch.cuda.synchronize(); eval_step = (time.time() - started) / 3
        minutes = EPOCHS[name] * (math.ceil(TRAIN_STUDIES / batch) * train_step
                            + math.ceil(VAL_STUDIES / batch) * eval_step) / 60
        results.append(dict(arm=name, backbone=backbone, batch=batch, train_step_s=round(train_step, 3),
                            eval_step_s=round(eval_step, 3), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                            projected_gpu_minutes_per_fold=round(minutes, 1)))
    except Exception as exc:
        results.append(dict(arm=name, backbone=backbone, error=repr(exc)))
    print(results[-1], flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
print('Projection excludes data loading; compare it with the first real epoch time.')
with open('/kaggle/working/gpu_preflight.json', 'w') as f: json.dump(results, f, indent=2)'''},
    ]}
    for c in preflight['cells']:
        c['source'] = c['source'].splitlines(keepends=True)
    write('03_gpu_preflight.ipynb', preflight)

    template = read('training.ipynb')
    put(template, 2, source(template, 2) + '''\n# Keep the preinstalled CUDA torch/torchvision stack; match inference's timm.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'timm==1.0.29'])
''')
    config = source(template, 5)
    start = config.index('CACHE_DIR ='); end = config.index('COMP_DIR ', start)
    config = config[:start] + discover + '''\n# Cache shards and label tables are discovered under /kaggle/input. Set CACHE_DIR to a
# comma-separated list of shard folders only if the discovery is ambiguous.
INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")   # other hosts: where inputs live
CACHE_DIR = ','.join(str(p.parent) for p in attached_files('cache_meta.json', INPUT_ROOT))
assert CACHE_DIR, 'Attach the 02_cache shard outputs'
LABEL_ROOT = INPUT_ROOT
''' + config[end:]
    # Session jobs keep FOLDS, PRESET, OUT and the resume folder as environment settings.
    job_config = config + '\nassert DEV == "cuda" or os.environ.get("KNEE_ALLOW_CPU"), "Enable a GPU before starting training"\n' + '''
RESUME_DIR = os.environ.get("KNEE_RESUME_DIR", "")
if RESUME_DIR:
    import shutil
    for fold in FOLDS_TO_RUN:
        for filename in (f'fold_{fold}.pt', f'last_fold_{fold}.pt'):
            previous = Path(RESUME_DIR) / os.environ.get('KNEE_RUN_NAME', PRESET) / filename
            if not previous.is_file():
                continue                       # this fold has no earlier progress
            destination = Path(OUT) / filename
            assert not destination.exists(), f'Restart the session before copying {destination}'
            shutil.copy2(previous, destination)
            RESUME = True
'''
    config = config.replace('FOLDS_TO_RUN = json.loads(os.environ.get("KNEE_FOLDS", "[0,1,2,3,4]"))  # one fold first on Kaggle', 'FOLDS_TO_RUN = [FOLD_NUMBER]')
    config = config.replace('# one T4 saturates here: batch 8 gave 364 img/s vs 369', '# initial batch; verify memory with the GPU preflight')
    config += '\nassert DEV == "cuda", "Enable a GPU before starting training"\n'
    config += '''\n# Optional interrupted-fold recovery: attach that fold's previous saved output.
RESUME_DIR = ""
if RESUME_DIR:
    import shutil
    for fold in FOLDS_TO_RUN:
        for filename in (f'fold_{fold}.pt', f'last_fold_{fold}.pt'):
            previous = Path(RESUME_DIR) / filename
            assert previous.is_file(), previous
            destination = Path(OUT) / filename
            assert not destination.exists(), f'Restart the session before copying {destination}'
            shutil.copy2(previous, destination)
    RESUME = True
'''
    preset_line = 'PRESET = os.environ.get("KNEE_PRESET", "reader_convnext")'
    assert preset_line in config
    inputs_note = ('Attach the competition, both cache datasets made from the 02 shard outputs '
                   '(knee-cache-v3-shard0/1, latest version: 18 files of ~1 GB each), the four public report-label datasets, your reader_labels dataset '
                   'and the 01_labels output (compi-label-generation, for the Flash-Lite labels). '
                   'Inputs are detected automatically.')
    # CoAtNet-RMLP-2 @384 ran out of memory on a T4 in the preflight; it has no launchers.
    arms = [('04_train_fold', 'reader_convnext', '2.5D ConvNeXt-T study reader (family 1)'),
            ('04b_train_slot_fold', 'slot_dinov2', 'DINOv2-S slot model (family 2)')]
    for prefix, preset, title in arms:
        arm_config = config.replace(preset_line, f'PRESET = "{preset}"')
        for fold in range(5):
            n = copy.deepcopy(template)
            put(n, 5, arm_config.replace('[FOLD_NUMBER]', f'[{fold}]'))
            put(n, 0, f'# 04 — Train fold {fold}: {title}\n\nGPU, Internet ON. {inputs_note} Run all cells and save the executed output. All folds of a family must use identical inputs and settings.')
            put(n, 23, f'## Next\n\nSave this executed output containing `fold_{fold}.pt`, `last_fold_{fold}.pt`, history and audit CSVs, then attach it to 05_submit.ipynb. Folds run independently; a family may contribute fewer than five folds.')
            write(f'{prefix}{fold}.ipynb', n)
    # Family 4: RadImageNet heads. The encoder is frozen, so all five folds train in one run.
    n = copy.deepcopy(template)
    put(n, 5, config.replace(preset_line, 'PRESET = "radimagenet"').replace('[FOLD_NUMBER]', '[0, 1, 2, 3, 4]'))
    put(n, 16, (ROOT / 'rad_runtime.py').read_text())
    put(n, 0, f'# 04d — Train all folds: RadImageNet heads (family 4)\n\nGPU, Internet ON. {inputs_note} Also attach the official RadImageNet ResNet-50 weights (marwanmath/resnet-50-radimagenet-marwan, file ResNet50.pt). The frozen encoder reads every study once, then one head per fold trains; all five fold_N.pt files are saved by this single run.')
    put(n, 23, '## Next\n\nAttach this output (fold_0.pt … fold_4.pt) to 05_submit.ipynb.')
    write('04d_train_radimagenet.ipynb', n)

    # Two-GPU sessions: the same job code as the single notebooks, one process per GPU.
    job_cells = [job_config] + [source(template, i) for i in (7, 8, 9, 11, 13)]
    train_job = '\n\n'.join(job_cells + [source(template, 16)])
    rad_job = '\n\n'.join(job_cells + [(ROOT / 'rad_runtime.py').read_text()])
    runner = (ROOT / 'session_runner.py').read_text()
    rad_note = ' Also attach the official RadImageNet ResNet-50 weights (marwanmath/resnet-50-radimagenet-marwan, file ResNet50.pt).'
    student_note = (' Also attach the output of 07_teacher_targets (teacher_targets.csv: out-of-fold predictions of our'
                    ' own readers); the run stops if it is missing.')
    r50_note = (' Attach the reader session outputs A-F (as session M): the session builds its teacher from their'
                ' out-of-fold predictions.')
    v5_note = (' Attach the FIVE knee-cache-v5-shard0..4 outputs (only one cache recipe may be attached) and the reader'
               ' session outputs A-F (teacher, as session M).')
    v4_note = (' Attach the THREE knee-cache-v4-shard0/1/2 outputs instead of the v3 cache datasets (only one cache'
               ' recipe may be attached) and the reader session outputs A-F (teacher, as session M).')
    rep_note = r50_note + ' Also attach the knee-report-embeddings dataset (scripts/report_embeddings.py).'
    student2_note = (' Attach the reader session outputs A-F and the round-1 student outputs M, N and O: the session'
                     ' builds its teacher from all their out-of-fold predictions.')
    sessions = [
        ('04_session_A_reader_f0_f1', {0: [('reader_convnext', [0])], 1: [('reader_convnext', [1])]},
         'ConvNeXt reader folds 0 and 1, one per GPU.', ''),
        ('04_session_B_reader_f2_f3', {0: [('reader_convnext', [2])], 1: [('reader_convnext', [3])]},
         'ConvNeXt reader folds 2 and 3, one per GPU.', ''),
        ('04_session_C_reader_f4_slot_rad', {0: [('reader_convnext', [4])],
                                             1: [('slot_dinov2', [0, 1, 2, 3, 4]), ('radimagenet', [0, 1, 2, 3, 4])]},
         'ConvNeXt reader fold 4 on GPU 0; on GPU 1 the five DINOv2 slot folds, then the five RadImageNet heads.', rad_note),
        # Extra seeds of the strongest family: same recipe, another seed (and fold split); the
        # submission averages them within the reader family.
        ('04_session_D_reader_s7_f0_f1', {0: [('reader_convnext', [0], 7)], 1: [('reader_convnext', [1], 7)]},
         'ConvNeXt reader, seed 7, folds 0 and 1 (a second member of the reader family).', ''),
        ('04_session_E_reader_s7_f2_f3', {0: [('reader_convnext', [2], 7)], 1: [('reader_convnext', [3], 7)]},
         'ConvNeXt reader, seed 7, folds 2 and 3.', ''),
        ('04_session_F_reader_s7_f4_s11_f0', {0: [('reader_convnext', [4], 7)], 1: [('reader_convnext', [0], 11)]},
         'ConvNeXt reader, seed 7 fold 4 on GPU 0 and seed 11 fold 0 on GPU 1.', ''),
        ('04_session_H_reader_s11_f1_f2', {0: [('reader_convnext', [1], 11)], 1: [('reader_convnext', [2], 11)]},
         'ConvNeXt reader, seed 11, folds 1 and 2 (a third reader seed; OOF: two seeds rank-averaged beat one by ~0.005).', ''),
        ('04_session_I_reader_s11_f3_f4', {0: [('reader_convnext', [3], 11)], 1: [('reader_convnext', [4], 11)]},
         'ConvNeXt reader, seed 11, folds 3 and 4 (completes seed 11).', ''),
        # Noisy students (2026-10-08): the ConvNeXt reader on report labels mixed with the OOF
        # predictions of our own readers (07_teacher_targets). Top public competitors report that
        # teacher-to-student pseudo-labels, not larger backbones, close the gap to 0.95+.
        ('04_session_M_student_f0_f1', {0: [('reader_student', [0])], 1: [('reader_student', [1])]},
         'Noisy-student ConvNeXt reader folds 0 and 1 (targets: report consensus mixed with our readers\' OOF predictions, teacher share 0.5 where report readers agree to 0.9 where they split; slot dropout 0.2).', student_note),
        ('04_session_N_student_f2_f3', {0: [('reader_student', [2])], 1: [('reader_student', [3])]},
         'Noisy-student ConvNeXt reader folds 2 and 3.', student_note),
        ('04_session_O_student_f4_s7_f0', {0: [('reader_student', [4])], 1: [('reader_student', [0], 7)]},
         'Noisy-student ConvNeXt reader fold 4 on GPU 0; a second student seed (7) fold 0 on GPU 1.', student_note),
        # Round 2: teachers = our readers + the round-1 students (attach A-F, M, N and O outputs).
        ('04_session_P_student2_f0_f1', {0: [('reader_student2', [0])], 1: [('reader_student2', [1])]},
         'Round-2 noisy student folds 0 and 1: the teacher also averages the round-1 students\' OOF predictions.', student2_note),
        ('04_session_Q_student2_f2_f3', {0: [('reader_student2', [2])], 1: [('reader_student2', [3])]},
         'Round-2 noisy student folds 2 and 3.', student2_note),
        ('04_session_R_student2_f4_s7_f0', {0: [('reader_student2', [4])], 1: [('reader_student2', [0], 7)]},
         'Round-2 noisy student fold 4 on GPU 0; seed 7 fold 0 on GPU 1.', student2_note),
        # ResNet-50 students (2026-10-10): round-1 teacher and recipe on a batch-norm CNN, for
        # ensemble diversity (LB: round 2 0.932 < round 1 0.935 on folds 0-1; more seeds ~+0.001).
        ('04_session_S_student_r50_f0_f1', {0: [('reader_student_r50', [0])], 1: [('reader_student_r50', [1])]},
         'ResNet-50 noisy student folds 0 and 1 (same teacher and recipe as the round-1 ConvNeXt students, session M).', r50_note),
        ('04_session_T_student_r50_f2_f3', {0: [('reader_student_r50', [2])], 1: [('reader_student_r50', [3])]},
         'ResNet-50 noisy student folds 2 and 3.', r50_note),
        ('04_session_U_student_r50_f4_s7_f0', {0: [('reader_student_r50', [4])], 1: [('reader_student_r50', [0], 7)]},
         'ResNet-50 noisy student fold 4 on GPU 0; seed 7 fold 0 on GPU 1.', r50_note),
        # Report-grounded students (2026-10-10): round-1 student + auxiliary regression of each
        # study's report embedding (multilingual E5, PCA 32) from the images.
        # CAFormer students (2026-10-10): the round-1 student on the CAFormer-S18 reader.
        ('04_session_W_student_caf_f0_f1', {0: [('reader_student_caf', [0])], 1: [('reader_student_caf', [1])]},
         'CAFormer-S18 noisy student folds 0 and 1 (round-1 teacher and recipe; ~17 min per epoch).', r50_note),
        ('04_session_X_student_caf_f2_f3', {0: [('reader_student_caf', [2])], 1: [('reader_student_caf', [3])]},
         'CAFormer-S18 noisy student folds 2 and 3.', r50_note),
        ('04_session_Y_student_caf_f4_s7_f0', {0: [('reader_student_caf', [4])], 1: [('reader_student_caf', [0], 7)]},
         'CAFormer-S18 noisy student fold 4 on GPU 0; seed 7 fold 0 on GPU 1.', r50_note),
        # knee-2c-v4 (2026-10-11): the ConvNeXt student on six v3 slots + two anchored 90 mm local slots.
        ('04_session_Z_student_v4_f0_f1', {0: [('reader_student_v4', [0])], 1: [('reader_student_v4', [1])]},
         'ConvNeXt noisy student on the knee-2c-v4 cache (two 90 mm local slots around the header-geometry knee-joint '
         'anchor), folds 0 and 1; same recipe and teacher as session M.', v4_note),
        ('04_session_Z5_student_v5_f0_f1', {0: [('reader_student_v5', [0])], 1: [('reader_student_v5', [1])]},
         'ConvNeXt noisy student on the knee-2c-v5 cache (288 px, 10 windows per series), folds 0 and 1; same recipe and '
         'teacher as session Z (micro-batch 2 x 4).', v5_note),
        # ConvNeXt-Base students on knee-2c-v5 on GPU (Colab Pro bonus hours, 2026-10-14).
        ('04_session_B5_student_cnxb_v5_f0_f1', {0: [('reader_student_cnxb_v5', [0])], 1: [('reader_student_cnxb_v5', [1])]},
         'ConvNeXt-Base noisy student on the knee-2c-v5 cache, folds 0 and 1 (micro-batch 1 x 8, gradient checkpointing).', v5_note),
        # CAFormer students on knee-2c-v5: micro-batch 1 x 8 on a T4, 7 epochs (~35 min/epoch, fits the 5.6 h GPU quota left 2026-10-15).
        # v5 students whose teacher is half our readers, half heliosli's compact model (out-of-fold, 10_compact_oof_s0/s1).
        ('04_session_K5_student_ct_v5_f0_f1', {0: [('reader_student_ct_v5', [0])], 1: [('reader_student_ct_v5', [1])]},
         'ConvNeXt-T noisy student on knee-2c-v5, folds 0 and 1, taught by our readers and heliosli\'s compact model '
         '(out-of-fold). Also attach the knee-compact-oof-s0 and -s1 outputs.', v5_note),
        ('04_session_L5_student_cnxb_ct_v5_f0_f1', {0: [('reader_student_cnxb_ct_v5', [0])], 1: [('reader_student_cnxb_ct_v5', [1])]},
         'ConvNeXt-B noisy student on knee-2c-v5, folds 0 and 1, taught by our readers and heliosli\'s compact model '
         '(out-of-fold). Also attach the knee-compact-oof-s0 and -s1 outputs.', v5_note),
        ('04_session_W5_student_caf_v5_f0_f1', {0: [('reader_student_caf_v5', [0])], 1: [('reader_student_caf_v5', [1])]},
         'CAFormer-S18 noisy student on the knee-2c-v5 cache, folds 0 and 1.', v5_note),
        ('04_session_X5_student_caf_v5_f2_f3', {0: [('reader_student_caf_v5', [2])], 1: [('reader_student_caf_v5', [3])]},
         'CAFormer-S18 noisy student on the knee-2c-v5 cache, folds 2 and 3.', v5_note),
        ('04_session_Y5_student_caf_v5_f4', {0: [('reader_student_caf_v5', [4])], 1: [('reader_student_v5', [0], 7)]},
         'CAFormer-S18 noisy student on knee-2c-v5 fold 4 on GPU 0; the ConvNeXt v5 student, seed 7, fold 0 on GPU 1.', v5_note),
        ('04_session_V_student_rep_f0_f1', {0: [('reader_student_rep', [0])], 1: [('reader_student_rep', [1])]},
         'Report-grounded noisy student folds 0 and 1: the round-1 ConvNeXt student recipe and teacher, plus an auxiliary loss (weight 0.2) regressing each study\'s report embedding from the images (training only; inference unchanged).', rep_note),
        # A second reader backbone (CAFormer-S18) for ensemble diversity; its own family, so
        # FAMILY_WEIGHTS must be re-chosen with 06_oof_blend_check before it is submitted.
        ('04_session_J_caformer_f0_f1', {0: [('reader_caformer', [0])], 1: [('reader_caformer', [1])]},
         'CAFormer-S18 reader folds 0 and 1, one per GPU (same reader recipe, half micro-batch).', ''),
        ('04_session_K_caformer_f2_f3', {0: [('reader_caformer', [2])], 1: [('reader_caformer', [3])]},
         'CAFormer-S18 reader folds 2 and 3, one per GPU.', ''),
        ('04_session_L_caformer_f4_reader_s13_f0', {0: [('reader_caformer', [4])], 1: [('reader_convnext', [0], 13)]},
         'CAFormer-S18 reader fold 4 on GPU 0; ConvNeXt reader seed 13 fold 0 on GPU 1.', ''),
        # The public stacks blend ~20 DINOv2 slot members; ten more of ours (two seeds x five folds).
        # Low priority: on OOF the slot family earns only 0.1 of the family blend.
        ('04_session_G_slot_s7_s11', {0: [('slot_dinov2', [0, 1, 2, 3, 4], 7)], 1: [('slot_dinov2', [0, 1, 2, 3, 4], 11)]},
         'DINOv2 slot model, seed 7 folds 0-4 on GPU 0 and seed 11 folds 0-4 on GPU 1 (ten more slot members).', ''),
    ]
    for name, plan, summary, extra in sessions:
        plan_text = '{\n' + ''.join(f'    {gpu}: {queue!r},\n' for gpu, queue in plan.items()) + '}'
        cells = [
            {'cell_type': 'markdown', 'metadata': {}, 'source': f'# {name}\n\n**Accelerator: GPU T4 x2, Internet ON.** {summary} Each GPU runs its jobs as separate processes with the exact single-GPU recipe; the notebook prints progress every two minutes and fails loudly if any job fails.\n\n{inputs_note}{extra}\n\nOutputs land in one folder per family (`reader_convnext/`, `slot_dinov2/`, `radimagenet/`); attach this whole output to 05_submit.ipynb.'},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': source(template, 2)},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source':
             f'# GPU -> jobs run in order on that GPU: (family preset, folds).\nGPU_JOBS = {plan_text}\n'
             '# Interrupted session: attach its saved output and set this to that output folder;\n'
             '# folds with fold_N.pt and last_fold_N.pt resume at their next epoch.\nRESUME_DIR = ""\n'},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source':
             f'# Training code for each job (generated from the shared sources).\nTRAIN_JOB_SOURCE = {train_job!r}\nRAD_JOB_SOURCE = {rad_job!r}\n'},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': runner},
        ]
        for c in cells:
            c['source'] = c['source'].splitlines(keepends=True)
        write(f'{name}.ipynb', {'cells': cells})

    # CPU input check: the jobs' own data setup, then timed cold reads of each cache shard.
    # Costs no GPU quota and catches mount, label and weight problems before a session.
    rad_hash = next(line for line in (ROOT / 'rad_runtime.py').read_text().splitlines()
                    if line.startswith('RAD_SHA256 = '))
    check = f'''# ---- input check: what a GPU session will find ----------------------------------
import hashlib
{rad_hash}
CHECK_STUDIES = 60
rng = np.random.default_rng(0)
mb_per_s = []
for d in CACHE_DIRS:
    arr = np.load(os.path.join(d, 'cache.npy'), mmap_mode='r')
    rows = rng.choice(arr.shape[0], size=min(CHECK_STUDIES, arr.shape[0]), replace=False)
    started, nbytes, blank = time.time(), 0, 0
    for r in rows:
        study = np.asarray(arr[r]); nbytes += study.nbytes; blank += int(study.max() == 0)
    mb_per_s.append(nbytes / 1e6 / max(time.time() - started, 1e-6))
    print(f'{{d}}: {{len(rows)}} random studies read at {{mb_per_s[-1]:.0f}} MB/s, {{blank}} blank')
epoch_gb = len(df) * np.prod(arr.shape[1:]) / 1e9
print(f'One reader epoch reads {{epoch_gb:.1f}} GB per job: about {{epoch_gb * 1e3 / min(mb_per_s) / 60:.0f}} min '
      'at this speed if nothing is cached (two jobs share the disk).')
weights = attached_files('ResNet50.pt', INPUT_ROOT)
if weights:
    digest = hashlib.sha256(open(weights[0], 'rb').read()).hexdigest()
    assert len(weights) == 1 and digest == RAD_SHA256, f'RadImageNet weights: {{weights}} {{digest}}'
    print('RadImageNet ResNet50.pt verified:', weights[0])
else:
    print('RadImageNet ResNet50.pt not attached (needed by session C only)')
print('\\nINPUT CHECK PASSED: caches, label readers and gold rows are as training expects.')
'''
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': f'# 04_input_check — run before the GPU sessions\n\n**Accelerator: None (CPU), Internet ON.** Attach exactly the inputs of a training session. {inputs_note}{rad_note} It runs the training jobs\' own data setup (cache shards, every label reader, gold rows), times random cache reads, and verifies the RadImageNet weights if attached. It uses no GPU quota; the last line must say INPUT CHECK PASSED.'},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': source(template, 2)},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': 'import os\nos.environ.setdefault("KNEE_ALLOW_CPU", "1")   # data setup only; nothing trains here\n\n' + job_config},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': source(template, 7)},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': check},
    ]
    for c in cells:
        c['source'] = c['source'].splitlines(keepends=True)
    write('04_input_check.ipynb', {'cells': cells})

    sub = read('submission-notebook.ipynb')
    install = '''# Install offline from the attached knee-wheelhouse dataset (the same wheels shard 0 saves).
import subprocess, sys
''' + discover + '''\nwheels = attached_files('timm-1.0.29-*.whl')
assert len(wheels) == 1, f'Attach exactly one wheelhouse (the knee-wheelhouse dataset, not the cache too): {wheels}'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
    '--no-deps', '--find-links', str(wheels[0].parent), 'timm==1.0.29',
    'pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'])
'''
    config = source(sub, 1)
    start = config.index('CKPT_PATHS ='); end = config.index('ckpt_path =', start)
    config = config[:start] + '''# Every attached fold_N.pt is a member. Members are rank-averaged within their family
# (reader_convnext, reader_coatnet, slot_dinov2, radimagenet), then families are averaged.
CKPT_PATHS = checkpoint_inputs()
# A knee-2c-v4 member (if any) defines the decode: v3 members then read its first six slots, which are
# pixel-identical to v3. Stable sort: v4 checkpoints first, the rest in discovery order.
CKPT_PATHS = sorted(CKPT_PATHS, key=lambda p: torch.load(p, map_location='cpu', weights_only=False)['cfg']
                    .get('preprocessing', {}).get('version') != 'knee-2c-v4')
# Family weights are fixed before each scoring run and never chosen on gold. Evidence (2026-10-08):
# OOF pseudo AUC (06_oof_blend_check): reader 0.889, equal blend 0.886, 0.7/0.1/0.2 blend 0.893;
# public LB: 17 models at equal weights 0.917 < two reader folds alone 0.923. Reader only until the
# 0.7/0.1/0.2 blend is measured. Weight 0 skips a family; every attached family must be listed.
FAMILY_WEIGHTS = {'reader_convnext': 1.0, 'slot_dinov2': 0.0, 'radimagenet': 0.0}
''' + config[end:]
    config = '\n'.join(line for line in config.splitlines() if not line.startswith('CKPT_DIR ='))
    put(sub, 1, install + '\n' + config)
    put(sub, 0, '# 05 — Offline submission\n\nGPU, Internet OFF. Attach the competition, the `knee-wheelhouse` dataset (offline packages; do not also attach the cache) and the completed training session outputs. Every attached `fold_N.pt` is a member: ranks are averaged within each family (checkpoint preset), then across families (`FAMILY_WEIGHTS`, fixed before scoring; currently reader only, see the comment there; weight 0 skips a family). A duplicated family-fold stops the run. Run all cells; inspect inference_diagnostics.json and submission.csv. No label inputs are needed here.')
    write('05_submit.ipynb', sub)
    # Noisy-student submission: the reader_student family alone (same preprocessing and inference).
    stu = copy.deepcopy(sub)
    config = ''.join(stu['cells'][1]['source'])
    old_w = "FAMILY_WEIGHTS = {'reader_convnext': 1.0, 'slot_dinov2': 0.0, 'radimagenet': 0.0}"
    assert config.count(old_w) == 1
    config = config.replace(old_w, "FAMILY_WEIGHTS = {'reader_student': 1.0, 'reader_convnext': 0.0, 'reader_caformer': 0.0,\n"
                                   "                  'slot_dinov2': 0.0, 'radimagenet': 0.0}   # students only")
    stu['cells'][1]['source'] = config.splitlines(keepends=True)
    put(stu, 0, '# 05c — Noisy-student submission\n\nGPU, Internet OFF. As 05_submit, but only the `reader_student` family '
                'counts (every other family has weight 0 and is skipped). Attach the competition, `knee-wheelhouse` and the '
                'student session outputs. Compare its score with the teacher readers on the same folds.')
    write('05c_submit_student.ipynb', stu)
    # The same, with transductive self-training on the test set (TTST) before the final prediction.
    tt = copy.deepcopy(stu)
    config = ''.join(tt['cells'][1]['source'])
    config += ("\n# Transductive self-training (inference_runtime.py): each reader member is fine-tuned on the test\n"
               "# volumes toward the ensemble's mean probability, with training augmentations, then predicts again.\n"
               "TTST_EPOCHS = 1\nTTST_LR = 1e-4\nTTST_BATCH = 4\n")
    tt['cells'][1]['source'] = config.splitlines(keepends=True)
    put(tt, 0, '# 05d — Noisy-student submission with test-time self-training\n\nGPU, Internet OFF. As 05c, plus '
               'TTST: after the first pass every reader member is fine-tuned for one epoch on the unlabeled test volumes, '
               'toward the ensemble mean probability, then predicts again (no labels used). Compare with 05c.')
    write('05d_submit_student_ttst.ipynb', tt)
    # Teachers and students together: the ConvNeXt readers and the noisy students at equal weight.
    mix = copy.deepcopy(sub)
    config = ''.join(mix['cells'][1]['source'])
    config = config.replace(old_w, "FAMILY_WEIGHTS = {'reader_convnext': 1.0, 'reader_student': 1.0, 'reader_caformer': 0.0,\n"
                                   "                  'slot_dinov2': 0.0, 'radimagenet': 0.0}   # teachers + students")
    assert "'reader_student': 1.0, 'reader_caformer'" in config
    mix['cells'][1]['source'] = config.splitlines(keepends=True)
    put(mix, 0, '# 05e — Readers and noisy students\n\nGPU, Internet OFF. As 05_submit, with the ConvNeXt readers and the '
                '`reader_student` family rank-averaged at equal family weight. Attach the competition, `knee-wheelhouse`, '
                'the reader sessions and the student sessions.')
    write('05e_submit_readers_students.ipynb', mix)
    # Students of both rounds, no teachers (LB 2026-10-09: readers + students 0.933 < students alone 0.936).
    # Attach only round-2 outputs to score round 2 alone; a listed family with no members is ignored.
    st12 = copy.deepcopy(sub)
    config = ''.join(st12['cells'][1]['source'])
    config = config.replace(old_w, "FAMILY_WEIGHTS = {'reader_student': 1.0, 'reader_student2': 1.0, 'reader_convnext': 0.0,\n"
                                   "                  'reader_caformer': 0.0, 'slot_dinov2': 0.0, 'radimagenet': 0.0}   # students only")
    assert "'reader_student2': 1.0" in config
    st12['cells'][1]['source'] = config.splitlines(keepends=True)
    put(st12, 0, '# 05g — Noisy students, rounds 1 and 2\n\nGPU, Internet OFF. As 05_submit, with the `reader_student` and '
                 '`reader_student2` families at equal weight (teachers skipped). Attach the competition, `knee-wheelhouse` and '
                 'the student session outputs; attach only round-2 outputs to score round 2 alone.')
    write('05g_submit_students_r1r2.ipynb', st12)
    # All student families (ConvNeXt rounds 1-2, ResNet-50, report-grounded) at equal family weight;
    # attach only the outputs to be scored (a listed family with no members is ignored).
    own = copy.deepcopy(sub)
    config = ''.join(own['cells'][1]['source'])
    config = config.replace(old_w, "FAMILY_WEIGHTS = {'reader_student': 1.0, 'reader_student2': 1.0, 'reader_student_r50': 1.0,\n"
                                   "                  'reader_student_rep': 1.0, 'reader_student_caf': 1.0, 'reader_student_cnxb': 1.0,\n"
                                  "                  'reader_student_v4': 1.0, 'reader_student_cnxb_v4': 1.0, 'reader_student_v5': 1.0,\n"
                                  "                  'reader_student_cnxb_v5': 1.0, 'reader_student_caf_v5': 1.0,\n"
                                  "                  'reader_student_ct_v5': 1.0, 'reader_student_cnxb_ct_v5': 1.0,\n"
                                  "                  'reader_convnext': 0.0, 'reader_caformer': 0.0,\n"
                                   "                  'slot_dinov2': 0.0, 'radimagenet': 0.0}   # students only")
    assert "'reader_student_r50': 1.0" in config
    own['cells'][1]['source'] = config.splitlines(keepends=True)
    put(own, 0, '# 05i — Our students, every family\n\nGPU, Internet OFF. As 05_submit, with every student family '
                '(`reader_student`, `reader_student2`, `reader_student_r50`, `reader_student_rep`) at equal family weight and '
                'teachers skipped. Attach the competition, `knee-wheelhouse` and the student outputs to be scored.')
    write('05i_submit_own_students.ipynb', own)

    # Hybrid submission: the public Raptor CoAtNet checkpoint (training recipe unpublished, so
    # its weights may be used under the project policy) rank-blended with our own ensemble.
    raptor_src = (ROOT / 'third_party' / 'raptor_swa_inference.py').read_text()
    # LB 2026-10-08: Raptor alone 0.949 (public); x0.5 with our equal-family 17-model ensemble (0.917) gave
    # 0.940. The reader-only ensemble is blended at smaller weights.
    for base, own_weight, fname in [(sub, 0.2, '05b_submit_raptor_blend_w20.ipynb'),
                                    (sub, 0.3, '05b_submit_raptor_blend_w30.ipynb'),
                                    (mix, 0.3, '05f_submit_raptor_mix_w30.ipynb'),
                                    (st12, 0.3, '05h_submit_raptor_students_w30.ipynb'),
                                    (own, 0.3, '05j_submit_raptor_own_students_w30.ipynb')]:
        hyb = copy.deepcopy(base)
        cells = hyb['cells']
        put(hyb, 0, f'# 05b — Hybrid submission: public Raptor CoAtNet + our ensemble (our weight {own_weight})\n\n'
            'GPU T4 x2, Internet OFF. Attach the competition, `knee-wheelhouse`, the training session outputs and the public '
            'dataset `nartaa/rsna-knee-publication-swa-weights-20261007` (checkpoint verified by SHA-256). First the public '
            'Raptor inference (third_party/raptor_swa_inference.py, unmodified except for its output path) runs in its own '
            'process on Kaggle\'s default packages; then our ensemble runs as in 05_submit; finally both are rank-blended per '
            f'finding with our ensemble at weight {own_weight}. If the Raptor member fails, the submission is our ensemble alone.')
        raptor_cell = (
            "# 1. Public Raptor CoAtNet member first, in its own process, before any package is installed here.\n"
            "import os, sys, subprocess, time\n"
            "NOTEBOOK_START = time.time()\n"
            f"OWN_WEIGHT = {own_weight}   # rank weight of our ensemble in the final blend, fixed before scoring\n"
            "RAPTOR_OUT = '/kaggle/working/raptor_submission.csv'\n"
            f"RAPTOR_SOURCE = {raptor_src!r}\n"
            "with open('/kaggle/working/raptor_infer.py', 'w') as handle: handle.write(RAPTOR_SOURCE)\n"
            "try:\n"
            "    subprocess.run([sys.executable, '/kaggle/working/raptor_infer.py'], check=True, timeout=5 * 3600,\n"
            "                   env=dict(os.environ, RAPTOR_OUT=RAPTOR_OUT))\n"
            "except Exception as exc:\n"
            "    print('Raptor member failed; the submission falls back to our ensemble alone:', type(exc).__name__, exc)\n"
            "RAPTOR_HOURS = (time.time() - NOTEBOOK_START) / 3600\n"
            "print(f'Raptor member finished after {RAPTOR_HOURS:.2f} h')\n")
        cells.insert(1, {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                         'source': raptor_cell.splitlines(keepends=True)})
        config = ''.join(cells[2]['source'])
        for old, new in [('OUT_CSV  = "/kaggle/working/submission.csv"', 'OUT_CSV  = "/kaggle/working/own_submission.csv"   # blended into submission.csv below'),
                         ('MAX_HOURS    = 8.0', 'MAX_HOURS    = max(1.0, 8.0 - RAPTOR_HOURS - 0.3)   # the 9 h limit covers both members')]:
            assert config.count(old) == 1, old
            config = config.replace(old, new)
        cells[2]['source'] = config.splitlines(keepends=True)
        blend = (
            "# 3. Rank blend per finding: (1 - OWN_WEIGHT) x Raptor + OWN_WEIGHT x our ensemble.\n"
            "import numpy as np, pandas as pd\n"
            "own = pd.read_csv(OUT_CSV, dtype={'StudyInstanceUID': str})\n"
            "labels = [c for c in own.columns if c != 'StudyInstanceUID']\n"
            "final, used = own.copy(), 'our ensemble alone'\n"
            "if os.path.exists(RAPTOR_OUT):\n"
            "    rap = pd.read_csv(RAPTOR_OUT, dtype={'StudyInstanceUID': str})\n"
            "    good = (set(rap.StudyInstanceUID) == set(own.StudyInstanceUID) and len(rap) == len(own)\n"
            "            and list(rap.columns) == list(own.columns) and np.isfinite(rap[labels].to_numpy(float)).all())\n"
            "    if good:\n"
            "        rap = rap.set_index('StudyInstanceUID').loc[own.StudyInstanceUID].reset_index()\n"
            "        for c in labels:\n"
            "            final[c] = (1 - OWN_WEIGHT) * rap[c].rank(pct=True).to_numpy() + OWN_WEIGHT * own[c].rank(pct=True).to_numpy()\n"
            "        used = f'Raptor x {1 - OWN_WEIGHT:.2f} + our ensemble x {OWN_WEIGHT:.2f}'\n"
            "    else:\n"
            "        print('Raptor output does not match the test rows or columns; using our ensemble alone')\n"
            "assert final[labels].notna().all().all() and len(final) == len(own)\n"
            "final.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "print('submission.csv:', used, final.shape)\n")
        cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                      'source': blend.splitlines(keepends=True)})
        write(fname, hyb)
    # The same own-students submission with test-time window subsets: the readers train on 6 random windows per
    # series and score all of them (10 on v5); each member also averages 4 passes over random 6-window subsets.
    tta = copy.deepcopy(own)
    config = ''.join(tta['cells'][1]['source'])
    config += ("\n# Test-time window subsets (inference_runtime.py): each prediction is half the all-window pass and half\n"
               "# the mean of WINDOW_TTA passes over random subsets of the training window count per series.\n"
               "WINDOW_TTA = 4\n")
    tta['cells'][1]['source'] = config.splitlines(keepends=True)
    put(tta, 0, '# 05l — Own students with test-time window subsets\n\nGPU, Internet OFF. As 05i, plus: each member\'s '
                'prediction is half its all-window pass and half the mean of 4 passes over random 6-window subsets per series, '
                'the window count it trained on. Attach the same inputs as 05i and compare with it.')
    write('05l_submit_own_students_tta.ipynb', tta)
    # Hybrid on the public 0.954 stack (user decision 2026-10-11: third-party weights are allowed when we can
    # explain the model; METHOD_NOTES documents heliosli's compact model). Stages 1-3 are heliosli's public
    # notebook rsna-knee-blend-gold-dataset-restored (Raptor CoAtNet x0.7 + compact ConvNeXt-Small x0.3), run
    # unchanged except DEADLINE_HOURS 8.3 -> 6.0 so our stage fits the 9 h limit; its submission is kept as
    # public_submission.csv; then our student ensemble (05i) runs and both are rank-blended per finding.
    public_nb = json.loads((ROOT / 'third_party' / 'heliosli_blend_gold' /
                            'rsna-knee-blend-gold-dataset-restored.ipynb').read_text())
    public_cells = [copy.deepcopy(c) for c in public_nb['cells'][2:7]]
    for c in public_cells:
        c['outputs'], c['execution_count'] = [], None
    src4 = ''.join(public_cells[2]['source'])
    assert src4.count('DEADLINE_HOURS = 8.3') == 1
    public_cells[2]['source'] = src4.replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 6.0   # [hybrid] was 8.3: leaves time for our stage').splitlines(keepends=True)
    for own_weight, fname in [(0.2, '05k_submit_public954_own_students_w20.ipynb'), (0.3, '05k_submit_public954_own_students_w30.ipynb')]:
        hyb = copy.deepcopy(own)
        cells = hyb['cells']
        put(hyb, 0, f'# 05k — Hybrid: public 0.954 stack + our students (our weight {own_weight})\n\n'
            'GPU T4 x2, Internet OFF. Attach the competition, `knee-wheelhouse`, our student session outputs, and the public '
            'datasets `heliosli/rsna-knee-compact-v1`, `nartaa/rsna-knee-publication-swa-weights-20261007`, '
            '`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Stages 1-3 are heliosli\'s public notebook '
            '(Raptor CoAtNet x0.7 + heliosli compact ConvNeXt-Small x0.3, public LB 0.954), unchanged except its compact '
            'stage deadline (8.3 h -> 6.0 h). Then our students run as in 05i and the two submissions are rank-blended per '
            f'finding with ours at weight {own_weight}. Architecture of the public members: METHOD_NOTES.md.')
        keep = {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': (
            "# Keep the public stack's submission; our stage writes its own file, the final blend writes submission.csv.\n"
            "import os, time, shutil\n"
            "shutil.move('/kaggle/working/submission.csv', '/kaggle/working/public_submission.csv')\n"
            "PUBLIC_HOURS = (time.time() - NB_T0) / 3600\n"
            "print(f'public stack finished after {PUBLIC_HOURS:.2f} h')\n").splitlines(keepends=True)}
        cells[1:1] = public_cells + [keep]
        ci = 1 + len(public_cells) + 1
        config = ''.join(cells[ci]['source'])
        for old_, new_ in [('OUT_CSV  = "/kaggle/working/submission.csv"', 'OUT_CSV  = "/kaggle/working/own_submission.csv"   # blended into submission.csv below'),
                           ('MAX_HOURS    = 8.0', 'MAX_HOURS    = max(1.0, 8.6 - PUBLIC_HOURS)   # the 9 h limit covers both stages')]:
            assert config.count(old_) == 1, old_
            config = config.replace(old_, new_)
        cells[ci]['source'] = config.splitlines(keepends=True)
        blend = (
            f"# Final: per-finding rank blend, public stack x{1 - own_weight:.1f} + our students x{own_weight} (fixed before scoring).\n"
            "import numpy as np, pandas as pd, json\n"
            f"OWN_WEIGHT = {own_weight}\n"
            "pub = pd.read_csv('/kaggle/working/public_submission.csv', dtype={'StudyInstanceUID': str})\n"
            "own = pd.read_csv('/kaggle/working/own_submission.csv', dtype={'StudyInstanceUID': str})\n"
            "labels = [c for c in pub.columns if c != 'StudyInstanceUID']\n"
            "assert pub['StudyInstanceUID'].tolist() == own.set_index('StudyInstanceUID').loc[pub['StudyInstanceUID']].index.tolist()\n"
            "own = own.set_index('StudyInstanceUID').loc[pub['StudyInstanceUID'], labels].reset_index()\n"
            "rank = lambda v: pd.Series(v).rank(method='average', pct=True).to_numpy()\n"
            "final = pub.copy()\n"
            "for c in labels:\n"
            "    final[c] = (1 - OWN_WEIGHT) * rank(pub[c].to_numpy()) + OWN_WEIGHT * rank(own[c].to_numpy())\n"
            "assert final[labels].notna().all().all() and len(final) == len(pub)\n"
            "final.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "print('submission.csv', final.shape, json.dumps({'own_weight': OWN_WEIGHT, 'public_hours': round(PUBLIC_HOURS, 2)}))\n")
        cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                      'source': blend.splitlines(keepends=True)})
        write(fname, hyb)
    # Hybrid on the public 0.9586 notebook (heliosli, 2026-10-17: Raptor 384 + 224, R6 compact with flip TTA, goodpjw2008's
    # 2.5D reader as a small leg), run unchanged; then our v5 students as one more leg in goodpjw2008's pattern: skipped when
    # the public stages leave too little of the 9 h limit, bounded by MAX_HOURS, and dropped (public submission kept) if
    # it fails or leaves more than 1 % of studies undecoded; studies it did not decode keep the public rank.
    pub9586 = json.loads((ROOT / 'third_party' / 'public_oct19' / 'heliosli_this-is-best-public-dataset-new-version-0-9586' /
                          'this-is-best-public-dataset-new-version-0-9586.ipynb').read_text())
    pub9586_cells = [copy.deepcopy(c) for c in pub9586['cells'] if c['cell_type'] == 'code']
    for c in pub9586_cells:
        c['outputs'], c['execution_count'] = [], None
    # heliosli's 'frozen' R6 dataset now holds R7 (bundle 8c7b666e), so the notebook fails as published (2026-10-19).
    # Mount rostigor/rsna-knee-r6-fe055406 instead: its manifest is byte-identical to the pinned R6 one (sha256 below),
    # and the notebook already checks every model file against that manifest.
    R6_MANIFEST_SHA = '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
    setup = next(c for c in pub9586_cells if "C_SLUG = 'rsna-knee-r6-frozen-fe055406'" in ''.join(c['source']))
    src = ''.join(setup['source'])
    for old_, new_ in [("C_SLUG = 'rsna-knee-r6-frozen-fe055406'", "C_SLUG = 'rsna-knee-r6-fe055406'   # [ours] rostigor's copy of R6"),
                       ("manifest = json.loads((M_SRC / 'manifest.json').read_text())",
                        f"assert hashlib.sha256((M_SRC / 'manifest.json').read_bytes()).hexdigest() == '{R6_MANIFEST_SHA}', 'R6 manifest differs'   # [ours]\n"
                        "manifest = json.loads((M_SRC / 'manifest.json').read_text())")]:
        assert src.count(old_) == 1, old_
        src = src.replace(old_, new_)
    setup['source'] = src.splitlines(keepends=True)
    for own_weight, fname in [(0.05, '05m_submit_public9586_own_w05.ipynb'), (0.1, '05m_submit_public9586_own_w10.ipynb'),
                              (0.2, '05m_submit_public9586_own_w20.ipynb')]:
        hyb = copy.deepcopy(own)
        cells = hyb['cells']
        put(hyb, 0, f'# 05m — Hybrid: public 0.9586 notebook + our v5 students (our weight {own_weight})\n\n'
            'GPU T4 x2, Internet OFF. Attach the competition, `knee-wheelhouse`, our v5 student session outputs, and the '
            'public 0.9586 inputs (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, `rostigor/rsna-knee-r6-fe055406` (R6; the heliosli frozen copy now holds R7), '
            '`nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, '
            'kernel `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`). The public notebook runs unchanged; our students '
            f'then run if time allows and are rank-blended per finding at weight {own_weight}, else the public submission stands.')
        keep = {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': (
            "# Keep the public notebook's submission; our leg writes its own file, the final blend writes submission.csv.\n"
            "import os, time, shutil\n"
            "shutil.copy('/kaggle/working/submission.csv', '/kaggle/working/public_submission.csv')\n"
            "PUBLIC_HOURS = (time.time() - NB_T0) / 3600\n"
            "OWN_LEG = PUBLIC_HOURS < 7.6   # our leg needs time to decode the test set; otherwise the public submission stands\n"
            "print(f'public notebook finished after {PUBLIC_HOURS:.2f} h; own leg', 'runs' if OWN_LEG else 'skipped')\n").splitlines(keepends=True)}
        cells[1:1] = pub9586_cells + [keep]
        ci = 1 + len(pub9586_cells) + 1
        config = ''.join(cells[ci]['source'])
        for old_, new_ in [('OUT_CSV  = "/kaggle/working/submission.csv"', 'OUT_CSV  = "/kaggle/working/own_submission.csv"   # blended into submission.csv below'),
                           ('MAX_HOURS    = 8.0', 'MAX_HOURS    = max(0.3, 8.5 - PUBLIC_HOURS)   # the 9 h limit covers every stage')]:
            assert config.count(old_) == 1, old_
            config = config.replace(old_, new_)
        cells[ci]['source'] = config.splitlines(keepends=True)
        # Our leg's code cells run only when OWN_LEG; each is wrapped so a failure keeps the public submission.
        for c in cells[ci:]:
            if c['cell_type'] != 'code':
                continue
            body = ''.join(c['source'])
            c['source'] = ("if OWN_LEG:\n    try:\n        exec(compile(" + repr(body) + ", '<own leg>', 'exec'), globals())\n"
                           "    except Exception as _own_exc:\n        OWN_LEG = False\n"
                           "        print('own leg FAILED, public submission kept:', repr(_own_exc)[:300], flush=True)\n").splitlines(keepends=True)
        blend = (
            f"# Final: per-finding rank blend, public 0.9586 x{1 - own_weight:.2f} + our students x{own_weight} (fixed before scoring).\n"
            "import numpy as np, pandas as pd, json, os, shutil\n"
            f"OWN_WEIGHT = {own_weight}\n"
            "pub = pd.read_csv('/kaggle/working/public_submission.csv', dtype={'StudyInstanceUID': str})\n"
            "labels = [c for c in pub.columns if c != 'StudyInstanceUID']\n"
            "rank = lambda v: pd.Series(v).rank(method='average', pct=True).to_numpy()\n"
            "used = 'public only'\n"
            "try:\n"
            "    assert OWN_LEG, 'own leg skipped or failed'\n"
            "    diag = json.load(open('/kaggle/working/inference_diagnostics.json'))\n"
            "    blank = set(diag.get('blank_study_ids', []))\n"
            "    assert diag.get('ensemble_members', 0) > 0, 'no own member predicted'\n"
            "    assert len(blank) <= 0.01 * len(pub), f'{len(blank)} studies undecoded by our leg'\n"
            "    own = pd.read_csv('/kaggle/working/own_submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[pub['StudyInstanceUID'], labels]\n"
            "    ok = ~pub['StudyInstanceUID'].isin(blank).to_numpy()\n"
            "    final = pub.copy()\n"
            "    for c in labels:\n"
            "        r_pub = rank(pub[c].to_numpy())\n"
            "        r_own = np.full(len(pub), np.nan)\n"
            "        r_own[ok] = rank(own[c].to_numpy()[ok])\n"
            "        final[c] = np.where(ok, (1 - OWN_WEIGHT) * r_pub + OWN_WEIGHT * np.nan_to_num(r_own), r_pub)\n"
            "    assert final[labels].notna().all().all() and len(final) == len(pub)\n"
            "    final.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "    used = f'public x{1 - OWN_WEIGHT:.2f} + own x{OWN_WEIGHT} ({len(blank)} undecoded studies keep the public rank)'\n"
            "except Exception as exc:\n"
            "    shutil.copy('/kaggle/working/public_submission.csv', '/kaggle/working/submission.csv')\n"
            "    print('blend skipped, public submission kept:', repr(exc)[:300])\n"
            "print('submission.csv:', used)\n")
        cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                      'source': blend.splitlines(keepends=True)})
        write(fname, hyb)
    # Out-of-fold predictions of heliosli's compact model on the training studies, as a stronger teacher for our
    # students (user decision 2026-10-11 allows its weights; METHOD_NOTES item 9 documents the model). Its setup cell is
    # heliosli's stage-2 cell unchanged except the deadline; scripts/compact_oof_cell.py runs compact_infer.py on --split train.
    oof_setup = copy.deepcopy(public_nb['cells'][4])
    oof_setup['outputs'], oof_setup['execution_count'] = [], None
    oof_src = ''.join(oof_setup['source'])
    oof_cell = (ROOT / 'scripts' / 'compact_oof_cell.py').read_text()
    for fname, shard, shards, limit, hours in [('10_compact_oof_probe.ipynb', 0, 1, 48, 1.0),
                                               ('10_compact_oof_s0.ipynb', 0, 2, 0, 11.0),
                                               ('10_compact_oof_s1.ipynb', 1, 2, 0, 11.0)]:
        setup = copy.deepcopy(oof_setup)
        setup['source'] = oof_src.replace('DEADLINE_HOURS = 8.3', f'DEADLINE_HOURS = {hours}   # [oof] was 8.3').splitlines(keepends=True)
        what = f'the first {limit} training studies (timing and pipeline probe)' if limit else f'training-study shard {shard} of {shards}'
        write(fname, {'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                f'# 10 — heliosli compact model: out-of-fold predictions on {what}\n\n',
                'GPU T4 x2, Internet OFF. Attach the competition, `heliosli/rsna-knee-compact-v1` and '
                '`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Each training study is scored by the one compact fold '
                'model that never trained on it (fold = sha1(StudyInstanceUID) mod 4, as in compact_train.py). Outputs '
                '`reader_compact/oof_fold_<k>.csv` for the student teacher and `compact_logits.csv`.']},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [
                'import time\n', 'NB_T0 = time.time()\n',
                f'COMPACT_SHARD, COMPACT_SHARDS, COMPACT_LIMIT = {shard}, {shards}, {limit}\n']},
            setup,
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
             'source': oof_cell.splitlines(keepends=True)}],
            'metadata': copy.deepcopy(public_nb.get('metadata', {})), 'nbformat': 4, 'nbformat_minor': 5})
    # CPU check of family blend weights on out-of-fold predictions (pseudo targets; never gold).
    check = (ROOT / 'scripts' / 'oof_blend_check.py').read_text()
    write('06_oof_blend_check.ipynb', {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# 06 OOF blend check\n\n**Accelerator: None (CPU). Internet: OFF is fine.** Attach the session outputs (A, B, C, D, ...). '
            'Scores family rank-blend weights on out-of-fold predictions against the pseudo (report-label) targets. Gold never '
            'enters: gold studies train in every fold and have no OOF rows. Source: `scripts/oof_blend_check.py`.']},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': check.splitlines(keepends=True)}]})
    # TPU sessions (v5e-8): jobs run one after another across all chips (tpu_runtime.py), same
    # recipes and output folders as the GPU sessions.
    tpu_job = '\n\n'.join(job_cells + [(ROOT / 'tpu_runtime.py').read_text()])
    tpu_runner = (ROOT / 'tpu_session_runner.py').read_text()
    # Hours per session when less than a full 9 h session of TPU quota is left (quota resets 2026-10-17).
    TPU_BUDGETS = {'08_tpu_student_cnxb_v4_f01': 4.5, '08_tpu_student_v5_f01': 4.5, '08_tpu_student_v5_f234': 4.6}
    tpu_sessions = [
        # (name, jobs, summary, CPU rehearsal?). TPU time is not a blocker: long or large-model
        # experiments go here and wait in Kaggle's TPU queue while GPU sessions continue.
        ('08_tpu_student_cnxb', [('reader_student_cnxb', [0, 1, 2, 3, 4])],
         'ConvNeXt-Base (ImageNet-22k) noisy student, folds 0-4: the round-1 student recipe and teacher on a backbone '
         'about three times the compute of ConvNeXt-T. A time guard skips folds that would not finish within 8.3 h.', False),
        ('08_tpu_student_v4', [('reader_student_v4', [2, 3, 4]), ('reader_student_v4', [0, 1, 2, 3, 4], 7)],
         'ConvNeXt noisy student on the knee-2c-v4 cache (session Z\'s recipe): folds 2-4 (Z trains 0-1 on GPU), then a '
         'second seed (7) on all five folds. The time guard skips folds that would not finish within 8.3 h.', False),
        ('08_tpu_student_cnxb_v4', [('reader_student_cnxb_v4', [0, 1, 2, 3, 4])],
         'ConvNeXt-Base noisy student on the knee-2c-v4 cache, folds 0-4 (~7 min/epoch on v5e-8; the time guard keeps the '
         'folds that fit in 8.3 h).', False),
        ('08_tpu_student_cnxb_v4_f01', [('reader_student_cnxb_v4', [0, 1])],
         'ConvNeXt-Base noisy student on the knee-2c-v4 cache, folds 0-1 only, under a 4.5 h budget: the TPU quota left '
         'before the 2026-10-17 reset (an LB comparison with the ConvNeXt-T v4 students, 0.938 on folds 0-1).', False),
        ('08_tpu_student_v5_f01', [('reader_student_v5', [0, 1])],
         'ConvNeXt noisy student on the knee-2c-v5 cache (288 px, 10 windows per series), folds 0-1 under a 4.5 h budget: '
         'an LB comparison with the v4 students (0.938 on folds 0-1).', False),
        ('08_tpu_student_v5_f234', [('reader_student_v5', [2, 3, 4])],
         'ConvNeXt noisy student on the knee-2c-v5 cache, folds 2-4 (session Z5 trained 0-1 on GPU), under a 4.6 h budget: '
         'the TPU quota left before the 2026-10-17 reset.', False),
        ('08_tpu_student_ct_v5_f234', [('reader_student_ct_v5', [2, 3, 4])],
         'ConvNeXt-T v5 student taught by our readers + heliosli compact OOF (K5 trained 0-1 on GPU), folds 2-4. '
         'Also attach the knee-compact-oof-s0 and -s1 outputs.', False),
        ('08_tpu_student_cnxb_ct_v5_f234', [('reader_student_cnxb_ct_v5', [2, 3, 4])],
         'ConvNeXt-B v5 student taught by our readers + heliosli compact OOF (L5 trained 0-1 on GPU), folds 2-4 (the '
         'time guard keeps the folds that fit in 8.3 h). Also attach the knee-compact-oof-s0 and -s1 outputs.', False),
        ('08_tpu_student_cnxb_v5_f234', [('reader_student_cnxb_v5', [2, 3, 4])],
         'ConvNeXt-Base noisy student on the knee-2c-v5 cache, folds 2-4 (session B5 trained 0-1 on GPU; the time guard '
         'keeps the folds that fit in 8.3 h).', False),
        ('09_tpu_preflight_cpu', [('reader_student_cnxb', [0])],
         'CPU rehearsal of 08_tpu_student_cnxb: installs torch 2.8.0 + torch_xla 2.8.0 (CPU), then runs the same TPU job '
         'code with PJRT_DEVICE=CPU on 40 studies (5 steps) for 3 epochs with ConvNeXt-Atto. Checks the SPMD calls, fixed shapes, '
         'the number of compiled graphs, checkpoints and outputs before a TPU queue slot is spent.', True),
    ]
    tpu_install = ('# Training requires Internet for pretrained weights; torch_xla ships with the TPU image.\n'
                   'import importlib.metadata, subprocess, sys\n'
                   "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'iterative-stratification==0.1.9'])\n"
                   "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'timm==1.0.29'])\n"
                   "# Versions only: importing torch_xla here could claim the TPU before the job process does.\n"
                   "print({p: importlib.metadata.version(p) for p in ('torch', 'torch_xla')})\n")
    cpu_install = ('# CPU rehearsal: the TPU image\'s torch / torch_xla versions, CPU builds, run on the PJRT CPU client.\n'
                   'import importlib.metadata, os, subprocess, sys\n'
                   "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.8.0', 'torchvision==0.23.0',\n"
                   "                       '--index-url', 'https://download.pytorch.org/whl/cpu'])\n"
                   "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch_xla==2.8.0',\n"
                   "                       'iterative-stratification==0.1.9'])\n"
                   "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'timm==1.0.29'])\n"
                   "print({p: importlib.metadata.version(p) for p in ('torch', 'torch_xla', 'timm')})\n"
                   "os.environ.update(PJRT_DEVICE='CPU', CPU_NUM_DEVICES='1', KNEE_TPU_PREFLIGHT='40', KNEE_EPOCHS='3',\n"
                   "                  KNEE_BACKBONE='convnext_atto.d2_in1k')\n")
    for name, jobs, summary, rehearsal in tpu_sessions:
        jobs_text = '[\n' + ''.join(f'    {job!r},\n' for job in jobs) + ']'
        accel = '**Accelerator: None (CPU), Internet ON.**' if rehearsal else '**Accelerator: TPU v5e-8, Internet ON.**'
        cells = [
            {'cell_type': 'markdown', 'metadata': {}, 'source': f'# {name}\n\n{accel} {summary} One process drives every '
             'chip (torch_xla SPMD): each step\'s 8 studies (the GPU recipe\'s 4 x 2) are split across the chips; fixed tensor '
             'shapes; the per-series jitter runs in the CPU loader. Same checkpoints and output folders as the GPU sessions.'
             f'\n\n{inputs_note}{v5_note if "v5" in name else v4_note if "v4" in name else r50_note}'},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
             'source': cpu_install if rehearsal else tpu_install},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source':
             f'# Jobs run in order, each on every TPU chip: (preset, folds) or (preset, folds, seed).\nTPU_JOBS = {jobs_text}\n'
             + (f"import os\nos.environ['KNEE_TIME_BUDGET_H'] = '{TPU_BUDGETS[name]}'   # folds that would not finish in time are skipped\n"
                if name in TPU_BUDGETS else '')},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source':
             f'# Training code for each job (generated from the shared sources).\nTPU_JOB_SOURCE = {tpu_job!r}\n'},
            {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': tpu_runner},
        ]
        for c in cells:
            if isinstance(c['source'], str): c['source'] = c['source'].splitlines(keepends=True)
        write(f'{name}.ipynb', {'cells': cells})
    # CPU: teacher targets for the noisy-student sessions, from our readers' out-of-fold predictions.
    teacher = (ROOT / 'scripts' / 'teacher_targets.py').read_text()
    write('07_teacher_targets.ipynb', {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# 07 Teacher targets\n\n**Accelerator: None (CPU). Internet: OFF is fine.** Attach the reader session outputs '
            '(A, B, C, D, E, F, and any CAFormer sessions). Writes `teacher_targets.csv`: per study, the mean out-of-fold '
            'prediction of our own readers (seeds averaged, then families). Gold studies have no OOF rows and get no teacher '
            'value. Attach this output to the noisy-student sessions. Source: `scripts/teacher_targets.py`.']},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': ('# Shared label code (label_builder.py), including build_teacher.\n'
                    + (ROOT / 'label_builder.py').read_text()).splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': teacher.splitlines(keepends=True)}]})
    # CPU: the v4 joint anchor on 60 real studies (offsets, fallback rate, montages) before the v4 cache build.
    write('03_anchor_check.ipynb', {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# 03 Anchor check (knee-2c-v4)\n\n**Accelerator: None (CPU). Internet: OFF is fine.** Attach the competition. '
            'Builds the v4 manifest for 60 training studies, reports how often the local slots are centred on the '
            'header-geometry joint anchor and how far it lies from the image centre, and saves montages of the full and '
            'local sagittal/coronal slots. Reads no labels. Source: `scripts/anchor_check.py`.']},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': ('# Shared preprocessing (preprocessing.py).\n' + (ROOT / 'preprocessing.py').read_text()).splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': (ROOT / 'scripts' / 'anchor_check.py').read_text().splitlines(keepends=True)}]})
    write('03b_laterality_check.ipynb', {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# 03b Laterality check\n\n**Accelerator: None (CPU). Internet: OFF is fine.** Attach the competition. '
            'For 600 training studies: how often the coordinate side fallback (first file of each series) and a stack-centre '
            'variant agree with the laterality tags where tags exist, and with each other where they do not. Reads no '
            'labels. Source: `scripts/laterality_check.py`.']},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': ('# Shared preprocessing (preprocessing.py).\n' + (ROOT / 'preprocessing.py').read_text()).splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
         'source': (ROOT / 'scripts' / 'laterality_check.py').read_text().splitlines(keepends=True)}]})
    import build_run_folder
    build_run_folder.build()          # KAGGLE_RUN/: the same notebooks, grouped in run order


if __name__ == '__main__':
    main()
