"""TPU (torch_xla SPMD) training runtime for reader25d presets; embedded in TPU session jobs.

Same recipe as train_runtime.py: folds, targets, optimiser, one-cycle schedule, EMA, epoch
selection on pseudo-validation BCE, checkpoint format. Differences that XLA requires:
- one process drives every TPU chip (SPMD): each optimiser step's 8 studies (the GPU recipe's
  4 x 2 accumulation) are split across the chips along the batch dimension and the compiler
  inserts the gradient all-reduce. A process per chip (xmp.spawn) failed on Kaggle's TPU VM
  ("Expected 8 worker addresses, got 1"): preset TPU topology variables conflict with it;
- fixed tensor shapes: the reader encodes a fixed number of windows for every slot and masks
  absent ones (Reader25D.static_shapes); every batch holds exactly 8 studies (training drops
  the remainder, prediction pads with the last study and discards the padding);
- the per-series jitter runs in the CPU data loader (augment_series_cpu), not on the device;
- bf16 autocast instead of fp16 with a gradient scaler;
- a time guard: a fold that would not finish within KNEE_TIME_BUDGET_H of the session start
  is skipped (Kaggle stops sessions at 9 h), so finished folds are kept.
KNEE_TPU_PREFLIGHT=N trains and validates on N studies per fold for KNEE_EPOCHS epochs: a CPU
rehearsal of the same code (PJRT_DEVICE=CPU). Without torch_xla (local tests) the loop runs on CPU.
"""
import contextlib, functools
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    import torch_xla.distributed.parallel_loader as xpl
    XLA = True
except ImportError:
    XLA = False

assert ARCH == 'reader25d', 'The TPU runtime trains reader25d presets'
assert not RESUME, 'Resume is not supported on TPU; start the fold again'
assert not globals().get('REPORT_AUX', 0.0), 'The report auxiliary loss is implemented for GPU sessions only'
PREFLIGHT = int(os.environ.get('KNEE_TPU_PREFLIGHT', 0))
if PREFLIGHT:
    EPOCHS = int(os.environ.get('KNEE_EPOCHS', 2))
SESSION_START = float(os.environ.get('KNEE_SESSION_START', time.time()))
TIME_BUDGET_H = float(os.environ.get('KNEE_TIME_BUDGET_H', 8.3))
os.makedirs(OUT, exist_ok=True)
folds = assign_folds(df, LABEL_COLS, N_FOLDS, SEED, GROUP_COL if GROUP_COL in df else None)
fold_table = df[['StudyInstanceUID', 'source']].assign(fold=folds)
import hashlib, importlib.metadata
identity = hashlib.sha256((df.to_json(orient='split') + json.dumps(CACHE_META, sort_keys=True)
                           + fold_table.to_json()).encode()).hexdigest()
# Identical keys and values to train_runtime.recipe, plus where it was trained.
recipe = dict(arch=ARCH, backbone=BACKBONE, img=IMG, n_slices=N_SLICES, planes=PLANES,
              label_cols=LABEL_COLS, t_train=T_TRAIN, preprocessing=CACHE_META,
              unfreeze_last=UNFREEZE_LAST, head_hidden=HEAD_HIDDEN, pct_start=PCT_START,
              normalization=NORMALIZATION, drop=DROPOUT, seed=SEED,
              epochs=EPOCHS, lr=LR, weight_decay=WD, n_folds=N_FOLDS, accum_steps=ACCUM_STEPS,
              encoder_lr_scale=ENCODER_LR_SCALE, batch=BATCH, identity=identity,
              gold_weight=GOLD_WEIGHT, ema=EMA_DECAY, clip=CLIP, train_windows=TRAIN_WINDOWS, slot_dropout=SLOT_DROPOUT,
              preset=PRESET, grad_checkpoint=GRAD_CHECKPOINT, teacher=TEACHER_SUMMARY,
              group_col=GROUP_COL if GROUP_COL in df else None,
              aux_dim=0, report_aux=None,
              versions={p: importlib.metadata.version(p) for p in ('torch','timm','numpy')},
              trained_on='xla' if XLA else 'cpu')
STUDIES_PER_STEP = BATCH * ACCUM_STEPS         # the GPU recipe's effective batch, one step here
TPU_WORKERS = int(os.environ.get('KNEE_TPU_WORKERS', 4))
BATCH_KEYS = ('x', 'present', 'y', 'w')


class _Backend:
    """SPMD over every local XLA device, with a single-process CPU stand-in for tests."""
    def __init__(self):
        if XLA:
            xr.use_spmd()                              # before any XLA tensor exists
            self.n_devices = xr.global_runtime_device_count()
            self.mesh = xs.Mesh(np.arange(self.n_devices), (self.n_devices, 1), ('data', 'model'))
            self.device = xm.xla_device()
            assert STUDIES_PER_STEP % self.n_devices == 0, (STUDIES_PER_STEP, self.n_devices)
        else:
            self.n_devices, self.mesh, self.device = 1, None, torch.device('cpu')

    def autocast(self):
        return torch.autocast('xla', dtype=torch.bfloat16) if XLA else contextlib.nullcontext()

    def step(self):
        if XLA: xm.mark_step()

    def loader(self, data_loader):
        """Batches as (x, present, y, w); on XLA each is split across the devices along dim 0."""
        if XLA:
            ranks = dict(x=5, present=2, y=2, w=2)
            sharding = {k: xs.ShardingSpec(self.mesh, ('data',) + (None,) * (r - 1)) for k, r in ranks.items()}
            data_loader = xpl.MpDeviceLoader(data_loader, self.device, input_sharding=sharding)
        return (tuple(batch[k] for k in BATCH_KEYS) for batch in data_loader)

    def save(self, obj, path):
        if XLA:
            xm.save(obj, str(path) + '.tmp')            # moves every tensor to the CPU first
            os.replace(str(path) + '.tmp', path)
        else:
            atomic_torch_save(obj, path)


def atomic_torch_save(obj, path):
    temp = str(path) + '.tmp'
    torch.save(obj, temp); os.replace(temp, path)


def dict_collate(batch, augment):
    xs_, ps, ys, ws = zip(*batch)
    x = torch.stack([augment_series_cpu(v) for v in xs_]) if augment else torch.stack(xs_)
    return dict(x=x, present=torch.stack(ps), y=torch.stack(ys), w=torch.stack(ws))


def tpu_worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)
    cv2.setNumThreads(0)


def _data_loader(rows, training, seed):
    return DataLoader(KneeDS(df, rows, training), batch_size=STUDIES_PER_STEP, shuffle=False,
                      num_workers=TPU_WORKERS, collate_fn=functools.partial(dict_collate, augment=training),
                      drop_last=training, worker_init_fn=tpu_worker_init,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False,
                      **({'prefetch_factor': 4} if TPU_WORKERS else {}))


def _predict(be, model, rows, seed):
    """Sigmoid predictions for rows, in order; the last batch is padded to the fixed size."""
    rows = np.asarray(rows, int)
    if not len(rows):
        return np.empty((0, len(LABEL_COLS)), np.float32)
    padded = np.concatenate([rows, np.repeat(rows[-1:], (-len(rows)) % STUDIES_PER_STEP)])
    preds = []
    model.eval()
    with torch.no_grad():
        for x, present, *_ in be.loader(_data_loader(padded, False, seed)):
            with be.autocast():
                logits, _ = model(x, present)
            preds.append(logits.float().sigmoid().cpu())
    return torch.cat(preds).numpy()[:len(rows)]


def train_fold(be, fold):
    fold_seed = SEED + fold
    torch.manual_seed(fold_seed); np.random.seed(fold_seed); random.seed(fold_seed)
    tr_rows = np.flatnonzero(((folds >= 0) & (folds != fold)) | ((folds == -1) & (GOLD_WEIGHT > 0)))
    va_rows = np.flatnonzero(folds == fold)
    gold_rows = np.flatnonzero(folds == -1)
    assert not set(tr_rows) & set(va_rows) and df.iloc[va_rows]['source'].eq('pseudo').all()
    if PREFLIGHT:
        tr_rows, va_rows, gold_rows = tr_rows[:PREFLIGHT], va_rows[:max(4, PREFLIGHT // 2)], gold_rows[:4]
    model = build_model(recipe, len(LABEL_COLS), len(PLANES), pretrained=True)
    model.static_shapes, model.augment_in_model = True, False
    model = model.to(be.device)
    encoder_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': encoder_params, 'lr': LR * ENCODER_LR_SCALE},
                             {'params': head_params, 'lr': LR}], weight_decay=WD)
    steps_per_epoch = len(tr_rows) // STUDIES_PER_STEP
    assert steps_per_epoch > 0, 'fewer training studies than one step'
    total_steps = EPOCHS * steps_per_epoch
    pct_start = PCT_START
    if abs(pct_start * total_steps - 1) < 1e-6:
        pct_start = 1.5 / total_steps if total_steps >= 2 else .5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR * ENCODER_LR_SCALE, LR],
                                                total_steps=total_steps, pct_start=pct_start)
    ema, ema_steps = None, 0
    if EMA_DECAY:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters(): p.requires_grad_(False)
    path = os.path.join(OUT, f'fold_{fold}.pt')
    assert not os.path.exists(path), f'{path} already exists; choose an empty OUT'
    best, hist, best_val, best_gold = float('inf'), [], None, None
    for ep in range(EPOCHS):
        epoch_seed = fold_seed * 1000 + ep
        torch.manual_seed(epoch_seed); np.random.seed(epoch_seed); random.seed(epoch_seed)
        order = np.random.default_rng(epoch_seed).permutation(tr_rows)
        model.train(); t0 = time.time()
        total = torch.zeros((), device=be.device)
        step = 0
        for x, present, y, w in be.loader(_data_loader(order, True, epoch_seed)):
            with be.autocast():
                logits, _ = model(x, present)
            loss = masked_bce(logits.float(), y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema_steps += 1
                decay = min(EMA_DECAY, (1 + ema_steps) / (10 + ema_steps))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - decay)
                    for b_e, b_m in zip(ema.buffers(), model.buffers()): b_e.copy_(b_m)
            total += loss.detach()
            step += 1
            be.step()
            if XLA and (PREFLIGHT or (ep == 0 and step <= 4) or step % 100 == 0):
                # Graph compilations must stop growing after the first steps (no per-step recompiles).
                import torch_xla.debug.metrics as met
                print(f'[xla] fold {fold} epoch {ep} step {step}: {(met.metric_data("CompileTime") or [0])[0]} '
                      f'compilations, {(time.time() - t0) / step:.1f} s/step', flush=True)
        train_loss = float(total.cpu()) / max(step, 1)
        train_minutes = (time.time() - t0) / 60
        judged = ema if ema is not None else model
        pv = _predict(be, judged, va_rows, epoch_seed)
        yv = df.iloc[va_rows][LABEL_COLS].to_numpy(np.float32)
        observed = np.isfinite(yv)
        clipped = np.clip(pv, 1e-7, 1 - 1e-7)
        val_loss = float((-(np.nan_to_num(yv) * np.log(clipped) +
                            (1 - np.nan_to_num(yv)) * np.log1p(-clipped)))[observed].mean())
        auc, _ = macro_auc(yv, pv, LABEL_COLS)
        rec = dict(fold=fold, epoch=ep, loss=train_loss, val_bce=val_loss, auc_pseudo=auc,
                   minutes=(time.time() - t0) / 60, train_minutes=train_minutes, steps=step)
        if XLA:
            import torch_xla.debug.metrics as met
            rec['compiles'] = (met.metric_data('CompileTime') or [0])[0]
        hist.append(rec)
        print(rec, flush=True)
        if val_loss < best:
            best = val_loss
            best_val = pv
            best_gold = _predict(be, judged, gold_rows, epoch_seed) if len(gold_rows) else None
            be.save(dict(model=judged.state_dict(), cfg=recipe, epoch=ep, fold=fold,
                         val_bce=best, auc_pseudo=auc), path)
        pd.DataFrame(hist).to_csv(os.path.join(OUT, f'history_fold_{fold}.csv'), index=False)
    output = pd.DataFrame(best_val, columns=LABEL_COLS)
    output.insert(0, 'StudyInstanceUID', df.iloc[va_rows]['StudyInstanceUID'].to_numpy())
    output['fold'] = fold
    output.to_csv(os.path.join(OUT, f'oof_fold_{fold}.csv'), index=False)
    if best_gold is not None:
        gold_output = pd.DataFrame(best_gold, columns=LABEL_COLS)
        gold_output.insert(0, 'StudyInstanceUID', df.iloc[gold_rows]['StudyInstanceUID'].to_numpy())
        gold_output.to_csv(os.path.join(OUT, f'gold_fold_{fold}.csv'), index=False)
        print('Gold audit after checkpoint selection'
              + (' (IN-SAMPLE: gold trained at weight %g)' % GOLD_WEIGHT if GOLD_WEIGHT > 0 else '') + ':',
              macro_auc(df.iloc[gold_rows][LABEL_COLS].to_numpy(float), best_gold, LABEL_COLS)[0], flush=True)


def run_folds():
    be = _Backend()
    print(f'TPU runtime: {"XLA SPMD" if XLA else "CPU"} | {be.n_devices} device(s) | folds {FOLDS_TO_RUN} | '
          f'{STUDIES_PER_STEP} studies/step | {TPU_WORKERS} loader workers'
          + (f' | PREFLIGHT {PREFLIGHT} studies, {EPOCHS} epochs' if PREFLIGHT else ''), flush=True)
    durations = []
    for fold in FOLDS_TO_RUN:
        elapsed_h = (time.time() - SESSION_START) / 3600
        need_h = 1.15 * max(durations) / 3600 if durations else 0
        if durations and elapsed_h + need_h > TIME_BUDGET_H:
            print(f'[time guard] skipping fold {fold}: {elapsed_h:.2f} h elapsed, a fold takes ~{need_h:.2f} h, '
                  f'budget {TIME_BUDGET_H} h', flush=True)
            continue
        started = time.time()
        train_fold(be, fold)
        durations.append(time.time() - started)


if __name__ == '__main__':
    write_atomic(os.path.join(OUT, 'folds.csv'), lambda p: fold_table.to_csv(p, index=False))
    write_atomic(os.path.join(OUT, 'run_config.json'), json.dumps(recipe, indent=2))
    run_folds()
    paths = [os.path.join(OUT, f'oof_fold_{f}.csv') for f in range(N_FOLDS)]
    if all(os.path.exists(p) for p in paths):
        oof = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        oof.to_csv(os.path.join(OUT, 'oof_predictions.csv'), index=False)
