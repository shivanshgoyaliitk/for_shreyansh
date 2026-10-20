"""Five-fold training runtime, embedded in training.ipynb."""
def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)
    cv2.setNumThreads(0)


def make_loader(rows, training, seed):
    return DataLoader(KneeDS(df, rows, training), batch_size=BATCH,
                      shuffle=training, num_workers=NUM_WORKERS,
                      collate_fn=collate, pin_memory=DEV=='cuda', drop_last=False,
                      persistent_workers=NUM_WORKERS > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      **({'prefetch_factor': 2} if NUM_WORKERS else {}))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, present, y, *_ in loader:
        logits, _ = model(x.to(DEV, non_blocking=True), present.to(DEV, non_blocking=True))
        ps.append(logits.float().sigmoid().cpu().numpy()); ys.append(y.numpy())
    if not ys: return np.empty((0,len(LABEL_COLS))), np.empty((0,len(LABEL_COLS)))
    return np.concatenate(ys), np.concatenate(ps)


def atomic_torch_save(obj, path):
    temp = str(path) + '.tmp'
    torch.save(obj, temp); os.replace(temp, path)


os.makedirs(OUT, exist_ok=True)
folds = assign_folds(df, LABEL_COLS, N_FOLDS, SEED, GROUP_COL if GROUP_COL in df else None)
fold_table = df[['StudyInstanceUID', 'source']].assign(fold=folds)
write_atomic(os.path.join(OUT, 'folds.csv'), lambda p: fold_table.to_csv(p, index=False))
# Record the dataset identity so resumed folds cannot silently use changed labels or cache.
import hashlib, importlib.metadata
identity = hashlib.sha256((df.to_json(orient='split') + json.dumps(CACHE_META, sort_keys=True)
                           + fold_table.to_json()).encode()).hexdigest()
recipe = dict(arch=ARCH, backbone=BACKBONE, img=IMG, n_slices=N_SLICES, planes=PLANES,
              label_cols=LABEL_COLS, t_train=T_TRAIN, preprocessing=CACHE_META,
              unfreeze_last=UNFREEZE_LAST, head_hidden=HEAD_HIDDEN, pct_start=PCT_START,
              normalization=NORMALIZATION, drop=DROPOUT, seed=SEED,
              epochs=EPOCHS, lr=LR, weight_decay=WD, n_folds=N_FOLDS, accum_steps=ACCUM_STEPS,
              encoder_lr_scale=ENCODER_LR_SCALE, batch=BATCH, identity=identity,
              gold_weight=GOLD_WEIGHT, ema=EMA_DECAY, clip=CLIP, train_windows=TRAIN_WINDOWS, slot_dropout=SLOT_DROPOUT,
              preset=PRESET, grad_checkpoint=GRAD_CHECKPOINT, teacher=TEACHER_SUMMARY,
              group_col=GROUP_COL if GROUP_COL in df else None,
              aux_dim=globals().get('AUX_DIM', 0), report_aux=globals().get('REPORT_AUX_SUMMARY'),
              versions={p: importlib.metadata.version(p) for p in ('torch','timm','numpy')})
write_atomic(os.path.join(OUT, 'run_config.json'), json.dumps(recipe, indent=2))
REPORT_AUX_WEIGHT = globals().get('REPORT_AUX', 0.0) if globals().get('AUX_DIM', 0) else 0.0


def report_aux_loss(pred, target, known):
    # Squared error to the report embedding (components scaled to mean variance 1), rows with one only.
    rows = known.all(1)
    if not rows.any():
        return pred.sum() * 0
    return ((pred[rows].float() - target[rows]) ** 2).mean()


all_hist = []
for fold in FOLDS_TO_RUN:
    assert fold in range(N_FOLDS)
    fold_seed = SEED + fold
    torch.manual_seed(fold_seed); np.random.seed(fold_seed); random.seed(fold_seed)
    # Gold studies never validate; with GOLD_WEIGHT > 0 they train in every fold.
    tr_rows = np.flatnonzero(((folds >= 0) & (folds != fold)) | ((folds == -1) & (GOLD_WEIGHT > 0)))
    va_rows = np.flatnonzero(folds == fold)
    assert not set(tr_rows) & set(va_rows)
    assert df.iloc[va_rows]['source'].eq('pseudo').all()
    assert GOLD_WEIGHT > 0 or df.iloc[tr_rows]['source'].eq('pseudo').all()
    dl_tr = make_loader(tr_rows, True, fold_seed)
    dl_val = make_loader(va_rows, False, fold_seed)
    model = build_model(recipe, len(LABEL_COLS), len(PLANES),
                        pretrained=not (RESUME and os.path.exists(os.path.join(OUT, f'last_fold_{fold}.pt')))).to(DEV)
    if GRAD_CHECKPOINT and hasattr(model.enc, 'set_grad_checkpointing'):
        model.enc.set_grad_checkpointing(True)     # recompute activations: fits large backbones on a T4
    # Frozen encoder blocks are left out of the optimizer entirely.
    encoder_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = [p for name,p in model.named_parameters() if not name.startswith('enc.')]
    print(f'trainable: encoder {sum(p.numel() for p in encoder_params)/1e6:.1f}M, '
          f'head {sum(p.numel() for p in head_params)/1e6:.2f}M')
    opt = torch.optim.AdamW([{'params':encoder_params,'lr':LR*ENCODER_LR_SCALE},
                            {'params':head_params,'lr':LR}], weight_decay=WD)
    total_steps = EPOCHS*math.ceil(len(dl_tr)/ACCUM_STEPS)
    pct_start = PCT_START
    if abs(pct_start * total_steps - 1) < 1e-6:
        # OneCycleLR divides by zero when the warmup is exactly one optimizer step.
        pct_start = 1.5 / total_steps if total_steps >= 2 else .5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR*ENCODER_LR_SCALE, LR], total_steps=total_steps,
                    pct_start=pct_start)
    scaler = torch.amp.GradScaler(DEV, enabled=DEV=='cuda')
    # Exponential moving average of the weights; it is what validates and what is saved.
    ema, ema_steps = None, 0
    if EMA_DECAY:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters(): p.requires_grad_(False)
    path = os.path.join(OUT, f'fold_{fold}.pt')
    last_path = os.path.join(OUT, f'last_fold_{fold}.pt')
    start, best, hist = 0, float('inf'), []
    if RESUME and os.path.exists(last_path):
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        assert saved['cfg'] == recipe, 'Resume recipe differs; use a new OUT directory'
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        start, best, hist = saved['epoch']+1, saved['best'], saved['history']
        if ema is not None:
            ema.load_state_dict(saved['ema']); ema_steps = saved['ema_steps']
        assert os.path.exists(path), 'Resume requires both last and best fold checkpoints'
    elif os.path.exists(path):
        raise RuntimeError(f'{path} already exists; set RESUME=True or choose an empty OUT')
    for ep in range(start, EPOCHS):
        # Epoch-local worker seeds make interrupted runs repeatable at epoch boundaries.
        epoch_seed = fold_seed*1000 + ep
        torch.manual_seed(epoch_seed); np.random.seed(epoch_seed); random.seed(epoch_seed)
        dl_tr = make_loader(tr_rows, True, epoch_seed)
        model.train(); total, count, aux_total = 0., 0, 0.; t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, present, y, w) in enumerate(tqdm(dl_tr, desc=f'fold {fold} epoch {ep+1}')):
            x, present, y, w = x.to(DEV, non_blocking=True), present.to(DEV), y.to(DEV), w.to(DEV)
            n_lab = len(LABEL_COLS)
            y_aux, w_aux, y, w = y[:, n_lab:], w[:, n_lab:], y[:, :n_lab], w[:, :n_lab]
            logits, _ = model(x, present)
            loss = masked_bce(logits.float(), y, w)
            if REPORT_AUX_WEIGHT:
                aux = report_aux_loss(model.last_aux, y_aux, w_aux > 0)
                loss = loss + REPORT_AUX_WEIGHT * aux
                aux_total += aux.item() * len(x)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            group_start = (step // ACCUM_STEPS) * ACCUM_STEPS
            group_size = min(ACCUM_STEPS, len(dl_tr)-group_start)
            scaler.scale(loss/group_size).backward()
            if (step+1) % ACCUM_STEPS == 0 or step+1 == len(dl_tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                old_scale = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= old_scale: sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema_steps += 1
                    decay = min(EMA_DECAY, (1 + ema_steps) / (10 + ema_steps))
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - decay)
                        for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            total += loss.item()*len(x); count += len(x)
        judged = ema if ema is not None else model
        yv, pv = evaluate(judged, dl_val)
        observed = np.isfinite(yv)
        clipped = np.clip(pv, 1e-7, 1-1e-7)
        val_loss = float((-(np.nan_to_num(yv)*np.log(clipped) +
                           (1-np.nan_to_num(yv))*np.log1p(-clipped)))[observed].mean())
        auc, _ = macro_auc(yv, pv, LABEL_COLS)
        rec = dict(fold=fold, epoch=ep, loss=total/count, val_bce=val_loss,
                   auc_pseudo=auc, minutes=(time.time()-t0)/60)
        if REPORT_AUX_WEIGHT:
            rec['report_aux'] = aux_total / count
        hist.append(rec); print(rec)
        # Soft-label BCE on pseudo-labelled validation selects epochs; gold is never consulted.
        if val_loss < best:
            best = val_loss
            atomic_torch_save(dict(model=judged.state_dict(), cfg=recipe, epoch=ep,
                                   fold=fold, val_bce=best, auc_pseudo=auc), path)
        atomic_torch_save(dict(model=model.state_dict(), cfg=recipe, epoch=ep, best=best,
                               optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                               scaler=scaler.state_dict(), history=hist,
                               ema=None if ema is None else ema.state_dict(), ema_steps=ema_steps), last_path)
        pd.DataFrame(hist).to_csv(os.path.join(OUT, f'history_fold_{fold}.csv'), index=False)
    model.load_state_dict(torch.load(path, map_location=DEV, weights_only=False)['model'])
    yv, pv = evaluate(model, dl_val)
    output = pd.DataFrame(pv, columns=LABEL_COLS)
    output.insert(0,'StudyInstanceUID',df.iloc[va_rows]['StudyInstanceUID'].to_numpy())
    output['fold'] = fold
    output.to_csv(os.path.join(OUT,f'oof_fold_{fold}.csv'), index=False)
    gold_rows = np.flatnonzero(folds == -1)
    yg, pg = evaluate(model, make_loader(gold_rows, False, fold_seed))
    gold_output = pd.DataFrame(pg, columns=LABEL_COLS)
    gold_output.insert(0,'StudyInstanceUID',df.iloc[gold_rows]['StudyInstanceUID'].to_numpy())
    gold_output.to_csv(os.path.join(OUT,f'gold_fold_{fold}.csv'), index=False)
    print('Gold audit after checkpoint selection'
          + (' (IN-SAMPLE: gold trained at weight %g)' % GOLD_WEIGHT if GOLD_WEIGHT > 0 else '') + ':',
          macro_auc(yg,pg,LABEL_COLS)[0])
    all_hist.extend(hist)
    del model, ema, opt, sched, scaler, dl_tr, dl_val
    gc.collect()
    if DEV=='cuda': torch.cuda.empty_cache()

paths = [os.path.join(OUT,f'oof_fold_{f}.csv') for f in range(N_FOLDS)]
if all(os.path.exists(p) for p in paths):
    oof = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = df.loc[folds>=0,'StudyInstanceUID']
    assert oof['StudyInstanceUID'].is_unique and set(oof['StudyInstanceUID']) == set(expected)
    aligned = oof.set_index('StudyInstanceUID').loc[expected]
    oof.to_csv(os.path.join(OUT,'oof_predictions.csv'), index=False)
    print('OOF pseudo AUC (noisy-label proxy, selection-biased):',
          macro_auc(df.loc[folds>=0,LABEL_COLS].to_numpy(),aligned[LABEL_COLS].to_numpy(),LABEL_COLS))
else:
    print('Partial fold run; complete all folds before comparing aggregate OOF results.')

# Fold predictions share the same gold study order; audit the exact inference blend.
gold_paths = [os.path.join(OUT,f'gold_fold_{f}.csv') for f in range(N_FOLDS)]
if all(os.path.exists(p) for p in gold_paths):
    gold_ids = df.loc[folds==-1,'StudyInstanceUID']
    if len(gold_ids):
        members = [pd.read_csv(p).set_index('StudyInstanceUID').loc[gold_ids,LABEL_COLS].to_numpy()
                   for p in gold_paths]
        blended = rank_blend(members)
        audit = pd.DataFrame(blended,columns=LABEL_COLS)
        audit.insert(0,'StudyInstanceUID',gold_ids.to_numpy())
        audit.to_csv(os.path.join(OUT,'gold_ensemble.csv'),index=False)
        print('Gold fold-ensemble audit' + (' (in-sample)' if GOLD_WEIGHT > 0 else '') + ':', macro_auc(df.loc[folds==-1,LABEL_COLS].to_numpy(),blended,LABEL_COLS))
