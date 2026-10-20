"""Decode once, run fold models sequentially, rank-ensemble, and write a submission."""
class TestDS(Dataset):
    def __init__(self, manifest): self.manifest = manifest
    def __len__(self): return len(self.manifest)
    def __getitem__(self, i):
        row = self.manifest.iloc[i]
        vol = np.zeros((len(PLANES), N_SLICES, IMG, IMG), np.uint8)
        present = np.zeros(len(PLANES), bool)
        for slot in range(len(PLANES)):
            directory = row[f'dir_{slot}']
            if not directory: continue
            try:
                vol[slot] = (load_series(directory, SLOTS[slot][1], row['study_side'],
                                         **slot_kwargs(RECIPE, slot, row_anchor(row)))
                             if MODERN else load_series(directory))
                present[slot] = True
            except Exception:
                continue
        return torch.from_numpy(vol), torch.from_numpy(present), i

# A scratch disk cache avoids repeating expensive DICOM decoding for every fold.
# It is private to this run and deleted once the CSV has been written.
import tempfile, shutil
# The disk with the most free space holds the decoded test volumes (v3: ~8 MB per study).
_scratch_roots = [d for d in ('/kaggle/tmp', tempfile.gettempdir(), os.path.dirname(OUT_CSV)) if os.path.isdir(d)]
scratch = tempfile.mkdtemp(prefix='knee-test-', dir=max(_scratch_roots, key=lambda d: shutil.disk_usage(d).free))
shape = (len(manifest), len(PLANES), N_SLICES, IMG, IMG)
needed = int(np.prod(shape))
assert shutil.disk_usage(scratch).free > needed + 1024**3, 'Insufficient scratch space for decode-once inference'
volumes = np.lib.format.open_memmap(os.path.join(scratch,'test.npy'), mode='w+', dtype=np.uint8, shape=shape)
presence = np.zeros(shape[:2], bool)
started = time.time()
# Single-process loading isolates DICOM failures and avoids worker crashes aborting the iterator.
for i in tqdm(range(len(manifest)), desc='decode once'):
    if time.time()-started > MAX_HOURS*3600*.65:
        print('Decode budget reached; remaining rows use fallback'); break
    try:
        x, present, _ = TestDS(manifest)[i]
        volumes[i] = x.numpy(); presence[i] = present.numpy()
    except Exception as exc:
        print(f'Study {i}: {type(exc).__name__}')
volumes.flush()
all_predictions, member_families, member_paths = [], [], []
FAMILY_WEIGHTS_GIVEN = globals().get('FAMILY_WEIGHTS') or {}
WINDOW_TTA = int(globals().get('WINDOW_TTA', 0))   # extra passes over random training-size window subsets
coverage = []
for checkpoint in CKPT_PATHS:
    if time.time()-started > MAX_HOURS*3600: break
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    member = f"{state['cfg'].get('arch', 'kneenet')}:{state['cfg']['backbone']}"
    family = state['cfg'].get('preset', member)
    if FAMILY_WEIGHTS_GIVEN and FAMILY_WEIGHTS_GIVEN.get(family, 1.0) == 0:
        # A zero-weight family cannot change the blend; skip its inference time.
        coverage.append(dict(checkpoint=checkpoint, member=member, family=family, predicted=0, skipped='weight 0'))
        del state
        continue
    # Members may differ in architecture or backbone; each is rebuilt from its own recipe. A v3 member on
    # v4 volumes reads the first len(its planes) slots (checked when the members were listed).
    n_slots = len(state['cfg'].get('planes', PLANES))
    model = build_model(state['cfg'], len(LABEL_COLS), n_slots, pretrained=False).to(DEV)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    pred = np.full((len(manifest),len(LABEL_COLS)), np.nan, np.float32)
    valid_rows = np.flatnonzero(presence.any(1))
    def predict_rows(rows):
        if not len(rows): return
        try:
            x = torch.from_numpy(np.asarray(volumes[rows][:, :n_slots]).copy()).to(DEV)
            p = torch.from_numpy(presence[rows][:, :n_slots].copy()).to(DEV)
            with torch.inference_mode():
                logits, _ = model(x,p)
                values = logits.float().sigmoid()
                if WINDOW_TTA and getattr(model, 'train_windows', None):
                    # Half the full pass, half the mean of WINDOW_TTA passes over random training-size window subsets.
                    model.subset_windows = model.train_windows
                    subsets = torch.stack([model(x,p)[0].float().sigmoid() for _ in range(WINDOW_TTA)]).mean(0)
                    model.subset_windows = None
                    values = .5 * values + .5 * subsets
                values = values.cpu().numpy()
            finite = np.isfinite(values).all(1)
            pred[rows[finite]] = values[finite]
        except Exception as exc:
            if DEV=='cuda': torch.cuda.empty_cache()
            if len(rows)>1:
                middle = len(rows)//2
                predict_rows(rows[:middle]); predict_rows(rows[middle:])
            else: print(f'Inference failed for row {rows[0]}: {type(exc).__name__}')
    for offset in tqdm(range(0,len(valid_rows),BATCH),desc=os.path.basename(checkpoint)):
        if time.time()-started > MAX_HOURS*3600: break
        predict_rows(valid_rows[offset:offset+BATCH])
    good = np.isfinite(pred).all(1)
    coverage.append(dict(checkpoint=checkpoint, member=member, family=family, predicted=int(good.sum())))
    # A partially evaluated member would distort ranks for its unprocessed tail; omit it.
    if good.sum() == len(valid_rows) and good.any():
        pred[~good] = np.median(pred[good], 0)
        all_predictions.append(pred); member_families.append(family); member_paths.append(checkpoint)
    elif not all_predictions and good.any():
        pred[~good] = np.median(pred[good], 0)
        all_predictions.append(pred); member_families.append(family); member_paths.append(checkpoint)
    del model
    gc.collect()
    if DEV=='cuda': torch.cuda.empty_cache()

# ---- optional transductive self-training on the test set (TTST, ours, 2026-10-09) ----------------
# Test studies come from many sites and protocols. Each reader member is fine-tuned briefly on the
# test volumes themselves, with the weighted ensemble's mean probability as soft targets and its
# usual training augmentations (affine/intensity jitter, window subsets, slot dropout), then
# predicts again. No label of any kind is used. Off unless TTST_EPOCHS > 0; a member that does not
# finish inside the time budget keeps its first-pass prediction.
TTST_EPOCHS = int(globals().get('TTST_EPOCHS', 0))
TTST_LR = float(globals().get('TTST_LR', 1e-4))     # head rate; the encoder trains at 0.2x
TTST_BATCH = int(globals().get('TTST_BATCH', 4))
ttst_log = []
if TTST_EPOCHS > 0 and len(all_predictions) > 0:
    _fw = globals().get('FAMILY_WEIGHTS') or {}
    _use = [k for k, f in enumerate(member_families) if _fw.get(f, 1.0) > 0]
    soft_target = np.mean([all_predictions[k] for k in _use], axis=0).astype(np.float32)
    rows_all = np.flatnonzero(presence.any(1))
    ttst_rng = np.random.default_rng(0)
    for k in _use:
        if time.time() - started > MAX_HOURS * 3600 * .9: break
        state = torch.load(member_paths[k], map_location='cpu', weights_only=False)
        if state['cfg'].get('arch') != 'reader25d' or len(state['cfg'].get('planes', PLANES)) != len(PLANES):
            del state; continue          # TTST fine-tunes full-slot readers only
        model = build_model(state['cfg'], len(LABEL_COLS), len(PLANES), pretrained=False).to(DEV)
        model.load_state_dict(state['model'], strict=True)
        del state
        enc_params = list(model.enc.parameters())
        enc_ids = {id(q) for q in enc_params}
        opt = torch.optim.AdamW([{'params': enc_params, 'lr': TTST_LR * .2},
                                 {'params': [q for q in model.parameters() if id(q) not in enc_ids], 'lr': TTST_LR}],
                                weight_decay=.02)
        scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda')
        finished, losses = True, []
        for _ in range(TTST_EPOCHS):
            model.train()
            order = rows_all[ttst_rng.permutation(len(rows_all))]
            for off in range(0, len(order), TTST_BATCH):
                if time.time() - started > MAX_HOURS * 3600 * .9:
                    finished = False; break
                b = np.sort(order[off:off + TTST_BATCH])
                x = torch.from_numpy(np.asarray(volumes[b]).copy()).to(DEV)
                p = torch.from_numpy(presence[b]).to(DEV)
                y = torch.from_numpy(soft_target[b]).to(DEV)
                logits, _ = model(x, p)
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update()
                losses.append(float(loss.detach()))
            if not finished: break
        if finished:
            model.eval()
            pred = np.full((len(manifest), len(LABEL_COLS)), np.nan, np.float32)
            with torch.inference_mode():
                for off in range(0, len(rows_all), BATCH):
                    b = rows_all[off:off + BATCH]
                    logits, _ = model(torch.from_numpy(np.asarray(volumes[b]).copy()).to(DEV),
                                      torch.from_numpy(presence[b]).to(DEV))
                    pred[b] = logits.float().sigmoid().cpu().numpy()
            good = np.isfinite(pred).all(1)
            if good[rows_all].all():
                pred[~good] = np.median(pred[good], 0)
                all_predictions[k] = pred
            else:
                finished = False
        ttst_log.append(dict(member=member_paths[k], finished=bool(finished), steps=len(losses),
                             first_loss=losses[0] if losses else None, last_loss=losses[-1] if losses else None))
        print('TTST', ttst_log[-1])
        del model, opt
        gc.collect()
        if DEV == 'cuda': torch.cuda.empty_cache()

# Rank-average within each family, then across families, so a family's vote does not
# grow with its number of folds. FAMILY_WEIGHTS (default equal) is fixed before scoring, from OOF evidence.
family_weights = globals().get('FAMILY_WEIGHTS') or {}
# Explicit weights must name every attached family: a new family would otherwise get weight 1.
unweighted = sorted(set(member_families) - set(family_weights)) if family_weights else []
assert not unweighted, f'FAMILY_WEIGHTS has no weight for {unweighted}; choose one with 06_oof_blend_check'
family_ranks = {}
for fam in dict.fromkeys(member_families):
    family_ranks[fam] = rank_blend([p for p, f in zip(all_predictions, member_families) if f == fam])
if not all_predictions:
    preds = np.full((len(manifest),len(LABEL_COLS)), .5)
elif len(all_predictions) == 1:
    preds = all_predictions[0]
else:
    total = sum(family_weights.get(f, 1.0) for f in family_ranks)
    preds = sum(family_weights.get(f, 1.0) * r for f, r in family_ranks.items()) / total
# Undecoded studies take the median, not the mean: with rare findings the mean ranks above most studies.
fill = np.median(preds, 0) if len(preds) else np.full(len(LABEL_COLS),.5)
sub = pd.DataFrame(preds,columns=LABEL_COLS)
sub.insert(0,'StudyInstanceUID',manifest['StudyInstanceUID'].to_numpy())
sample = pd.read_csv(os.path.join(COMP_DIR,'sample_submission.csv'))
sub = sample[['StudyInstanceUID']].merge(sub,on='StudyInstanceUID',how='left',validate='one_to_one')
for k,col in enumerate(LABEL_COLS): sub[col] = sub[col].fillna(float(fill[k]))
sub = sub[sample.columns]
assert len(sub)==len(sample) and np.isfinite(sub[LABEL_COLS].to_numpy()).all()
assert sub[LABEL_COLS].to_numpy().min()>=0 and sub[LABEL_COLS].to_numpy().max()<=1
sub.to_csv(OUT_CSV,index=False)
diagnostics = dict(models=coverage, ensemble_members=len(all_predictions),
                   families={f: member_families.count(f) for f in family_ranks},
                   family_weights={f: family_weights.get(f, 1.0) for f in family_ranks},
                   blank_studies=int((~presence.any(1)).sum()), ttst=ttst_log,
                   largest_tie={c:int(sub[c].value_counts().max()) for c in LABEL_COLS},
                   elapsed_minutes=(time.time()-started)/60)
# Studies with no decoded series: a downstream blend keeps the other model's rank for them.
diagnostics['blank_study_ids'] = manifest.loc[~presence.any(1), 'StudyInstanceUID'].astype(str).tolist()
with open(os.path.join(os.path.dirname(OUT_CSV),'inference_diagnostics.json'),'w') as handle:
    json.dump(diagnostics,handle,indent=2)
print('Wrote',OUT_CSV,diagnostics)
del volumes
shutil.rmtree(scratch)
