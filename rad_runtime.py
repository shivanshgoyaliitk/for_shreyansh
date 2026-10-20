"""RadImageNet head family: encode every study once, then train one head per fold.

Embedded in the RadImageNet training notebook in place of train_runtime.py. The frozen
encoder never changes, so its tokens are computed a single time; each fold then trains
only the query head, following the public E9/E11 loop (AdamW 2e-4, weight decay 3e-3,
batch 48, at most 24 epochs, early stop after 5 epochs without a pseudo-validation gain).
"""
import hashlib, importlib.metadata

RAD_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
RAD_PATIENCE = 5
RAD_VERIFY = globals().get('RAD_VERIFY', True)   # official checkpoint hash check


def find_radimagenet(root):
    hits = []
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        hits += [os.path.join(path, f) for f in files if f == 'ResNet50.pt']
    assert len(hits) == 1, f'Attach the official RadImageNet ResNet50.pt exactly once (found {hits})'
    digest = hashlib.sha256(open(hits[0], 'rb').read()).hexdigest()
    assert digest == RAD_SHA256 or not RAD_VERIFY, f'RadImageNet checkpoint differs: {digest}'
    return hits[0], digest


class CentreSlices(Dataset):
    # Reads only the window-centre slices the encoder uses (a third of the cache).
    def __init__(self, frame, centres):
        self.frame, self.centres, self.caches = frame.reset_index(drop=True), centres, {}
        self.present = self.frame[[f"present_{s}" for s in range(len(PLANES))]].to_numpy(bool)
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        shard = int(self.frame.at[i, 'cache_id']) if 'cache_id' in self.frame else 0
        if shard not in self.caches:
            self.caches[shard] = np.load(os.path.join(CACHE_DIRS[shard], "cache.npy"), mmap_mode="r")
        vol = np.asarray(self.caches[shard][int(self.frame.at[i, 'row'])][:, self.centres])
        return torch.from_numpy(np.ascontiguousarray(vol)), torch.from_numpy(self.present[i].copy())


os.makedirs(OUT, exist_ok=True)
folds = assign_folds(df, LABEL_COLS, N_FOLDS, SEED, GROUP_COL if GROUP_COL in df else None)
fold_table = df[['StudyInstanceUID', 'source']].assign(fold=folds)
write_atomic(os.path.join(OUT, 'folds.csv'), lambda p: fold_table.to_csv(p, index=False))
weights_path, weights_sha = find_radimagenet(LABEL_ROOT)
centres = window_centres(N_SLICES)
identity = hashlib.sha256((df.to_json(orient='split') + json.dumps(CACHE_META, sort_keys=True)
                           + fold_table.to_json()).encode()).hexdigest()
recipe = dict(arch='radhead', backbone='radimagenet_resnet50', preset=PRESET, img=IMG, n_slices=N_SLICES, planes=PLANES,
              label_cols=LABEL_COLS, preprocessing=CACHE_META, head_hidden=HEAD_HIDDEN,
              n_windows=len(centres), seed=SEED, epochs=EPOCHS, lr=LR, weight_decay=WD, batch=BATCH,
              clip=CLIP, patience=RAD_PATIENCE, n_folds=N_FOLDS, gold_weight=GOLD_WEIGHT,
              group_col=GROUP_COL if GROUP_COL in df else None, encoder_sha256=weights_sha,
              identity=identity, versions={p: importlib.metadata.version(p) for p in ('torch','timm','numpy')})
write_atomic(os.path.join(OUT, 'run_config.json'), json.dumps(recipe, indent=2))

# ---- encode once ----------------------------------------------------------------
torch.manual_seed(SEED)
encoder = build_model(dict(recipe, weights_path=weights_path), len(LABEL_COLS), len(PLANES)).to(DEV).eval()
tokens = np.zeros((len(df), len(PLANES) * len(centres), 2048), np.float16)
masks = np.zeros((len(df), len(PLANES) * len(centres)), np.float32)
loader = DataLoader(CentreSlices(df, centres), batch_size=8, num_workers=NUM_WORKERS, shuffle=False)
started, offset = time.time(), 0
for x, present in tqdm(loader, desc='RadImageNet encode'):
    # The encoder expects window layout (centre at 3k+1); rebuild it from the centre slices.
    full = torch.zeros((*x.shape[:2], 3 * len(centres), *x.shape[3:]), dtype=torch.uint8)
    full[:, :, [3 * k + 1 for k in range(len(centres))]] = x
    t, m = encoder.encode(full.to(DEV), present.to(DEV))
    tokens[offset:offset + len(x)], masks[offset:offset + len(x)] = t.cpu().numpy(), m.cpu().numpy()
    offset += len(x)
print(f'encoded {offset:,} studies in {(time.time() - started) / 60:.1f} min')
backbone_state = {k: v for k, v in encoder.state_dict().items() if k.startswith('backbone.')}
y_all = df[LABEL_COLS].to_numpy(np.float32)
w_all = df[[f'w_{c}' for c in LABEL_COLS]].to_numpy(np.float32) if f'w_{LABEL_COLS[0]}' in df else np.ones_like(y_all)


def predict_tokens(model, rows):
    model.eval(); out = []
    with torch.inference_mode():
        for b in range(0, len(rows), 256):
            r = rows[b:b + 256]
            logits = model.head(torch.from_numpy(tokens[r]).to(DEV), torch.from_numpy(masks[r]).to(DEV))
            out.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, len(LABEL_COLS)))


for fold in FOLDS_TO_RUN:
    torch.manual_seed(SEED + 100 + fold); np.random.seed(SEED + 100 + fold)
    tr_rows = np.flatnonzero(((folds >= 0) & (folds != fold)) | ((folds == -1) & (GOLD_WEIGHT > 0)))
    va_rows = np.flatnonzero(folds == fold)
    assert not set(tr_rows) & set(va_rows) and df.iloc[va_rows]['source'].eq('pseudo').all()
    model = build_model(recipe, len(LABEL_COLS), len(PLANES), pretrained=False).to(DEV)
    model.load_state_dict({**model.state_dict(), **backbone_state})
    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    best, best_state, stale, hist = -1., None, 0, []
    for ep in range(EPOCHS):
        model.train(); t0, total = time.time(), 0.
        order = np.random.permutation(tr_rows)
        for b in range(0, len(order), BATCH):
            r = order[b:b + BATCH]
            feature, mask = torch.from_numpy(tokens[r]).to(DEV), torch.from_numpy(masks[r]).to(DEV)
            y, w = torch.from_numpy(y_all[r]).to(DEV), torch.from_numpy(w_all[r]).to(DEV)
            loss = masked_bce(model.head(feature, mask).float(), y, w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, CLIP); opt.step()
            total += loss.item() * len(r)
        pv = predict_tokens(model, va_rows)
        auc, _ = macro_auc(y_all[va_rows], pv, LABEL_COLS)
        hist.append(dict(fold=fold, epoch=ep, loss=total / len(tr_rows), auc_pseudo=auc, minutes=(time.time() - t0) / 60))
        print(hist[-1])
        # Pseudo-label validation AUC selects the epoch; gold is never consulted.
        if auc > best + 2e-4:
            best, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= RAD_PATIENCE: break
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state, cfg=recipe, epoch=int(np.argmax([h['auc_pseudo'] for h in hist])),
                    fold=fold, auc_pseudo=best), os.path.join(OUT, f'fold_{fold}.pt'))
    pd.DataFrame(hist).to_csv(os.path.join(OUT, f'history_fold_{fold}.csv'), index=False)
    oof = pd.DataFrame(predict_tokens(model, va_rows), columns=LABEL_COLS)
    oof.insert(0, 'StudyInstanceUID', df.iloc[va_rows]['StudyInstanceUID'].to_numpy()); oof['fold'] = fold
    oof.to_csv(os.path.join(OUT, f'oof_fold_{fold}.csv'), index=False)
    gold_rows = np.flatnonzero(folds == -1)
    pg = predict_tokens(model, gold_rows)
    gold_output = pd.DataFrame(pg, columns=LABEL_COLS)
    gold_output.insert(0, 'StudyInstanceUID', df.iloc[gold_rows]['StudyInstanceUID'].to_numpy())
    gold_output.to_csv(os.path.join(OUT, f'gold_fold_{fold}.csv'), index=False)
    print('Gold audit' + (' (IN-SAMPLE: gold trained at weight %g)' % GOLD_WEIGHT if GOLD_WEIGHT > 0 else '') + ':',
          macro_auc(df.iloc[gold_rows][LABEL_COLS].to_numpy(), pg, LABEL_COLS)[0])
    del model, opt
    gc.collect()
    if DEV == 'cuda': torch.cuda.empty_cache()

paths = [os.path.join(OUT, f'oof_fold_{f}.csv') for f in range(N_FOLDS)]
if all(os.path.exists(p) for p in paths):
    oof = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = df.loc[folds >= 0, 'StudyInstanceUID']
    aligned = oof.set_index('StudyInstanceUID').loc[expected]
    oof.to_csv(os.path.join(OUT, 'oof_predictions.csv'), index=False)
    print('OOF pseudo AUC (noisy-label proxy, selection-biased):',
          macro_auc(df.loc[folds >= 0, LABEL_COLS].to_numpy(), aligned[LABEL_COLS].to_numpy(), LABEL_COLS))
