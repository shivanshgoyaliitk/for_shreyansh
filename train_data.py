# ---- cache shards -------------------------------------------------------------
# Rows are addressed as (shard, row in shard); every shard shares one recipe.
parts = []
for shard_id, directory in enumerate(CACHE_DIRS):
    part = pd.read_csv(os.path.join(directory, "cache_index.csv"))
    shape = np.load(os.path.join(directory, "cache.npy"), mmap_mode="r").shape
    assert shape[1:] == (len(PLANES), N_SLICES, CACHE_IMG, CACHE_IMG), f'{directory}: cache shape {shape}'
    assert len(part) == shape[0] and part['row'].between(0, shape[0] - 1).all()
    parts.append(part.assign(cache_id=shard_id))
    print(f"cache shard {shard_id}: {shape[0]:,} studies, {np.prod(shape)/1e9:.1f} GB  ({directory})")
index = pd.concat(parts, ignore_index=True)
assert index['StudyInstanceUID'].is_unique, 'a study appears in two cache shards'
T_TRAIN = min(T_TRAIN, N_SLICES)

# ---- targets: report-label readers + gold ------------------------------------
train_csv = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'), dtype={'StudyInstanceUID': str})
LABEL_TABLES = find_label_tables(LABEL_ROOT)
print('label readers:', list(LABEL_TABLES))
_missing = [r for r in REQUIRED_READERS if not any(fnmatch.fnmatch(name, r) for name in LABEL_TABLES)]
assert not _missing, f'Attach the label inputs providing {_missing} (see KAGGLE_RUN/README.md)'
targets, label_report = build_targets(train_csv, LABEL_COLS, LABEL_TABLES, READER_WEIGHTS, GOLD_WEIGHT)
print(pd.DataFrame(label_report).T.round(3).to_string())
print('(gold AUC of each reader is a diagnostic only; it chooses nothing. NaN: the reader has no gold rows)')
TEACHER_SUMMARY = None
if np.max(TEACHER_WEIGHT) > 0:
    _teacher = find_teacher(LABEL_ROOT)
    if _teacher is None:
        # No teacher table attached: build it here from the attached reader session outputs
        # (their out-of-fold predictions), exactly as 07_teacher_targets would.
        _gold = set(targets.loc[targets['source'].eq('gold'), 'StudyInstanceUID'])
        _frame, _meta = build_teacher(LABEL_ROOT, LABEL_COLS, gold_ids=_gold, families=globals().get('TEACHER_FAMILY_WEIGHTS'),
                                      quantile_map=globals().get('TEACHER_QUANTILE_MAP', ()))
        print('teacher built in session from:', json.dumps(_meta['members']))
        _absent = sorted(f for f, w in (globals().get('TEACHER_FAMILY_WEIGHTS') or {}).items() if w > 0 and f not in _meta['families'])
        assert not _absent, f'Attach the teacher inputs for families {_absent}'
        os.makedirs(OUT, exist_ok=True)
        _teacher = os.path.join(OUT, 'teacher_targets_used.csv')
        write_atomic(_teacher, lambda p: _frame.to_csv(p, index=False))
    targets, TEACHER_SUMMARY = apply_teacher(targets, _teacher, LABEL_COLS, TEACHER_WEIGHT)
    print('teacher targets:', TEACHER_SUMMARY)
write_atomic(os.path.join(OUT, 'targets_used.csv'), lambda p: targets.to_csv(p, index=False))

WEIGHT_COLS = [f'w_{c}' for c in LABEL_COLS]
df = index[index["n_present"] > 0].merge(targets, on="StudyInstanceUID", how="inner")
df = df[df[LABEL_COLS].notna().any(axis=1)].reset_index(drop=True)
print(f"\nusable studies: {len(df):,}  ({df['source'].value_counts().to_dict()})")
# Report-embedding auxiliary targets (training only): NaN rows have no embedding and are masked.
REP_COLS, AUX_DIM, REPORT_AUX_SUMMARY = [], 0, None
if globals().get('REPORT_AUX', 0.0) > 0:
    _rep_path = one_input('report_embeddings.csv', root=LABEL_ROOT)
    _rep = pd.read_csv(_rep_path, dtype={'StudyInstanceUID': str})
    REP_COLS = [c for c in _rep.columns if c.startswith('rep_')]
    assert REP_COLS and _rep['StudyInstanceUID'].is_unique and not set(REP_COLS) & set(df.columns)
    df = df.merge(_rep[['StudyInstanceUID', *REP_COLS]], on='StudyInstanceUID', how='left')
    AUX_DIM = len(REP_COLS)
    import hashlib as _hashlib
    REPORT_AUX_SUMMARY = dict(weight=REPORT_AUX, dims=AUX_DIM,
                              studies=int(df[REP_COLS[0]].notna().sum()),
                              sha256=_hashlib.sha256(open(_rep_path, 'rb').read()).hexdigest())
    print('report auxiliary targets:', REPORT_AUX_SUMMARY)
y_all = df[LABEL_COLS].to_numpy(np.float32)
print(f"observed label cells: {(~np.isnan(y_all)).mean():.1%}")
print("\nmean soft target (observed cells):")
print(pd.Series(np.nanmean(y_all[df['source'].eq('pseudo')], axis=0), index=LABEL_COLS).round(3).to_string())
assert df['source'].isin(['gold','pseudo']).all(), 'Unknown label source'
assert ((np.isnan(y_all)) | ((y_all>=0)&(y_all<=1))).all(), 'Targets outside [0,1]'
gold_check = train_csv.set_index('StudyInstanceUID')[LABEL_COLS].apply(pd.to_numeric, errors='coerce')
gold_ids = df.loc[df['source'].eq('gold'), 'StudyInstanceUID']
assert np.array_equal(df.loc[df['source'].eq('gold'), LABEL_COLS].to_numpy(float),
                      gold_check.loc[gold_ids].to_numpy(float)), 'Gold rows must carry the official labels'
