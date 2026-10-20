"""Notebook cache fill cell; shared source is embedded by build_notebooks.py."""
# A recipe too large for one Kaggle output is built as shards: shard k holds every
# n-th study. Each shard is a separate notebook run; training attaches all of them.
# PIECES > 1 writes each shard as that many ~1 GB cache files, each a complete shard of
# its own (cache.npy, cache_index.csv, cache_meta.json): Kaggle could not mount the
# 17 GiB single-file shards as inputs (2026-10-07), and training discovers pieces as shards.
N_SHARDS = int(globals().get('N_SHARDS', 1))
SHARD = int(globals().get('SHARD', 0))
PIECES = int(globals().get('PIECES', 1))
assert 0 <= SHARD < N_SHARDS and PIECES >= 1
manifest = manifest.iloc[SHARD::N_SHARDS].reset_index(drop=True)
N = len(manifest)
SHAPE = (N, len(SLOTS), N_SLICES, IMG, IMG)
meta = preprocessing_meta()
if N_SHARDS > 1:
    meta = dict(meta, shard=SHARD, n_shards=N_SHARDS)
fingerprint = cache_fingerprint(manifest, meta if PIECES == 1 else dict(meta, pieces=PIECES))
PROGRESS = os.path.join(WORK, 'cache_progress.npz')
IDENTITY = os.path.join(WORK, 'cache_identity.json')
assert np.prod(SHAPE) < 19e9, 'cache exceeds Kaggle output budget'
assert N >= PIECES
BOUNDS = np.linspace(0, N, PIECES + 1).astype(int)
PIECE_OF = np.searchsorted(BOUNDS, np.arange(N), side='right') - 1


def piece_file(j, name):
    folder = WORK if PIECES == 1 else os.path.join(WORK, f'cache_piece_{SHARD * PIECES + j:02d}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def piece_shape(j):
    return (int(BOUNDS[j + 1] - BOUNDS[j]),) + SHAPE[1:]


if os.path.exists(piece_file(0, 'cache.npy')):
    assert os.path.exists(IDENTITY), 'Unversioned cache: use an empty output directory to rebuild'
    assert json.load(open(IDENTITY))['fingerprint'] == fingerprint, 'Cache recipe or row order changed: rebuild'
    for j in range(PIECES):
        check = np.load(piece_file(j, 'cache.npy'), mmap_mode='r')
        assert check.shape == piece_shape(j) and check.dtype == np.uint8
        del check
else:
    for j in range(PIECES):
        np.lib.format.open_memmap(piece_file(j, 'cache.npy'), mode='w+', dtype=np.uint8, shape=piece_shape(j)).flush()
    with open(IDENTITY, 'w') as handle: json.dump(dict(fingerprint=fingerprint), handle)

done, present = np.zeros(N, bool), np.zeros((N, len(SLOTS)), bool)
if os.path.exists(PROGRESS):
    with np.load(PROGRESS) as saved:
        assert str(saved['fingerprint']) == fingerprint
        done, present = saved['done'].copy(), saved['present'].copy()
    assert done.shape == (N,) and present.shape == (N, len(SLOTS))

def save_progress():
    # Done and presence commit together; write-then-rename leaves the old state intact on interruption.
    temp = PROGRESS + '.tmp'
    with open(temp, 'wb') as handle:
        np.savez(handle, done=done, present=present, fingerprint=fingerprint)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, PROGRESS)

JOBS = [(row[[f'dir_{i}' for i in range(len(SLOTS))]].tolist(), row['study_side'], row_anchor(row))
        for _, row in manifest.iterrows()]

def fill_row(r):
    j = int(PIECE_OF[r]); local = r - int(BOUNDS[j])
    mm = np.lib.format.open_memmap(piece_file(j, 'cache.npy'), mode='r+')
    mm[local] = 0
    dirs, side, anchor = JOBS[r]
    filled = []
    for slot, directory in enumerate(dirs):
        if not directory: continue
        try:
            mm[local, slot] = load_series(directory, SLOTS[slot][1], side, **slot_kwargs(recipe(), slot, anchor))
            filled.append(slot)
        except Exception:
            continue
    mm.flush()  # pixels reach disk before the completion record
    del mm
    return r, filled

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
workers = min(4, os.cpu_count() or 1)
if 'fork' in mp.get_all_start_methods():
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'))
else:
    pool = ThreadPoolExecutor(max_workers=workers)
todo = np.flatnonzero(~done).tolist()
print(f'{done.sum()}/{N} already complete')
try:
    with pool as ex:
        for k, (r, filled) in enumerate(tqdm(ex.map(fill_row, todo), total=len(todo)), 1):
            present[r] = False
            present[r, filled] = True
            done[r] = True
            if k % 50 == 0: save_progress()
finally:
    save_progress()

index = manifest[['StudyInstanceUID', 'study_side', 'side_src']].copy()
index['shard'] = SHARD
for i in range(len(SLOTS)): index[f'present_{i}'] = present[:, i]
index['row'] = np.arange(N)
index['n_present'] = present.sum(1)
for j in range(PIECES):
    lo, hi = int(BOUNDS[j]), int(BOUNDS[j + 1])
    part = index.iloc[lo:hi].copy()
    piece_meta = meta
    if PIECES > 1:
        part['shard'], part['row'] = SHARD * PIECES + j, np.arange(hi - lo)
        piece_meta = dict(meta, shard=SHARD * PIECES + j, n_shards=N_SHARDS * PIECES)
    part.to_csv(piece_file(j, 'cache_index.csv'), index=False)
    # Metadata last and atomically: a piece is discoverable only once pixels and index exist.
    temp = piece_file(j, 'cache_meta.json.tmp')
    with open(temp, 'w') as handle: json.dump(piece_meta, handle, indent=2)
    os.replace(temp, piece_file(j, 'cache_meta.json'))
if PIECES > 1:
    print(f'Wrote {PIECES} pieces: shards {SHARD * PIECES}..{SHARD * PIECES + PIECES - 1} of {N_SHARDS * PIECES}, '
          f'{np.prod(piece_shape(0)) / 1e9:.2f} GB each')
print('Decoded slots per study:', index['n_present'].value_counts().sort_index().to_dict())
print('Studies with each slot:', dict(zip(meta['slots'], present.sum(0).tolist())))
