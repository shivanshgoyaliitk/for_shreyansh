# Out-of-fold predictions of heliosli's compact model on the training studies (teacher for our students).
# compact_train.py assigns pool study u to fold int(sha1(u)[:8], 16) % 4 and fold model k never trains on fold k
# (gold, blanked and review-holdout studies are in no training set), so the fold-k model's logits are out-of-fold
# for every training study of hash fold k. heliosli's compact_infer.py runs unchanged with --split train, once per
# fold on a folder holding only that fold's model (a quarter of the GPU work of the four-model ensemble); each GPU
# takes whole folds in turn. Gold studies are skipped (never teacher targets).
# Outputs: reader_compact/oof_fold_<k>.csv (probabilities; read by label_builder.build_teacher),
# compact_logits.csv (out-of-fold logits), worker logs and receipts.
import csv, hashlib, json, os, shutil, subprocess, sys, threading, time
import numpy as np, pandas as pd

deadline = T0 + DEADLINE_HOURS * 3600
labels = pd.read_csv(DATA / 'sample_submission.csv', nrows=0).columns[1:].tolist()
train = pd.read_csv(DATA / 'train.csv', dtype={'StudyInstanceUID': str})
gold = set(train.loc[train[labels].notna().all(axis=1), 'StudyInstanceUID'])
ids = [u for u in sorted(train['StudyInstanceUID'].unique()) if u not in gold][COMPACT_SHARD::COMPACT_SHARDS]
if COMPACT_LIMIT:
    ids = ids[:COMPACT_LIMIT]
K = len(manifest['models'])
assert manifest['models'] == [f'{ARM}_s0_f{k}.pt' for k in range(K)], manifest['models']
fold_of = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K

def dicom_files(uid):
    try:
        return sum(len(os.listdir(s.path)) for s in os.scandir(DATA / 'train_series' / uid) if s.is_dir())
    except OSError:
        return 0

n_gpu = max(1, len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines()))
print(f'compact OOF: shard {COMPACT_SHARD}/{COMPACT_SHARDS}, {len(ids)} studies, {n_gpu} GPUs', flush=True)
OOF_TMP = TMP / 'oof'
runs = []
for k in range(K):
    fold_ids = sorted((u for u in ids if fold_of(u) == k), key=lambda u: (-dicom_files(u), u))
    if not fold_ids:
        continue
    models_k = BUNDLE / f'fold{k}'
    models_k.mkdir(exist_ok=True)
    shutil.copy(BUNDLE / manifest['models'][k], models_k / manifest['models'][k])
    wdir = OOF_TMP / f'fold{k}'
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / 'ids.json').write_text(json.dumps(fold_ids[::2] + fold_ids[1::2]))   # big studies not back to back
    runs.append((k, models_k, wdir, len(fold_ids)))

def run_gpu(gi, my_runs, codes):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MALLOC_ARENA_MAX='2', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for k, models_k, wdir, n in my_runs:
        cmd = [sys.executable, '-u', str(SRC / WORKER), '--data', str(DATA), '--split', 'train', '--code', str(CODE),
               '--arm', ARM, '--assignment', str(wdir / 'ids.json'), '--output', str(wdir), '--amp', 'auto', '--workers', '3',
               '--deadline', str(deadline), '--models', str(models_k), *WORKER_ARGS, '--dump-logits', str(wdir / 'logits.json')]
        with open(wdir / 'log.txt', 'w', encoding='utf-8') as log:
            codes[k] = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

codes = {}
threads = [threading.Thread(target=run_gpu, args=(gi, runs[gi::n_gpu], codes)) for gi in range(n_gpu)]
for t in threads:
    t.start()
last = 0
while any(t.is_alive() for t in threads):
    time.sleep(10)
    if time.time() - last > 600:
        last = time.time()
        for k, _, wdir, n in runs:
            if (wdir / 'log.txt').exists():
                lines = [l for l in (wdir / 'log.txt').read_text(encoding='utf-8', errors='replace').splitlines() if l.startswith('{')]
                print(f'fold {k} ({n} studies):', lines[-1] if lines else '(starting)', flush=True)

logits, header = {}, None
for k, _, wdir, n in runs:
    if (wdir / 'log.txt').exists():
        shutil.copyfile(wdir / 'log.txt', WORK / f'compact_fold{k}_log.txt')
    if (wdir / 'receipt.json').exists():
        shutil.copyfile(wdir / 'receipt.json', WORK / f'compact_fold{k}_receipt.json')
    print(f'fold {k} exit {codes.get(k)}', flush=True)
    if codes.get(k) != 0 and (wdir / 'log.txt').exists():
        print((wdir / 'log.txt').read_text(encoding='utf-8', errors='replace')[-3000:], flush=True)
    if (wdir / 'logits.json').exists():
        for uid, z in json.loads((wdir / 'logits.json').read_text()).items():
            assert fold_of(uid) == k and np.shape(z) == (1, len(labels)), (uid, k, np.shape(z))
            logits[uid] = np.asarray(z[0], dtype=np.float64)
    if (wdir / 'predictions.csv').exists():
        with open(wdir / 'predictions.csv', encoding='utf-8') as fh:
            header = next(csv.reader(fh))[1:]
assert logits, 'no study was scored'
assert sorted(header) == sorted(labels), (header, labels)

rows = [(uid, fold_of(uid), z) for uid, z in sorted(logits.items())]
dump = pd.DataFrame({'StudyInstanceUID': [r[0] for r in rows], 'fold': [r[1] for r in rows],
                     **{c: [r[2][j] for r in rows] for j, c in enumerate(header)}})[['StudyInstanceUID', 'fold'] + labels]
dump.to_csv(WORK / 'compact_logits.csv', index=False)
oof = dump.copy()
oof[labels] = 1 / (1 + np.exp(-oof[labels].to_numpy(float)))
oof = oof[~oof['StudyInstanceUID'].isin(gold)]
(WORK / 'reader_compact').mkdir(exist_ok=True)
for k, part in oof.groupby('fold'):
    part.to_csv(WORK / 'reader_compact' / f'oof_fold_{k}.csv', index=False)
missing = sorted(set(ids) - set(logits))
(WORK / 'compact_oof_receipt.json').write_text(json.dumps({
    'shard': COMPACT_SHARD, 'shards': COMPACT_SHARDS, 'studies': len(ids), 'scored': len(logits), 'missing': len(missing),
    'gold_scored': len(gold & set(logits)), 'oof_rows': len(oof), 'hours': round((time.time() - T0) / 3600, 2),
    'bundle_id': BUNDLE_ID, 'arm': ARM}, indent=1))
print('COMPACT_OOF_DONE', json.dumps({'scored': len(logits), 'of': len(ids), 'oof_rows': len(oof),
                                      'per_fold': oof['fold'].value_counts().sort_index().tolist(),
                                      'mean_prob': oof[labels].mean().round(3).tolist()}), flush=True)
