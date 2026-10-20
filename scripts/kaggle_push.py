"""Push a generated notebook to Kaggle as a kernel and start its run (Kaggle CLI, ~/.kaggle token).

    python scripts/kaggle_push.py SLUG NOTEBOOK.ipynb [--gpu | --tpu] [--internet] [--kernels a,b] [--datasets a,b]
                                  [--competition] [--title TITLE] [--dry-run]

SLUG is the kernel slug under our account; NOTEBOOK is a file from RUN_ON_KAGGLE/. Inputs are given as
'owner/slug' (owner defaults to ours). The Docker image is pinned to the one our sessions and
submissions already use, so training and inference keep the same package versions. TPU kernels
(TPU v5e-8) keep Kaggle's TPU image, which ships torch_xla; the pinned image is a GPU image.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'shreyanshgoyaliitb'
COMPETITION = 'rsna-knee-abnormality-detection'
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
LABEL_DATASETS = [f'{OWNER}/knee-reader-labels', 'laymond/rsna-knee-abnormality-qwen3-8b-weak-labels',
                  'stevenleehans/rsna-knee-llm-report-labels', 'lixin73/rsna-knee-llm-report-labels-sol56',
                  'pilkwang/rsna-knee-llm-labels']
CACHE_DATASETS = [f'{OWNER}/knee-cache-v3-shard0', f'{OWNER}/knee-cache-v3-shard1']


def full(ref):
    return ref if '/' in ref else f'{OWNER}/{ref}'


def metadata(slug, code_file, title, gpu, internet, kernels, datasets, competition, tpu=False):
    assert not (gpu and tpu), 'choose one accelerator'
    meta = {'id': f'{OWNER}/{slug}', 'title': title or slug, 'code_file': code_file, 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': gpu, 'enable_tpu': tpu,
            'enable_internet': internet, 'dataset_sources': [full(d) for d in datasets],
            'competition_sources': [COMPETITION] if competition else [],
            'kernel_sources': [full(k) for k in kernels], 'model_sources': [],
            'docker_image': DOCKER, 'machine_shape': 'NvidiaTeslaT4' if gpu else ''}
    if tpu:
        del meta['docker_image']
        meta['machine_shape'] = 'TpuV5E8'
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('slug'); ap.add_argument('notebook')
    ap.add_argument('--title'); ap.add_argument('--gpu', action='store_true'); ap.add_argument('--tpu', action='store_true')
    ap.add_argument('--internet', action='store_true')
    ap.add_argument('--kernels', default=''); ap.add_argument('--datasets', default='')
    ap.add_argument('--competition', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    nb = Path(a.notebook)
    assert nb.suffix == '.ipynb' and nb.exists(), nb
    split = lambda s: [x.strip() for x in s.split(',') if x.strip()]
    meta = metadata(a.slug, nb.name, a.title, a.gpu, a.internet, split(a.kernels), split(a.datasets), a.competition, a.tpu)
    work = Path(tempfile.mkdtemp(prefix='kpush-'))
    shutil.copy2(nb, work / nb.name)
    (work / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    if a.dry_run:
        return 0
    kaggle = str(Path(sys.executable).with_name('kaggle'))
    return subprocess.call([kaggle, 'kernels', 'push', '-p', str(work)])


if __name__ == '__main__':
    sys.exit(main())
