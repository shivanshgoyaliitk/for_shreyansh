"""Two-GPU training session: one queue of jobs per GPU, each job its own process.

Embedded in the session notebooks. Kaggle counts GPU quota per session hour, and its T4
sessions come with two GPUs, so two independent trainings run side by side for the
price of one. These are separate processes pinned with CUDA_VISIBLE_DEVICES, not
DataParallel: every job is exactly the single-GPU recipe of the per-fold notebooks.
Expects GPU_JOBS, RESUME_DIR, TRAIN_JOB_SOURCE and RAD_JOB_SOURCE from earlier cells.
A job is (preset, folds) or (preset, folds, seed); a non-default seed trains another member
of the same family in its own folder, <preset>_s<seed>.
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path

WORK = Path(globals().get('WORK', '/kaggle/working'))
POLL_SECONDS = globals().get('POLL_SECONDS', 120)
WORKERS_PER_JOB = globals().get('WORKERS_PER_JOB', 2)      # 4 CPU cores shared by two jobs
REQUIRE_GPUS = globals().get('REQUIRE_GPUS', True)
KEYWORDS = ('recipe:', 'label readers', 'trainable', "'epoch'", 'Gold audit', 'OOF pseudo',
            'encoded', 'Error', 'Traceback', 'Killed')

jobs_dir, logs_dir = WORK / '_jobs', WORK / 'logs'
jobs_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(exist_ok=True)
(jobs_dir / 'train_job.py').write_text(TRAIN_JOB_SOURCE)
(jobs_dir / 'rad_job.py').write_text(RAD_JOB_SOURCE)
if REQUIRE_GPUS:
    import torch
    assert torch.cuda.device_count() >= len(GPU_JOBS), \
        f'This session runs {len(GPU_JOBS)} GPU queues: set Accelerator to GPU T4 x2 (found {torch.cuda.device_count()})'
def job_parts(job):
    preset, folds, *rest = job
    seed = rest[0] if rest else None
    return preset, folds, seed, (preset if seed is None else f'{preset}_s{seed}')


seen = [(name, f) for queue in GPU_JOBS.values() for job in queue
        for _, folds, _, name in [job_parts(job)] for f in folds]
assert len(seen) == len(set(seen)), f'A family-fold appears twice in GPU_JOBS: {GPU_JOBS}'

results = {}
low_ram = {}


def available_ram_gb():
    # Cache pages of the memory-mapped shards are reclaimable and counted as available;
    # this tracks what the jobs themselves hold. None where /proc is missing (local tests).
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    except OSError:
        return None


def run_queue(gpu, queue):
    for job in queue:
        preset, folds, seed, run_name = job_parts(job)
        out = WORK / run_name
        out.mkdir(exist_ok=True)
        name = f"gpu{gpu}_{run_name}_f{''.join(map(str, folds))}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), KNEE_PRESET=preset, KNEE_RUN_NAME=run_name,
                   KNEE_FOLDS=json.dumps(folds), KNEE_OUT=str(out), KNEE_WORKERS=str(WORKERS_PER_JOB),
                   KNEE_RESUME_DIR=RESUME_DIR, TQDM_DISABLE='1', PYTHONUNBUFFERED='1')
        if seed is not None:
            env['KNEE_SEED'] = str(seed)
        script = jobs_dir / ('rad_job.py' if preset == 'radimagenet' else 'train_job.py')
        started = time.time()
        with open(logs_dir / f'{name}.log', 'w') as handle:
            code = subprocess.run([sys.executable, str(script)], env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        results[name] = dict(returncode=code, minutes=round((time.time() - started) / 60, 1))
        print(f'[{name}] finished with code {code} after {results[name]["minutes"]} min', flush=True)
        if code != 0:
            break       # later jobs on this GPU are skipped; the failure is reported below


def show_new_lines(offsets):
    for log in sorted(logs_dir.glob('*.log')):
        lines = log.read_text(errors='replace').splitlines()
        for line in lines[offsets.get(log, 0):]:
            if any(k in line for k in KEYWORDS):
                print(f'[{log.stem}] {line[:400]}', flush=True)
        offsets[log] = len(lines)


session_start = time.time()
threads = [threading.Thread(target=run_queue, args=(gpu, queue), daemon=True) for gpu, queue in GPU_JOBS.items()]
for thread in threads: thread.start()
offsets = {}
while any(thread.is_alive() for thread in threads):
    time.sleep(POLL_SECONDS)
    show_new_lines(offsets)
    ram = available_ram_gb()
    if ram is not None and ram < low_ram.get('gb', float('inf')):
        low_ram.update(gb=round(ram, 1), minute=round((time.time() - session_start) / 60))
        print(f'[ram] lowest available so far: {ram:.1f} GB', flush=True)
for thread in threads: thread.join()
show_new_lines(offsets)
print(json.dumps(results, indent=2))
if low_ram:
    print(f"Lowest available RAM: {low_ram['gb']} GB (minute {low_ram['minute']})")
(WORK / 'session_results.json').write_text(json.dumps(dict(jobs=results, lowest_available_ram=low_ram), indent=2))
expected = sum(len(queue) for queue in GPU_JOBS.values())
failed = {k: v for k, v in results.items() if v['returncode'] != 0}
for name in failed:
    print(f'\n===== tail of {name} =====\n' + '\n'.join((logs_dir / f'{name}.log').read_text(errors='replace').splitlines()[-40:]))
assert not failed and len(results) == expected, f'Jobs failed or were skipped: {failed or "see above"}'
print('All jobs finished. Outputs: ' + ', '.join(sorted({job_parts(j)[3] for q in GPU_JOBS.values() for j in q})))
