"""TPU session runner: jobs run one after another, each across every TPU chip (tpu_runtime.py).

Expects TPU_JOBS and TPU_JOB_SOURCE from earlier cells. A job is (preset, folds) or
(preset, folds, seed); outputs land in one folder per job, <preset> or <preset>_s<seed>, exactly
as the GPU sessions write them, so the submission notebook attaches them the same way.
"""
import os, sys, json, time, subprocess
from pathlib import Path

WORK = Path(os.environ.get('KNEE_WORK', '/kaggle/working'))
jobs_dir = WORK / 'jobs'; jobs_dir.mkdir(parents=True, exist_ok=True)
script = jobs_dir / 'tpu_job.py'
script.write_text(TPU_JOB_SOURCE)
WORKERS = max(1, min(32, (os.cpu_count() or 8) - 4))     # loader workers of the single SPMD process
SESSION_START = time.time()                                 # the jobs' time guard counts from here
KEYWORDS = ('TPU runtime', 'recipe:', 'label readers', 'teacher', "'epoch'", 'Gold audit', '[xla]',
            '[time guard]', 'Traceback', 'Error', 'error:', 'Killed')
results = {}
for job in TPU_JOBS:
    preset, folds, *rest = job
    seed = rest[0] if rest else None
    run_name = preset if seed in (None, 42) else f'{preset}_s{seed}'
    out = WORK / run_name; out.mkdir(exist_ok=True)
    env = dict(os.environ, KNEE_PRESET=preset, KNEE_FOLDS=json.dumps(folds), KNEE_OUT=str(out),
               KNEE_RUN_NAME=run_name, KNEE_ALLOW_CPU='1', KNEE_TPU_WORKERS=str(WORKERS),
               KNEE_SESSION_START=str(SESSION_START),
               TQDM_DISABLE='1', PYTHONUNBUFFERED='1')
    if seed is not None:
        env['KNEE_SEED'] = str(seed)
    started = time.time()
    name = f'tpu_{run_name}_f{"".join(map(str, folds))}'
    with open(WORK / f'{name}.log', 'w') as log:
        proc = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in KEYWORDS):
                print(f'[{name}] {line.rstrip()[:400]}', flush=True)
        code = proc.wait()
    results[name] = dict(returncode=code, minutes=round((time.time() - started) / 60, 1))
    print(f'[{name}] finished with code {code} after {results[name]["minutes"]} min', flush=True)
    if code != 0:
        break
(WORK / 'session_results.json').write_text(json.dumps(results, indent=2))
failed = {k: v for k, v in results.items() if v['returncode'] != 0}
assert not failed and len(results) == len(TPU_JOBS), f'TPU jobs failed or were skipped: {failed or results}'
print('TPU SESSION DONE', results)
