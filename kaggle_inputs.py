"""Bounded discovery of attached Kaggle artifacts; embedded in launch notebooks."""
import fnmatch, os
from pathlib import Path


DICOM_DIRS = ('train_series', 'test_series')   # competition image trees: never listed


def attached_files(name, root='/kaggle/input', max_depth=5):
    # Kaggle mounts datasets and notebook outputs at different depths; artifacts live in
    # folders 1-5 levels below the root. The DICOM trees are pruned: listing them took
    # ~2 min on local disk and over an hour on the TPU VM's network-mounted inputs.
    root = Path(root)
    found = set()
    for path, dirs, files in os.walk(root):
        depth = len(Path(path).relative_to(root).parts)
        dirs[:] = [d for d in dirs if d not in DICOM_DIRS] if depth < max_depth else []
        if depth >= 1:
            found.update(Path(path) / f for f in fnmatch.filter(files, name) if not f.startswith('.'))
    return sorted(found)


def one_input(name, override='', root='/kaggle/input'):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f'Configured input does not exist: {path}')
        return str(path)
    matches = attached_files(name, root)
    if len(matches) != 1:
        raise RuntimeError(f'Attach exactly one intended {name} input or set its path explicitly. '
                           f'Found: {[str(p) for p in matches]}')
    print(f'{name}: {matches[0]}')
    return str(matches[0])


def fold_inputs(expected, root='/kaggle/input', per_fold=1):
    # per_fold > 1 when several training runs (e.g. DINOv2 and DINOv3 arms) are attached.
    paths = []
    for fold in expected:
        matches = attached_files(f'fold_{fold}.pt', root)
        if len(matches) != per_fold:
            raise RuntimeError(f'Attach exactly {per_fold} fold_{fold}.pt input(s) '
                               f'({"exactly one" if per_fold == 1 else "one per training run"}). '
                               f'Found: {[str(p) for p in matches]}')
        paths.extend(str(p) for p in matches)
    for path in paths: print(f'checkpoint: {path}')
    return paths


def checkpoint_inputs(root='/kaggle/input'):
    # Every attached fold_N.pt is an ensemble member; families are resolved from each
    # checkpoint's recipe, and duplicates of one family-fold are rejected by the caller.
    paths = [str(p) for p in attached_files('fold_*.pt', root)]
    if not paths:
        raise RuntimeError('Attach at least one training output containing fold_N.pt')
    for path in paths: print(f'checkpoint: {path}')
    return paths
