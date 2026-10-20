# Kaggle execution — 2026-09-18

- Cache v2 launched in shreyanshgoyaliitb/knee-mri-step2c-cache at 17:10 SGT, CPU, Internet on, pinned July 1 environment. Old v1 remains available.
- Consensus label output: shreyanshgoyaliitb/compi-label-generation, run 348754261; saved run successful. Locally checked train_targets.csv: 4,407 unique studies, 4,349 pseudo + 58 gold, 99.9036% observed cells, two entirely unobserved studies (training filters them).
- Training default changed to this consensus output instead of older pseudo-label dataset. Prepared a single-fold first run, six epochs; complete remaining folds after observing runtime and memory.

- Cache run URL: https://www.kaggle.com/code/shreyanshgoyaliitb/knee-mri-step2c-cache?scriptVersionId=350787060. Logs passed header scan at 157.8s; laterality sources coords=2213/tag=2183/default=11; fill began at 161.4s. No error at last check.
- Training editor imported runs/2026-09-18/training-fold0.ipynb, with KNEE_FOLDS=[0]. Consensus and cache outputs attached. Cache attachment currently exposes cache_done.npy (old v1); use input menu Check for updates AFTER cache v2 completes. Do not start training against v1.
- Training GPU setting was already T4 x2; Internet on; environment pinned July 1. UI quota reads 00:00 /30 hrs, interpretation not yet verified.
- Created private draft https://www.kaggle.com/code/shreyanshgoyaliitb/knee-gpu-preflight/edit. It is still the default empty notebook; intended source is runs/2026-09-18/gpu-preflight.ipynb, not yet imported or run. Do not claim benchmark results.
- Local label fold check after excluding the two entirely unobserved rows: fold sizes 869/871/870/869/868, gold=58; both classes present for all twelve findings in every fold.
- Local tests rerun successfully: 12 passed.
- Native Chrome control became unreliable: clicks often leave the UI unchanged, paste once timed out, and browser provider chrome is unavailable (only native app control works). Asked user to bring Chrome to foreground and leave it briefly. No Kaggle CLI or configured local credentials exist; do not extract browser session credentials.
