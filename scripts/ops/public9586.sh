#!/bin/zsh
# Our copy of heliosli's public 0.9586 notebook (unchanged): wait for the commit run, submit, wait for the score.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
while true; do s=$(T $K kernels status shreyanshgoyaliitb/knee-public-9586 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) knee-public-9586 $s"; exit 1;; esac; sleep 300; done
echo "$(date -u +%H:%M) knee-public-9586 complete"
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-public-9586 -v 2 -f submission.csv \
  -m "Public: heliosli 'best public dataset new version 0.9586' run unchanged except the R6 weights mount (rostigor copy, manifest sha256 pinned) (Raptor 384 + 224, R6 compact with flip TTA, goodpjw2008 2.5D reader)" 2>&1 | tail -1
scripts/ops/wait_scores.sh "best public dataset new version 0.9586" "all five folds each (f2-4 on TPU)"
