#!/bin/zsh
# ConvNeXt-B v4 A/B: when the TPU job (folds 0-1) completes, push and submit its 05i kernel.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do s=$(st knee-tpu-student-cnxb-v4-f01); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "TPU $s"; exit 1;; esac; sleep 900; done
echo "$(date -u +%H:%M) TPU ConvNeXt-B v4 complete"
for i in $(seq 1 30); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-student-cnxb-v4 RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-tpu-student-cnxb-v4-f01 2>&1 | tail -1); echo "$(date -u +%H:%M) $out" | cut -c1-120; [[ "$out" == *successfully* ]] && break; sleep 300; done
sleep 120
while true; do s=$(st knee-submit-student-cnxb-v4); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel $s"; exit 1;; esac; sleep 240; done
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-student-cnxb-v4 -v 1 -f submission.csv \
  -m "A/B: ConvNeXt-Base noisy students on knee-2c-v4, folds 0-1 only (TPU); compare ConvNeXt-T v4 folds 0-1 = 0.938" 2>&1 | tail -1
echo "$(date -u +%H:%M) ConvNeXt-B v4 A/B submitted"
