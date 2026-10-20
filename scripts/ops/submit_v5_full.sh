#!/bin/zsh
# When the TPU v5 job (folds 2-4) completes: push the full v5 ConvNeXt-T submission (Z5 f0-1 + TPU f2-4) and submit it.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do s=$(st knee-tpu-student-v5-f234); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "TPU v5 $s"; exit 1;; esac; sleep 900; done
echo "$(date -u +%H:%M) TPU v5 f2-4 complete"
for i in $(seq 1 30); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-student-v5-full RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1,knee-tpu-student-v5-f234 2>&1 | tail -1); echo "$(date -u +%H:%M) $out" | cut -c1-120; [[ "$out" == *successfully* ]] && break; sleep 300; done
sleep 120
while true; do s=$(st knee-submit-student-v5-full); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel $s"; exit 1;; esac; sleep 240; done
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-student-v5-full -v 1 -f submission.csv \
  -m "Own models only: ConvNeXt-T noisy students on knee-2c-v5 (288 px, 10 windows, anchored local slots), folds 0-4 (f2-4 on TPU)" 2>&1 | tail -1
echo "$(date -u +%H:%M) v5 full submitted"
