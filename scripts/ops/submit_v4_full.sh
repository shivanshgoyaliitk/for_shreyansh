#!/bin/zsh
# After the TPU v4 job completes: push the full v4 ConvNeXt-T submission (session Z f0-1 + TPU f2-4 + seed 7 f0-4) and submit it.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do s=$(st knee-tpu-student-v4); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "TPU v4 $s"; exit 1;; esac; sleep 600; done
echo "$(date -u +%H:%M) TPU v4 complete"
for i in $(seq 1 30); do
  out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-student-v4-full RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z-student-v4-f0-f1,knee-tpu-student-v4 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $out"; [[ "$out" == *successfully* ]] && break; sleep 300
done
sleep 120
while true; do s=$(st knee-submit-student-v4-full); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "submit kernel $s"; exit 1;; esac; sleep 240; done
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-student-v4-full -v 1 -f submission.csv \
  -m "Own models only: ConvNeXt-T noisy students on knee-2c-v4, folds 0-4 + seed 7 folds 0-4 (10 models; folds 2-4 and seed 7 trained on TPU)" 2>&1 | tail -1
echo "$(date -u +%H:%M) submitted"
