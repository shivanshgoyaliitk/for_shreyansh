#!/bin/zsh
# After session Z completes: push the v4 A/B submission kernel (05i, Z only) and submit it. Exits on any failure.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do s=$(st knee-session-z-student-v4-f0-f1); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "Z $s"; exit 1;; esac; sleep 300; done
echo "$(date -u +%H:%M) Z complete"
for i in $(seq 1 30); do
  out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-student-v4 RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z-student-v4-f0-f1 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $out"; [[ "$out" == *successfully* ]] && break; sleep 300
done
sleep 120
while true; do s=$(st knee-submit-student-v4); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "submit kernel $s"; exit 1;; esac; sleep 240; done
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-student-v4 -v 1 -f submission.csv \
  -m "A/B: ConvNeXt noisy students on knee-2c-v4 (two 90 mm local slots around a header-geometry knee-joint anchor), folds 0-1 only; compare v3 students folds 0-1 = 0.935" 2>&1 | tail -1
echo "$(date -u +%H:%M) submitted"
