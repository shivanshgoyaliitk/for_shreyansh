#!/bin/zsh
# A/B of test-time window subsets (05l) on ConvNeXt-T + ConvNeXt-B v5 (Z5 + B5, folds 0-1 each); compare 05i = 0.940.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
for i in $(seq 1 20); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-v5-tb-tta RUN_ON_KAGGLE/05l_submit_own_students_tta.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1,knee-session-b5-student-cnxb-v5-f0-f1 2>&1 | tail -1); echo "$(date -u +%H:%M) $out" | cut -c1-120; [[ "$out" == *successfully* ]] && break; sleep 300; done
sleep 120
while true; do s=$(st knee-submit-v5-tb-tta); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel $s"; exit 1;; esac; sleep 120; done
T $K competitions submit $C -k shreyanshgoyaliitb/knee-submit-v5-tb-tta -v 1 -f submission.csv \
  -m "A/B: ConvNeXt-T + ConvNeXt-B v5 students (f0-1 each) with test-time window subsets (half all-window pass, half mean of 4 random 6-window passes); compare without = 0.940" 2>&1 | tail -1
echo "$(date -u +%H:%M) TTA A/B submitted"
