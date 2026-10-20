#!/bin/zsh
# Own v5 ensemble with a third backbone: ConvNeXt-T (Z5) + ConvNeXt-B (B5) + CAFormer (W5, 7 epochs), folds 0-1 each; compare T+B = 0.940.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
for i in $(seq 1 20); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-v5-tbc RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1,knee-session-b5-student-cnxb-v5-f0-f1,knee-session-w5-student-caf-v5-f0-f1 2>&1 | tail -1); echo "$(date -u +%H:%M) $out" | cut -c1-120; [[ "$out" == *successfully* ]] && break; sleep 300; done
sleep 120
while true; do s=$(st knee-submit-v5-tbc); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel $s"; exit 1;; esac; sleep 120; done
T $K competitions submit $C -k shreyanshgoyaliitb/knee-submit-v5-tbc -v 1 -f submission.csv \
  -m "Own models only: ConvNeXt-T + ConvNeXt-B + CAFormer-S18 (7 epochs) noisy students on knee-2c-v5, folds 0-1 each; compare T+B = 0.940" 2>&1 | tail -1
while true; do sub=$(T $K competitions submissions $C --csv 2>/dev/null | sed -n 2p); if [ -n "$sub" ] && ! echo "$sub" | grep -q PENDING; then echo "$(date -u +%H:%M) T+B+CAF: $(echo "$sub" | awk -F, '{print $(NF-1)}')"; break; fi; sleep 300; done
