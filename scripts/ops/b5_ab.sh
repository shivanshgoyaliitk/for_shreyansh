#!/bin/zsh
# B5 A/B (both folds 0-1, knee-2c-v5): ConvNeXt-B students alone, and ConvNeXt-B + ConvNeXt-T (Z5) students; compare Z5 alone = 0.939.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 30); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); echo "$(date -u +%H:%M) $1: $out" | cut -c1-120; [[ "$out" == *successfully* ]] && return 0; sleep 300; done; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-300}; done; }
NB=RUN_ON_KAGGLE/05i_submit_own_students.ipynb
push knee-submit-cnxb-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-b5-student-cnxb-v5-f0-f1 || exit 1
push knee-submit-v5-tb $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1,knee-session-b5-student-cnxb-v5-f0-f1 || exit 1
sleep 300
wait_done knee-submit-cnxb-v5 240 && T $K competitions submit $C -k shreyanshgoyaliitb/knee-submit-cnxb-v5 -v 1 -f submission.csv \
  -m "A/B: ConvNeXt-B noisy students on knee-2c-v5, folds 0-1 only; compare ConvNeXt-T v5 folds 0-1 = 0.939" 2>&1 | tail -1
wait_done knee-submit-v5-tb 240 && T $K competitions submit $C -k shreyanshgoyaliitb/knee-submit-v5-tb -v 1 -f submission.csv \
  -m "Own models only: ConvNeXt-T + ConvNeXt-B noisy students on knee-2c-v5, folds 0-1 each; compare ConvNeXt-T alone = 0.939" 2>&1 | tail -1
echo "$(date -u +%H:%M) B5 A/B submitted"
# Wait for both public scores.
while true; do
  out=$(T $K competitions submissions $C 2>&1 | head -4)
  pending=$(echo "$out" | grep -E 'ConvNeXt-B' | grep -ciE 'pending|running')
  [[ $pending -eq 0 && $(echo "$out" | grep -c 'ConvNeXt-B') -ge 2 ]] && { echo "$out" | grep ConvNeXt-B | awk "{print \$1, \$(NF-1), \$NF}"; exit 0; }
  sleep 900
done
