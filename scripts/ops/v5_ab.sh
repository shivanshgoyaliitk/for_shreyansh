#!/bin/zsh
# v5 A/B: wait for the five v5 cache shards, train session Z5 (GPU, folds 0-1), then push and submit its 05i kernel.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 30); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); echo "$(date -u +%H:%M) $1: $out" | cut -c1-120; [[ "$out" == *successfully* ]] && return 0; sleep 300; done; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-300}; done; }
for s in 0 1 2 3 4; do wait_done knee-cache-v5-shard$s || exit 1; done
echo "$(date -u +%H:%M) v5 cache complete"
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v5-shard0,knee-cache-v5-shard1,knee-cache-v5-shard2,knee-cache-v5-shard3,knee-cache-v5-shard4,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
push knee-session-z5-student-v5-f0-f1 RUN_ON_KAGGLE/04_session_Z5_student_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $KS || exit 1
sleep 300; wait_done knee-session-z5-student-v5-f0-f1 600 || exit 1
push knee-submit-student-v5 RUN_ON_KAGGLE/05i_submit_own_students.ipynb --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1 || exit 1
sleep 120; wait_done knee-submit-student-v5 240 || exit 1
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-student-v5 -v 1 -f submission.csv \
  -m "A/B: ConvNeXt noisy students on knee-2c-v5 (288 px, 10 windows per series, anchored local slots), folds 0-1 only; compare v4 folds 0-1 = 0.938" 2>&1 | tail -1
echo "$(date -u +%H:%M) v5 A/B submitted"
