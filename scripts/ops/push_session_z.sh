#!/bin/zsh
# Waits for the three knee-2c-v4 cache shard kernels, then pushes session Z (ConvNeXt student on v4, folds 0/1).
# Exits early (non-zero) if a shard fails. Kaggle CLI calls are time-limited so a network stall cannot hang it.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 120; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do
  done_n=0
  for s in 0 1 2; do
    x=$(st knee-cache-v4-shard$s)
    case "$x" in *COMPLETE*) done_n=$((done_n+1));; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) shard$s $x"; exit 1;; esac
  done
  [ $done_n -eq 3 ] && break
  sleep 300
done
echo "$(date -u +%H:%M) all v4 shards complete"
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v4-shard0,knee-cache-v4-shard1,knee-cache-v4-shard2,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
for i in $(seq 1 40); do
  out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-session-z-student-v4-f0-f1 RUN_ON_KAGGLE/04_session_Z_student_v4_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $KS 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $out"
  [[ "$out" == *successfully* ]] && exit 0
  sleep 300
done
exit 1
