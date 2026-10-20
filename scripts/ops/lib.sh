# Shared helpers for the Kaggle queue scripts (push / wait / idempotent submit).
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 200); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); [[ "$out" == *successfully* ]] && { echo "$(date -u +%H:%M) pushed $1"; return 0; }; sleep 300; done; echo "$(date -u +%H:%M) gave up pushing $1: $out" | cut -c1-160; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) echo "$(date -u +%H:%M) $1 complete"; return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-600}; done; }
has_sub() { T $K competitions submissions $C --csv 2>/dev/null | grep -qF -- "$1"; }
submit() {   # kernel, description, version
  for i in $(seq 1 100); do
    has_sub "$2" && { echo "$(date -u +%H:%M) submitted $1"; return 0; }
    T $K competitions submit $C -k shreyanshgoyaliitb/$1 -v ${3:-1} -f submission.csv -m "$2" >/dev/null 2>&1
    sleep 120
    has_sub "$2" && { echo "$(date -u +%H:%M) submitted $1"; return 0; }
    sleep 1800
  done
  echo "$(date -u +%H:%M) could not submit $1"; return 1
}
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v5-shard0,knee-cache-v5-shard1,knee-cache-v5-shard2,knee-cache-v5-shard3,knee-cache-v5-shard4,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
CT=$KS,knee-compact-oof-s0,knee-compact-oof-s1
NB=RUN_ON_KAGGLE/05i_submit_own_students.ipynb
