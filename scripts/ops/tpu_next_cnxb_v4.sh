#!/bin/zsh
# When the queued/running TPU job knee-tpu-student-v4 ends, push the ConvNeXt-B v4 TPU job (one TPU session at a time).
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
while true; do s=$(T $K kernels status shreyanshgoyaliitb/knee-tpu-student-v4 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 900; done
echo "$(date -u +%H:%M) knee-tpu-student-v4 $s"
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v4-shard0,knee-cache-v4-shard1,knee-cache-v4-shard2,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
for i in $(seq 1 30); do
  out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-tpu-student-cnxb-v4-f01 RUN_ON_KAGGLE/08_tpu_student_cnxb_v4_f01.ipynb --tpu --internet --competition --datasets $DS --kernels $KS 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $out"; [[ "$out" == *successfully* ]] && exit 0; sleep 300
done
exit 1
