#!/bin/zsh
# After the 2026-10-17 00:00 UTC quota reset:
#  1. TPU: ConvNeXt-B v5 students folds 2-4 (B5 trained 0-1 on GPU).
#  2. GPU: heliosli compact out-of-fold predictions on the training studies (two shards, both GPU slots).
#  3. GPU: compact-taught students K5 (ConvNeXt-T v5) and L5 (ConvNeXt-B v5), folds 0-1.
#  4. Submit K5 alone (compare Z5 = 0.939), then K5 + L5 (compare Z5 + B5 = 0.940).
# Exits on the first failure so the watcher wakes Claude.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 30); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); echo "$(date -u +%H:%M) $1: $out" | cut -c1-120; [[ "$out" == *successfully* ]] && return 0; sleep 300; done; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) echo "$(date -u +%H:%M) $1 complete"; return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-600}; done; }
submit() { T $K competitions submit $C -k shreyanshgoyaliitb/$1 -v 1 -f submission.csv -m "$2" 2>&1 | tail -1; }
score() { scripts/ops/wait_scores.sh "$1"; }

start=$(date -j -u -f "%Y-%m-%d %H:%M" "2026-10-17 00:05" +%s)
while [ $(date -u +%s) -lt $start ]; do sleep 600; done
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v5-shard0,knee-cache-v5-shard1,knee-cache-v5-shard2,knee-cache-v5-shard3,knee-cache-v5-shard4,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
CDS=heliosli/rsna-knee-compact-v1,heliosli/rsna-orthofoundation-fold0-epoch7-20261004

push knee-tpu-student-cnxb-v5-f234 RUN_ON_KAGGLE/08_tpu_student_cnxb_v5_f234.ipynb --tpu --internet --competition --datasets $DS --kernels $KS
push knee-compact-oof-s0 RUN_ON_KAGGLE/10_compact_oof_s0.ipynb --gpu --competition --datasets $CDS || exit 1
push knee-compact-oof-s1 RUN_ON_KAGGLE/10_compact_oof_s1.ipynb --gpu --competition --datasets $CDS || exit 1
sleep 1800
wait_done knee-compact-oof-s0 || exit 1
wait_done knee-compact-oof-s1 || exit 1

push knee-session-k5-student-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_K5_student_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $KS,knee-compact-oof-s0,knee-compact-oof-s1 || exit 1
push knee-session-l5-student-cnxb-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_L5_student_cnxb_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $KS,knee-compact-oof-s0,knee-compact-oof-s1 || exit 1
sleep 3600
wait_done knee-session-k5-student-ct-v5-f0-f1 || exit 1
NB=RUN_ON_KAGGLE/05i_submit_own_students.ipynb
push knee-submit-ct-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1 || exit 1
sleep 300; wait_done knee-submit-ct-v5 240 || exit 1
submit knee-submit-ct-v5 "A/B: ConvNeXt-T v5 students taught by our readers + heliosli compact OOF (quantile-mapped, half weight), folds 0-1; compare same student with our readers' teacher = 0.939"
score "ConvNeXt-T v5 students taught by our readers + heliosli compact OOF"
wait_done knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
push knee-submit-ct-tb-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1,knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
sleep 300; wait_done knee-submit-ct-tb-v5 240 || exit 1
submit knee-submit-ct-tb-v5 "Own students: ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF, folds 0-1 each; compare T + B with our readers' teacher = 0.940"
score "ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF"
