#!/bin/zsh
# 2026-10-19 (the Oct 17 chain stopped after the compact OOF and TPU ConvNeXt-B f2-4 runs).
#  GPU: K5 (compact-taught ConvNeXt-T v5 f0-1) + the full five-fold T + B v5 submission; L5 when that submission frees its slot.
#  TPU: compact-taught ConvNeXt-T v5 folds 2-4 (used only if K5 beats 0.939).
#  Submits: T x5 + B x5 (compare f0-1 each = 0.940); K5 alone (compare 0.939); K5 + L5 (compare 0.940).
cd /Users/shreyanshgoyal/Documents/compi
# (K5 and the TPU job were pushed at 07:20 UTC, before this restart.)
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 60); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); echo "$(date -u +%H:%M) $1: $out" | cut -c1-120; [[ "$out" == *successfully* ]] && return 0; sleep 300; done; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) echo "$(date -u +%H:%M) $1 complete"; return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-600}; done; }
submit() { T $K competitions submit $C -k shreyanshgoyaliitb/$1 -v 1 -f submission.csv -m "$2" 2>&1 | tail -1; }
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v5-shard0,knee-cache-v5-shard1,knee-cache-v5-shard2,knee-cache-v5-shard3,knee-cache-v5-shard4,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
CT=$KS,knee-compact-oof-s0,knee-compact-oof-s1
NB=RUN_ON_KAGGLE/05i_submit_own_students.ipynb

push knee-submit-v5-tb-x5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-z5-student-v5-f0-f1,knee-tpu-student-v5-f234,knee-session-b5-student-cnxb-v5-f0-f1,knee-tpu-student-cnxb-v5-f234 || exit 1
sleep 600
wait_done knee-submit-v5-tb-x5 240 && submit knee-submit-v5-tb-x5 "Own students: ConvNeXt-T + ConvNeXt-B v5, all five folds each (f2-4 on TPU); compare folds 0-1 each = 0.940"
push knee-session-l5-student-cnxb-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_L5_student_cnxb_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $CT || exit 1
scripts/ops/wait_scores.sh "all five folds each (f2-4 on TPU)"
wait_done knee-session-k5-student-ct-v5-f0-f1 || exit 1
push knee-submit-ct-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1 || exit 1
sleep 300; wait_done knee-submit-ct-v5 240 || exit 1
submit knee-submit-ct-v5 "A/B: ConvNeXt-T v5 students taught by our readers + heliosli compact OOF (quantile-mapped, half weight), folds 0-1; compare same student with our readers' teacher = 0.939"
scripts/ops/wait_scores.sh "ConvNeXt-T v5 students taught by our readers + heliosli compact OOF"
wait_done knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
push knee-submit-ct-tb-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1,knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
sleep 300; wait_done knee-submit-ct-tb-v5 240 || exit 1
submit knee-submit-ct-tb-v5 "Own students: ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF, folds 0-1 each; compare T + B with our readers' teacher = 0.940"
scripts/ops/wait_scores.sh "ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF"
