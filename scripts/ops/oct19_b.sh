#!/bin/zsh
# Queue after the 2026-10-19 fixes (K5 teacher families; idempotent submits).
#  - K5 v2 and TPU compact-taught ConvNeXt-T f2-4 v2 re-pushed (v1 stopped on the missing reader_caformer family).
#  - 05m hybrids w10 / w20 (commit runs pushed 09:44 / 11:06): submit when complete.
#  - L5 after K5; then K5 alone and K5 + L5 submissions.
# submit() is idempotent: it submits only if no submission with that description exists, and counts it done only
# once the description appears in the submission list (the daily limit of 5 resets 00:00 UTC; it retries every 30 min).
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
D_K5="A/B: ConvNeXt-T v5 students taught by our ConvNeXt readers + heliosli compact OOF (quantile-mapped, 1:1), folds 0-1; compare same student with the readers alone = 0.939"
D_K5L5="Own students: ConvNeXt-T + ConvNeXt-B v5 taught by our ConvNeXt readers + heliosli compact OOF (1:1), folds 0-1 each; compare T + B with the readers alone = 0.940"

push knee-tpu-student-ct-v5-f234 RUN_ON_KAGGLE/08_tpu_student_ct_v5_f234.ipynb --tpu --internet --competition --datasets $DS --kernels $CT
(  # hybrids
  for w in 10 20; do
    ( wait_done knee-submit-public9586-own-w$w 300 && submit knee-submit-public9586-own-w$w "Hybrid: public 0.9586 notebook x0.$((10 - w / 10)) + own ConvNeXt-T + ConvNeXt-B v5 students (f0-1 each, LB 0.940) x0.$((w / 10)), per-finding rank blend; public-only fallback" ) &
  done
  wait
) &
(  # K5, then L5; submissions
  push knee-session-k5-student-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_K5_student_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $CT || exit 1
  push knee-session-l5-student-cnxb-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_L5_student_cnxb_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $CT
  sleep 1800
  wait_done knee-session-k5-student-ct-v5-f0-f1 900 || exit 1
  push knee-submit-ct-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1 || exit 1
  wait_done knee-submit-ct-v5 300 && submit knee-submit-ct-v5 "$D_K5"
  wait_done knee-session-l5-student-cnxb-ct-v5-f0-f1 900 || exit 1
  push knee-submit-ct-tb-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1,knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
  wait_done knee-submit-ct-tb-v5 300 && submit knee-submit-ct-tb-v5 "$D_K5L5"
) &
wait
echo "$(date -u +%H:%M) all branches done; waiting for scores"
perl -e "alarm 10*3600; exec @ARGV" scripts/ops/wait_scores.sh "heliosli 0.9586 notebook run unchanged" "Hybrid: public 0.9586 notebook x0.9" "Hybrid: public 0.9586 notebook x0.8" "students taught by our ConvNeXt readers + heliosli compact OOF (quantile-mapped" "ConvNeXt-B v5 taught by our ConvNeXt readers + heliosli compact OOF"
