#!/bin/zsh
# Everything queued on 2026-10-19 (Kaggle accepts two GPU sessions; each push retries until a slot frees).
#  A. knee-public-9586 v2 (public 0.9586 notebook, R6 mount fixed): submit when its commit run completes.
#  B. 05m hybrids: public 0.9586 + our ConvNeXt-T + ConvNeXt-B v5 students (Z5 + B5, folds 0-1) at weight 0.1 and 0.2.
#  C. K5 (queued since 07:20): submit alone; L5 (re-pushed after the 05m runs): submit K5 + L5.
# Submissions retry every 30 min when the daily limit (5, resets 00:00 UTC) is reached. Logs one line per event.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
C=rsna-knee-abnormality-detection
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
push() { for i in $(seq 1 200); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py "$@" 2>&1 | tail -1); [[ "$out" == *successfully* ]] && { echo "$(date -u +%H:%M) pushed $1"; return 0; }; sleep 300; done; echo "$(date -u +%H:%M) gave up pushing $1: $out" | cut -c1-160; return 1; }
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) echo "$(date -u +%H:%M) $1 complete"; return 0;; *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $1 $s"; return 1;; esac; sleep ${2:-600}; done; }
submit() { for i in $(seq 1 100); do out=$(T $K competitions submit $C -k shreyanshgoyaliitb/$1 -v ${3:-1} -f submission.csv -m "$2" 2>&1 | tail -1); [[ "$out" == *uccessfully* ]] && { echo "$(date -u +%H:%M) submitted $1"; return 0; }; sleep 1800; done; echo "submit $1 failed: $out"; return 1; }
DS=knee-reader-labels,laymond/rsna-knee-abnormality-qwen3-8b-weak-labels,stevenleehans/rsna-knee-llm-report-labels,lixin73/rsna-knee-llm-report-labels-sol56,pilkwang/rsna-knee-llm-labels
KS=knee-cache-v5-shard0,knee-cache-v5-shard1,knee-cache-v5-shard2,knee-cache-v5-shard3,knee-cache-v5-shard4,compi-label-generation,knee-session-a-reader-f0-f1,knee-session-b-reader-f2-f3,knee-session-c-reader-f4-slot-rad,knee-session-d-reader-s7-f0-f1,knee-session-e-reader-s7-f2-f3,knee-session-f-reader-s7-f4-s11-f0
CT=$KS,knee-compact-oof-s0,knee-compact-oof-s1
NB=RUN_ON_KAGGLE/05i_submit_own_students.ipynb
PUB_DS=knee-wheelhouse,goodpjw2008/rsna-knee-2-5d-convnext-reader,rostigor/rsna-knee-r6-fe055406,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004
PUB_KS=goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958,knee-session-z5-student-v5-f0-f1,knee-session-b5-student-cnxb-v5-f0-f1

(  # A
  wait_done knee-public-9586 300 && submit knee-public-9586 "Public: heliosli 0.9586 notebook run unchanged except the R6 weights mount (rostigor copy, manifest sha256 pinned): Raptor 384 + 224, R6 compact with flip TTA, goodpjw2008 2.5D reader" 2
) &
(  # B, then L5 into the slot B frees
  for w in 10 20; do
    push knee-submit-public9586-own-w$w RUN_ON_KAGGLE/05m_submit_public9586_own_w$w.ipynb --gpu --competition --datasets $PUB_DS --kernels $PUB_KS || continue
    ( wait_done knee-submit-public9586-own-w$w 300 && submit knee-submit-public9586-own-w$w "Hybrid: public 0.9586 notebook x0.$((10 - w / 10)) + own ConvNeXt-T + ConvNeXt-B v5 students (f0-1 each, LB 0.940) x0.$((w / 10)), per-finding rank blend; public-only fallback" ) &
  done
  push knee-session-l5-student-cnxb-ct-v5-f0-f1 RUN_ON_KAGGLE/04_session_L5_student_cnxb_ct_v5_f0_f1.ipynb --gpu --internet --competition --datasets $DS --kernels $CT || exit 1
  wait_done knee-session-l5-student-cnxb-ct-v5-f0-f1 900 || exit 1
  wait_done knee-session-k5-student-ct-v5-f0-f1 900 || exit 1
  push knee-submit-ct-tb-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1,knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
  wait_done knee-submit-ct-tb-v5 300 && submit knee-submit-ct-tb-v5 "Own students: ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF, folds 0-1 each; compare T + B with our readers' teacher = 0.940"
  wait
) &
(  # C: K5 alone
  wait_done knee-session-k5-student-ct-v5-f0-f1 900 || exit 1
  push knee-submit-ct-v5 $NB --gpu --competition --datasets knee-wheelhouse --kernels knee-session-k5-student-ct-v5-f0-f1 || exit 1
  wait_done knee-submit-ct-v5 300 && submit knee-submit-ct-v5 "A/B: ConvNeXt-T v5 students taught by our readers + heliosli compact OOF (quantile-mapped, half weight), folds 0-1; compare same student with our readers' teacher = 0.939"
) &
wait
echo "$(date -u +%H:%M) all branches done; waiting for scores"
perl -e "alarm 10*3600; exec @ARGV" scripts/ops/wait_scores.sh "heliosli 0.9586 notebook run unchanged" "all five folds each (f2-4 on TPU)" "Hybrid: public 0.9586 notebook x0.9" "Hybrid: public 0.9586 notebook x0.8" "ConvNeXt-T v5 students taught by our readers + heliosli compact OOF" "ConvNeXt-T + ConvNeXt-B v5 taught by our readers + heliosli compact OOF"
