#!/bin/zsh
# Oct 12: run three submission kernels now (two GPU sessions at a time), then submit each after the 00:00 UTC reset.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
CNX=knee-session-m-student-f0-f1,fork-of-knee-session-m-student-f0-f1,notebookeade41a204
CAF=knee-session-w-student-caf-f0-f1,knee-session-x-student-caf-f2-f3,knee-session-y-student-caf-f4-s7-f0
PUBDS=knee-wheelhouse,heliosli/rsna-knee-compact-v1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004
push() {  # slug notebook datasets kernels
  for i in $(seq 1 40); do
    out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py $1 RUN_ON_KAGGLE/$2 --gpu --competition --datasets $3 --kernels $4 2>&1 | tail -1)
    echo "$(date -u +%H:%M) $1: $out"; [[ "$out" == *successfully* ]] && return 0; sleep 300
  done; return 1
}
wait_done() { while true; do s=$(st $1); case "$s" in *COMPLETE*) return 0;; *ERROR*|*CANCEL*) echo "$1 $s"; return 1;; esac; sleep 240; done; }
push knee-submit-student-cnxb 05i_submit_own_students.ipynb knee-wheelhouse knee-tpu-student-cnxb || exit 1
push knee-submit-own-v3v4 05i_submit_own_students.ipynb knee-wheelhouse $CNX,$CAF,knee-session-z-student-v4-f0-f1 || exit 1
wait_done knee-submit-student-cnxb || exit 1
push knee-submit-public954-v4 05k_submit_public954_own_students_w20.ipynb $PUBDS knee-session-z-student-v4-f0-f1 || exit 1
wait_done knee-submit-own-v3v4 || exit 1
wait_done knee-submit-public954-v4 || exit 1
echo "$(date -u +%H:%M) all three kernels complete; waiting for the 00:00 UTC reset"
while [ "$(date -u +%Y%m%d%H%M)" -lt "202610120002" ]; do sleep 120; done
sub() { T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/$1 -v 1 -f submission.csv -m "$2" 2>&1 | tail -1; sleep 20; }
sub knee-submit-student-cnxb "A/B: ConvNeXt-Base (in22k) noisy students x5 alone, trained on TPU v5e-8 (SPMD); compare ConvNeXt-T students x6 = 0.936"
sub knee-submit-own-v3v4 "Own models only: ConvNeXt v3 x6 + CAFormer v3 x6 + ConvNeXt v4 (anchored local slots) x2, equal family rank weights; v3 members read the v4 decode's first six slots"
sub knee-submit-public954-v4 "Hybrid: public 0.954 stack (heliosli: Raptor x0.7 + compact ConvNeXt-Small x0.3) x0.8 + own v4 students (folds 0-1) x0.2, per-finding rank blend"
echo "$(date -u +%H:%M) submitted three"
