#!/bin/zsh
# Hybrid test: public 0.954 stack x0.7 + own v5 students (x5) x0.3; push, wait, submit.
cd /Users/shreyanshgoyal/Documents/compi
K=.venv/bin/kaggle
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
st() { T $K kernels status shreyanshgoyaliitb/$1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
PUBDS=knee-wheelhouse,heliosli/rsna-knee-compact-v1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004
for i in $(seq 1 20); do out=$(perl -e 'alarm 280; exec @ARGV' .venv/bin/python scripts/kaggle_push.py knee-submit-public954-v5-w30 RUN_ON_KAGGLE/05k_submit_public954_own_students_w30.ipynb --gpu --competition --datasets $PUBDS --kernels knee-session-z5-student-v5-f0-f1,knee-tpu-student-v5-f234 2>&1 | tail -1); echo "$(date -u +%H:%M) $out" | cut -c1-120; [[ "$out" == *successfully* ]] && break; sleep 300; done
sleep 120
while true; do s=$(st knee-submit-public954-v5-w30); case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel $s"; exit 1;; esac; sleep 240; done
T $K competitions submit rsna-knee-abnormality-detection -k shreyanshgoyaliitb/knee-submit-public954-v5-w30 -v 1 -f submission.csv \
  -m "Hybrid: public 0.954 stack (heliosli: Raptor x0.7 + compact ConvNeXt-Small x0.3) x0.7 + own v5 students x5 x0.3, per-finding rank blend" 2>&1 | tail -1
while true; do sub=$(T $K competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p); if [ -n "$sub" ] && ! echo "$sub" | grep -q PENDING; then echo "$(date -u +%H:%M) hybrid v5 w30: $(echo "$sub" | awk -F, '{print $(NF-1)}')"; break; fi; sleep 600; done
