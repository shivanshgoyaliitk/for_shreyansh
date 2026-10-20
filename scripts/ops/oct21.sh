#!/bin/zsh
# 2026-10-21 queue (submissions reset 00:00 UTC). Own models have not lifted the public 0.958 stack at weight 0.1-0.2
# (0.957 every time), so the remaining hybrids use weight 0.05 with the full-fold compact-taught students:
#  A. public 0.9586 + ct ConvNeXt-T x5 (K5 f0-1 + TPU f2-4) + ct ConvNeXt-B f0-1 (L5), weight 0.05 (commit run now).
#  B. when TPU ct ConvNeXt-B f2-4 completes: public 0.9586 + ct T x5 + ct B x5, weight 0.05.
source /Users/shreyanshgoyal/Documents/compi/scripts/ops/lib.sh
PUB_DS=knee-wheelhouse,goodpjw2008/rsna-knee-2-5d-convnext-reader,rostigor/rsna-knee-r6-fe055406,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004
PUB=goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958
NB5=RUN_ON_KAGGLE/05m_submit_public9586_own_w05.ipynb
D_A="Hybrid: public 0.9586 notebook x0.95 + own compact-taught ConvNeXt-T x5 + ConvNeXt-B f0-1 v5 students x0.05, per-finding rank blend; public-only fallback"
D_B="Hybrid: public 0.9586 notebook x0.95 + own compact-taught ConvNeXt-T x5 + ConvNeXt-B x5 v5 students x0.05, per-finding rank blend; public-only fallback"
(
  push knee-submit-public9586-ct7-w05 $NB5 --gpu --competition --datasets $PUB_DS --kernels $PUB,knee-session-k5-student-ct-v5-f0-f1,knee-tpu-student-ct-v5-f234,knee-session-l5-student-cnxb-ct-v5-f0-f1 || exit 1
  wait_done knee-submit-public9586-ct7-w05 300 && submit knee-submit-public9586-ct7-w05 "$D_A"
) &
(
  wait_done knee-tpu-student-cnxb-ct-v5-f234 900 || exit 1
  push knee-submit-public9586-ct10-w05 $NB5 --gpu --competition --datasets $PUB_DS --kernels $PUB,knee-session-k5-student-ct-v5-f0-f1,knee-tpu-student-ct-v5-f234,knee-session-l5-student-cnxb-ct-v5-f0-f1,knee-tpu-student-cnxb-ct-v5-f234 || exit 1
  wait_done knee-submit-public9586-ct10-w05 300 && submit knee-submit-public9586-ct10-w05 "$D_B"
) &
wait
perl -e "alarm 8*3600; exec @ARGV" scripts/ops/wait_scores.sh "ConvNeXt-T x5 + ConvNeXt-B f0-1 v5 students x0.05" "ConvNeXt-T x5 + ConvNeXt-B x5 v5 students x0.05"
