#!/bin/zsh
# 2026-10-20: compact-taught students work (K5 0.945, K5+L5 0.946 vs 0.939 / 0.940). Does the stronger own ensemble
# now add to the public 0.958 stack? 05m at weight 0.1 with K5 + L5. TPU: compact-taught ConvNeXt-B folds 2-4 after the
# queued compact-taught ConvNeXt-T folds 2-4 (one TPU session at a time).
source /Users/shreyanshgoyal/Documents/compi/scripts/ops/lib.sh
PUB_DS=knee-wheelhouse,goodpjw2008/rsna-knee-2-5d-convnext-reader,rostigor/rsna-knee-r6-fe055406,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004
PUB_KS=goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958,knee-session-k5-student-ct-v5-f0-f1,knee-session-l5-student-cnxb-ct-v5-f0-f1
D_HYB="Hybrid: public 0.9586 notebook x0.9 + own compact-taught ConvNeXt-T + ConvNeXt-B v5 students (K5 + L5, f0-1 each, LB 0.946) x0.1, per-finding rank blend; public-only fallback"
(
  push knee-submit-public9586-ct-w10 RUN_ON_KAGGLE/05m_submit_public9586_own_w10.ipynb --gpu --competition --datasets $PUB_DS --kernels $PUB_KS || exit 1
  wait_done knee-submit-public9586-ct-w10 300 && submit knee-submit-public9586-ct-w10 "$D_HYB"
) &
(
  wait_done knee-tpu-student-ct-v5-f234 900
  push knee-tpu-student-cnxb-ct-v5-f234 RUN_ON_KAGGLE/08_tpu_student_cnxb_ct_v5_f234.ipynb --tpu --internet --competition --datasets $DS --kernels $CT
  wait_done knee-tpu-student-cnxb-ct-v5-f234 900
) &
wait
perl -e "alarm 6*3600; exec @ARGV" scripts/ops/wait_scores.sh "own compact-taught ConvNeXt-T + ConvNeXt-B v5 students (K5 + L5"
