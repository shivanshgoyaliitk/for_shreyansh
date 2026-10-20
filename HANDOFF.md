# Handoff — 2026-10-21 (final deadline 2026-10-22)

Read `CLAUDE.md`, then this file, then `METHOD_NOTES.md` (results table at the end).

## Public leaderboard (rank ~2,000 of 6,204 on Oct 19 before the pivot; 1,385 teams ≥ 0.958, top 0.969)

| Submission (kernel) | Public LB |
|---|---|
| `knee-public-9586` v2: heliosli's public 0.9586 notebook, R6 mount fixed (below) | **0.958** (submitted 3× by a script bug) |
| `knee-submit-public9586-own-w10` / `-w20`: 0.9586 + Z5+B5 at 0.1 / 0.2 | 0.957 / 0.957 |
| `knee-submit-public9586-ct-w10`: 0.9586 + K5+L5 (compact-taught) at 0.1 | 0.957 |
| `knee-submit-ct-tb-v5`: K5 + L5, own models only | 0.946 (own best) |
| `knee-submit-ct-v5`: K5 alone | 0.945 |

Own models do not lift the 0.958 stack at weight 0.1–0.2.

## Running / queued on Kaggle (started from the Mac; check with the Kaggle CLI)

- TPU `knee-tpu-student-cnxb-ct-v5-f234`: compact-taught ConvNeXt-B v5, folds 2–4.
- `scripts/ops/oct21.sh`, if the Mac is still on:
  - commits and submits `knee-submit-public9586-ct7-w05`: 0.9586 + compact-taught T ×5 + B f0-1 at 0.05;
  - after the TPU job, `knee-submit-public9586-ct10-w05`: T ×5 + B ×5 at 0.05.
  - If the Mac is off, run those steps by hand (commands are in the script).

## Final selection — must be done by hand on the Submissions page before the deadline

Kaggle's automatic pick would take two of the identical 0.958 copies. Select:

1. `knee-public-9586 - Version 2` (0.958), one copy.
2. The best 0.05 hybrid if it reaches 0.958; otherwise `knee-submit-public9586-ct-w10` (0.957, includes our models).

## Kaggle facts learned the hard way

- At most 2 batch GPU sessions at a time. Queued ones count, and so do submission scoring runs; extra pushes are refused, not queued.
- `kaggle competitions submit` prints "N submissions remaining today" as its last line. Check the submission list instead (`scripts/ops/lib.sh` `submit()` is idempotent).
- 5 submissions/day, reset 00:00 UTC.
- heliosli replaced the contents of `heliosli/rsna-knee-r6-frozen-fe055406` with R7 (bundle 8c7b666e), which breaks the 0.9586 notebook. Mount `rostigor/rsna-knee-r6-fe055406` instead: its manifest sha256 is 6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648, the pinned R6. `scripts/build_launchers.py` applies this to the 05m notebooks.
- Kaggle credentials come from the environment (KAGGLE_USERNAME / KAGGLE_KEY); never print them.
- Wrap every Kaggle CLI call in a timeout (`perl -e 'alarm N; exec @ARGV'` on macOS); hung calls stalled scripts before.

## Key method result (2026-10-20)

Teaching the v5 students with heliosli's compact model gave +0.006 for our own models (K5 0.945 vs 0.939; K5+L5 0.946 vs 0.940). The compact model's out-of-fold predictions come from `10_compact_oof_s0/s1`; the teacher is quantile-mapped and mixed 1:1 with our ConvNeXt readers. See METHOD_NOTES item 13.
