"""Convert local report-label runs into extra_labels_*.csv reader tables for training.

Run from the repository root: python scripts/export_reader_labels.py
Upload the reader_labels/ folder as one Kaggle dataset and attach it to the training
notebooks; label_builder.py averages these with the public tables and with the Flash
member file read directly from the 01_labels notebook output.

Readers exported here (Gemini 3.1 Pro is not used):
  GPT Sol, protocol knee-report-labels-v2 (partial runs are fine; rerun when it finishes)
  Sonnet, legacy protocol (complete)
State -> probability maps are label_builder.STATE_P, fixed before any training.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_builder import STATE_KEYS, iter_jsonl, states_frame  # noqa: E402

OUT = ROOT / 'reader_labels'
LABELS = list(STATE_KEYS)


def main():
    OUT.mkdir(exist_ok=True)
    for stale in OUT.glob('extra_labels_*.csv'):
        stale.unlink()
    readers = {}
    # GPT Sol, v2 protocol. Records 400-700 of the GPT-5.6 file show a positive-rate drift
    # (effusion definite 38-74% against 10-20% elsewhere), so that window is left out.
    sol56 = list(iter_jsonl(ROOT / 'sol_label_generation/output_v2/pseudo_sol_light.jsonl'))
    sol61 = list(iter_jsonl(ROOT / 'sol_label_generation/output_v2_gpt_6_1/pseudo_sol_light.jsonl'))
    readers['extra_labels_sol_v2.csv'] = states_frame(sol56[:400] + sol56[700:] + sol61, LABELS, 'v2')
    # Sonnet, legacy protocol: the Kaggle-cached batch plus the completion run.
    sonnet = [r for p in ['sonnet_label_completion/pseudo_anthropic_claude_sonnet_5_default.jsonl',
                          'sonnet_label_completion/output/sonnet_new_labels.jsonl']
              for r in iter_jsonl(ROOT / p)]
    readers['extra_labels_sonnet_legacy.csv'] = states_frame(sonnet, LABELS, 'legacy')
    for name, frame in readers.items():
        assert frame['StudyInstanceUID'].is_unique
        frame.to_csv(OUT / name, index=False)
        print(f'{name}: {len(frame):,} studies, {frame[LABELS].notna().mean().mean():.1%} observed cells')


if __name__ == '__main__':
    main()
