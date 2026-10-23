"""Attack our cached model (vanilla) or model + InfoGAN (baseline TMD) from the command line, with the authors' own
attack code (TextDefender's vendored TextAttack fork). Same code path as the notebooks (tmd_lab, BACKEND='fork');
useful when you want a number without opening a notebook. Results go to <work>/<dataset>/results_fork_<victim>.json.

usage (Colab):
  !python /content/for_shreyansh/scripts/fork_attack.py --work /content/drive/MyDrive/tmd_run_paperft \
      --victim baseline --attack pwws --n 200
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmd_lab as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--dataset', default='ag_news')
    ap.add_argument('--victim', choices=['vanilla', 'baseline'], required=True)
    ap.add_argument('--attack', choices=['pwws', 'textfooler', 'bertattack'], required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--chunk', type=int, default=50)
    a = ap.parse_args()

    T.setup_env('fork')
    lab = T.Lab(a.dataset, work=a.work)
    lab.use_model('base')
    gan = None
    if a.victim == 'baseline':
        has_gan = os.path.exists(lab._path('infogan_base_best.json')) or os.path.exists(
            lab._path(f"infogan_base_L{lab.cfg['K']}C{lab.cfg['d']}.pt"))
        assert has_gan, 'no trained InfoGAN in this folder: run notebook 00 first'
        gan = lab.get_gan('base', None, None)
    victim = T.Victim(lab, gan=gan, mode='tmd' if gan is not None else 'none', proj_mode='sample', k=a.k)
    E_te, Y_te = lab.embeddings('test', 'base')
    res = T.evaluate(lab, victim, a.victim, E_te, Y_te, attacks=(a.attack,), n=a.n, chunk=a.chunk)
    print(json.dumps({k: res[k] for k in ('clean', a.attack)}, indent=1))


if __name__ == '__main__':
    main()
