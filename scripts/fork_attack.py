"""Run the authors' own attack code (TextDefender's vendored TextAttack fork) on our cached model and InfoGAN.

Why: our harness rebuilds the paper's attacks on pip `textattack`. Vanilla PWWS/TextFooler and TMD TextFooler now match
the paper, but TMD PWWS does not. This script answers whether the remaining gap comes from the attack code: same model,
same GAN, same test examples, but the authors' fork does the attacking.

It runs as its own process because the notebook kernel has already imported pip `textattack`; here `TextDefender/` is
first on sys.path, so `import textattack` resolves to the fork. The attack is built exactly like
TextDefender/utils/textattack_utils.py::build_english_attacker and results are counted like TextDefender's
SimplifidResult. Note that the fork silently drops examples whose attack raises an exception
(TextDefender/textattack/shared/attack.py, attack_dataset); the number dropped is reported.

usage (Colab):
  !python /content/for_shreyansh/scripts/fork_attack.py --work /content/drive/MyDrive/tmd_run_paperft \
      --victim baseline --attack pwws --n 200
"""
import argparse
import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'TextDefender'))
sys.path.insert(1, ROOT)
# The fork's ensemble wrapper imports TextDefender's `args` module, which pulls in every trainer and their
# dependencies. Only the ProgramArgs name is needed at import time, so a stub is enough.
_args = types.ModuleType('args')
_args.ProgramArgs = type('ProgramArgs', (), {})
sys.modules['args'] = _args

import torch  # noqa: E402

import tmd_lab as T  # noqa: E402


def build_attacker(name, wrapper, rho, K=50, sim=0.840845057):
    """build_english_attacker (TextDefender/utils/textattack_utils.py) for pwws, textfooler and bae/bertattack."""
    import textattack
    from textattack.attack_recipes import BERTAttackLi2020, PWWSRen2019, TextFoolerJin2019
    from textattack.constraints.overlap import MaxWordsPerturbed
    from textattack.constraints.pre_transformation import InputColumnModification
    from textattack.constraints.semantics import WordEmbeddingDistance
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.goal_functions import UntargetedClassification
    from textattack.shared.attack import Attack
    from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
    assert 'TextDefender' in textattack.__file__, f'expected the fork, got {textattack.__file__}'

    if name == 'pwws':
        attacker = PWWSRen2019.build(wrapper)
    elif name == 'textfooler':
        attacker = TextFoolerJin2019.build(wrapper)
    else:
        attacker = BERTAttackLi2020.build(wrapper)
        attacker.transformation = WordSwapMaskedLM(method='bert-attack', max_candidates=K)
    if name in ['textfooler', 'pwws']:
        attacker.transformation = WordSwapEmbedding(max_candidates=K)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
    attacker.constraints.append(MaxWordsPerturbed(max_percent=rho))
    attacker.constraints.append(UniversalSentenceEncoder(threshold=sim, metric='angular', compare_against_original=False,
                                                         window_size=15, skip_text_shorter_than_window=True))
    attacker.pre_transformation_constraints.append(InputColumnModification(['premise', 'hypothesis'], {'premise'}))
    attacker.goal_function = UntargetedClassification(wrapper, query_budget=K)  # fork: budget = K x number of words
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def make_wrapper(victim):
    from textattack.models.wrappers import ModelWrapper

    class VictimWrapper(ModelWrapper):
        def __init__(self, victim):
            self.victim = victim
            self.model = victim.lab.model

        def __call__(self, text_list):
            out = []
            with torch.no_grad():
                for i in range(0, len(text_list), 32):  # TextDefender's wrapper batch size
                    out.append(self.victim.scores(list(text_list[i:i + 32])).float().cpu())
            return torch.cat(out)

        def get_grad(self, text_input):
            raise NotImplementedError

    return VictimWrapper(victim)


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

    T.setup_env()
    lab = T.Lab(a.dataset, work=a.work)
    lab.use_model('base')
    gan = None
    if a.victim == 'baseline':
        has_gan = os.path.exists(lab._path('infogan_base_best.json')) or os.path.exists(
            lab._path(f"infogan_base_L{lab.cfg['K']}C{lab.cfg['d']}.pt"))
        assert has_gan, 'no trained InfoGAN in this folder: run notebook 00 first'
        gan = lab.get_gan('base', None, None)
    victim = T.Victim(lab, gan=gan, mode='tmd' if gan is not None else 'none', proj_mode='sample', k=a.k)
    attack = build_attacker(a.attack, make_wrapper(victim), lab.cfg['rho'])
    pairs = lab.attack_pairs(a.n)

    from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
    path = lab._path(f'results_fork_{a.victim}.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    if a.attack in res and res[a.attack].get('n_requested') == len(pairs):
        print(json.dumps({a.attack: res[a.attack]}, indent=1))
        return
    key = f'_partial_{a.attack}'
    st = res.get(key) if res.get(key, {}).get('n_requested') == len(pairs) else None
    st = st or dict(n_requested=len(pairs), done=0, succ=0, fail=0, skip=0, dropped=0, queries=0.0, seconds=0.0)
    if st['done']:
        print(f'resuming at example {st["done"]}', flush=True)
    while st['done'] < len(pairs):
        part = pairs[st['done']:st['done'] + a.chunk]
        t0, got = time.time(), 0
        for r in attack.attack_dataset(part):
            got += 1
            if isinstance(r, SuccessfulAttackResult):
                st['succ'] += 1
            elif isinstance(r, FailedAttackResult):
                st['fail'] += 1
            elif isinstance(r, SkippedAttackResult):
                st['skip'] += 1
            st['queries'] += float(r.perturbed_result.num_queries)
        st['dropped'] += len(part) - got
        st['done'] += len(part)
        st['seconds'] += time.time() - t0
        counted = st['succ'] + st['fail'] + st['skip']
        print(f'  [fork {a.victim} {a.attack}] {st["done"]}/{len(pairs)} done, acc under attack '
              f'{100 * st["fail"] / max(1, counted):.1f}% (dropped so far: {st["dropped"]}), '
              f'{st["seconds"] / 60:.1f} min', flush=True)
        res[key] = st
        json.dump(res, open(path, 'w'), indent=1)
    counted = st['succ'] + st['fail'] + st['skip']
    res[a.attack] = dict(
        acc=100 * st['fail'] / max(1, counted),  # TextDefender's Accu(rob): dropped examples are not counted
        acc_counting_dropped_as_wrong=100 * st['fail'] / len(pairs),
        success_rate=100 * st['succ'] / max(1, st['succ'] + st['fail']),
        orig_acc=100 * (st['succ'] + st['fail']) / max(1, counted),
        n=counted, n_requested=len(pairs), dropped=st['dropped'],
        avg_queries=st['queries'] / max(1, counted), minutes=st['seconds'] / 60)
    res.pop(key, None)
    json.dump(res, open(path, 'w'), indent=1)
    print(json.dumps({a.attack: res[a.attack]}, indent=1))


if __name__ == '__main__':
    main()
