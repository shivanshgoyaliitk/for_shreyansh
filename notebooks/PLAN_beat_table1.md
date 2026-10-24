# Can we beat Table 1 fully? (analysis, not yet implemented)

Target: the paper's TMD row for BERT / AG News, under the paper's own protocol (the authors' TextAttack fork,
adaptive attacks, 1000 test sentences), on every column at once.

## Where we stand

| | Paper TMD | Our TMD | enc_adv3 (n=200) | Needed to *beat* (paper + noise, n=1000) |
|---|---|---|---|---|
| Clean | 94.29 | 93.8 | 93.7 | keep within ~0.8 (a tie is the realistic goal) |
| PWWS | 70.0 | 64.0 | 56.0 | > ~74 |
| TextFooler | 50.0 | 50.2 | 51.0 | > ~54 |
| BERT-Attack | 55.2 | 53.0 (n=200) | 56.0 | > ~59 |

"Beat" here means above the paper's number by more than the combined 95% sampling noise (about 4 points per attack
column at n=1000), which is what `compare()` checks. Our TMD numbers are n=1000 except BERT-Attack.

**The bottleneck is PWWS.** enc_adv3 is already level with the paper on TextFooler and BERT-Attack, but needs about
+18 points on PWWS. Our own TMD reproduction is also 6 below the paper there.

## Why PWWS is hard to beat honestly

TMD's 64-70 on adaptive PWWS is largely *masking*. The same TMD keeps only 10% on PWWS transfer, and its PWWS score
moves 19 points with the attack implementation (pip rebuild 47 vs fork 66), while vanilla does not move. The fork's
PWWS ranks every word once, on noisy scores, then commits to that order, so a random defense derails it. A
deterministic defense that really cleans inputs (enc_adv3) does not get that free help.

So beating 70 on PWWS needs one of three things: (a) genuinely much stronger robustness, (b) the same kind of
randomness TMD benefits from, or (c) a stronger base model underneath.

## Options, cheapest first

| # | Idea | What changes | Expected effect | Cost (L4) | Risk / honesty note |
|---|---|---|---|---|---|
| 0 | **Close our own PWWS offset**: learning-rate grid for the BERT fine-tuning, as the authors did | base model | +2-6 on PWWS for every defense built on it (vanilla is 35.3 vs paper 39.2) | ~1 h per learning rate, then GAN search + encoder rounds again (~4 h) | none: it is the paper's own recipe; makes the reproduction tighter too |
| 1 | **Stochastic cleaned projection**: z = E(t,c) + sigma * noise, *no* "closest to input" selection; tune sigma on validation clean accuracy | inference only, no retraining | adaptive PWWS could rise toward TMD's level while staying near the cleaned point; TF/BA probably unchanged or up | minutes to implement, ~1 h to evaluate at n=200 | gains on PWWS would be masking, like TMD's: must be reported with the transfer test and an EOT attack (row 5) |
| 2 | **More adversarial rounds focused on PWWS** (enc_adv4/5: PWWS and TextFooler pairs against the current encoder, more pairs, keep the best epoch by validation) | encoder training | genuine robustness gains; adaptive TextFooler rose 32 -> 51 over the first rounds, PWWS is the column with room | ~1-1.5 h of pair generation per round | slow, uncertain size; still the most defensible route |
| 3 | **Adversarially fine-tuned BERT underneath the encoder** (training on attack examples of the train split, or the synonym augmentation already in notebook 04) | base model + encoder | raises every column, including vanilla | ~2-4 h fine-tuning + regenerate GAN/encoder | a different defense stack; legitimate (Table 1's baselines also change BERT's training), but must be described as such |
| 4 | **Unfreeze the classifier head jointly with the encoder** (projection-aware head, notebook 02 idea) | encoder training | small, mostly clean accuracy and TF/BA | short | low |
| 5 | **Randomness-aware (EOT) attack**: the attacker averages the defense's output over several projections per query (Athalye et al. 2018, cited by the paper) | evaluation only | not a way to raise our numbers; shows how much of TMD's 70 survives a stronger attack | ~1-2 h | a stronger attack than the paper's, so reported *next to* Table 1, not instead of it |

## Assessment

- **Beating all of Table 1 with genuine robustness alone (options 0, 2, 3, 4):** possible but not likely soon. The
  PWWS gap (+18 over enc_adv3, +10 over our TMD) is large, and each step costs hours of pair generation and retraining.
  Option 3 is the one with real upside on every column.
- **Beating it by adding randomness (option 1):** plausibly reaches TMD-level PWWS cheaply, and may edge past it on
  TF and BA. But it would win the same way TMD wins on PWWS, by confusing the attack. If we do it, the report must show
  the transfer test and the EOT attack alongside, or the claim is not defensible.
- **The strongest honest claim may be the reverse framing:** under a randomness-aware attack (option 5), TMD's PWWS
  number likely collapses while enc_adv3 should not. Then the encoder beats TMD on every column under a stricter attack,
  while matching it under the paper's.

## Suggested order once the n=1000 run is in

1. Option 1 at n=200 (an hour): shows whether PWWS can reach ~70 without retraining.
2. Option 5 on TMD and enc_adv3 at n=200: measures how much of each defense's score is masking.
3. Decide with these numbers: if option 1 + an EOT-robust encoder clears all columns, confirm at n=1000 and two seeds;
   otherwise invest in options 0 + 3 (the expensive route) or report the matched-on-the-paper's-protocol,
   better-under-stricter-attacks result.

Any "beats Table 1" claim needs: n=1000, the authors' attack code, at least two training seeds for the encoder, and the
transfer / EOT numbers reported next to it.
