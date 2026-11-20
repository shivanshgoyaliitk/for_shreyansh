# Report fixes: find and replace

Each "Find" is copied from the current .tex text. In Overleaf, search for the first few words of a Find (line breaks in the source may differ), select the whole passage and paste the Replace. Numbers in brackets refer to the review list.

---

## main.tex

**[18] Page-limit comment (line 2)**

Find:
```
%% AI6103 Group Project Report -- AAAI format (max 6 pages incl. refs)
```
Replace:
```
%% AI6103 Group Project Report -- AAAI format (max 6 pages excl. refs)
```

**[10] AI-use statement**: insert directly above `\bibliography{references}`:
```latex
\section*{Use of AI Tools}
The code for this project was written with the help of an AI coding
assistant (Claude, Anthropic). All design decisions, code and results
were checked by the team, and we can explain each of them.
```
(Cengiz is an author, so the encoder design needs no separate credit here. Contributions go in the peer review.)

---

## abstract.tex

**[1] Last sentence**

Find:
```
On AG News, our
defense achieves higher accuracy than TMD under all three attacks we
evaluate, and it also stays more robust against an attack it never saw
during training.
```
Replace:
```latex
On AG News, our defense is far more robust than the
adversarial-training defenses reported in the TMD paper (ASCC and DNE)
under all three attacks. Compared with TMD, it is more accurate under
TextFooler and BERT-Attack with both training seeds, and under PWWS
with one of the two seeds (the other ties). It also stays more robust
than TMD against an attack it never saw during training.
```

---

## introduction.tex

**[16] Description of adversarial training**

Find:
```
The first is
adversarial training: attacked sentences are added to the training
data, so that the model learns to classify them correctly.
```
Replace:
```latex
The first is
adversarial training: the model is trained on perturbed inputs, either
attacked sentences or, as in ASCC and DNE, combinations of each word's
synonyms, so that it learns to classify them correctly.
```

**[14] Weakness (iii)**

Find:
```
and choosing the candidate closest to an adversarial input tends to keep the
perturbation.
```
Replace:
```latex
and we hypothesise that choosing the candidate closest to an
adversarial input tends to keep part of the perturbation.
```

---

## related_work.tex

**[7] Add DeepWordBug**: at the end of the "Word-substitution attacks" paragraph, after the BERT-Attack sentence, add:
```latex
DeepWordBug~\citep{gao2018deepwordbug} instead changes characters
inside words, for example by swapping or deleting letters.
```

**[7] Defenses paragraph**

Find:
```
As discussed in the introduction, defenses either retrain the model on
adversarial examples or change the input before classification.
TMD~\citep{nguyen2022tmd}, which our work builds on, belongs to the
second group; we describe it in detail in Section~\ref{sec:background}.
```
Replace:
```latex
Defenses either make the model itself robust or change the input
before classification. ASCC~\citep{dong2021ascc} and
DNE~\citep{zhou2021dne} are adversarial-training defenses that train
the model on the convex hull of each word's synonyms.
TRADES~\citep{zhang2019trades} adds a KL term that keeps the
prediction on a perturbed input close to the clean prediction; we use
this term when fine-tuning BERT. SAFER~\citep{ye2020safer} is a
certified defense based on randomized smoothing over random synonym
substitutions. TMD~\citep{nguyen2022tmd}, which our work builds on,
changes the embedding instead; we describe it in detail in
Section~\ref{sec:background}. We evaluate all defenses with the
benchmark protocol of \citet{li2021textdefender}, which uses the
TextAttack framework~\citep{morris2020textattack}.
```

---

## experimental_setup.tex

**[5] Dataset paragraph**

Find:
```
All final accuracies are reported on the
test set, and each experiment is run once on the test set.
```
Replace:
```latex
Final results are reported on the test
set, and each final model is attacked once on it. Model selection and
the ablations in Tables~\ref{tab:ablation} and~\ref{tab:controls} use
200 sentences from a validation split (10\% of the training set),
never the test set.
```

**[6] + [7] Attacks paragraph**

Find:
```
Besides the three attacks in Section~\ref{sec:background}, we also
evaluate on DeepWordBug, an attack that our defense never sees during
training. All attacks use the settings of the TMD paper, described in
Section~\ref{sec:method-pairs}.
```
Replace:
```latex
Besides the three attacks in Section~\ref{sec:background}, we also
evaluate on DeepWordBug~\citep{gao2018deepwordbug}, a character-level
attack that our defense never sees during training. All attacks use the
TMD authors' attack code and the protocol of
\citet{li2021textdefender}: 50 candidate words per position, at most
30\% of the words changed, a USE similarity of at least 0.84, and a
query budget of 50 times the number of words. We attack 1000 random
test sentences with more than four words. Accuracy under attack is the
share of these sentences that are still classified correctly after the
attack. Clean accuracy uses the full test set of 7,600 sentences.
```

**[6] New "Statistics" paragraph**: add after the Attacks paragraph, before Hardware:
```latex
\paragraph{Statistics.}
When we compare with numbers from the TMD paper, we count a difference
as real only if it is larger than the combined 95\% sampling noise of
the two 1000-sentence samples, about $\pm4$ points. When we compare
with our own TMD reproduction, both defenses are attacked on the same
sentences, so we use an exact McNemar test on the sentences that only
one of the two defenses survives.
```

---

## figure_embedding.tex

**[17] Stray "d" in the caption**

Find:
```
d  with the TMD authors' attack code on 1000 test sentences.}
```
Replace:
```latex
with the TMD authors' attack code on 1000 test sentences.}
```

---

## method.tex

**[12] + [7] BERT fine-tuning: optimizer and TRADES**

Find:
```
The cross-entropy (CE) is computed on every sentence in the batch. We
train for 2 epochs with Adam with a learning rate of $10^{-5}$,
```
Replace:
```latex
The cross-entropy (CE) is computed on every sentence in the batch. The
KL term follows TRADES~\citep{zhang2019trades}, and the cosine term
keeps the embedding itself stable, because our projection works on the
embedding. Both weights are 1 and were not tuned. We train for 2 epochs
with AdamW with a learning rate of $10^{-5}$,
```

**[4] How the attack rounds were collected**

Find:
```
We collect these pairs in rounds. In
each round, we attack the model from the round before. In this way, the
later rounds contain the attacks that still work against our defense.
```
Replace:
```latex
We collect these pairs in two rounds.
The first round uses attacks on the plain BERT and on our earlier
encoder defenses, which used a frozen BERT. After the first adversarial
fine-tuning, the second round attacks the new BERT itself, so that the
added pairs are attacks that still work against it.
```

---

## results.tex

**[13] Reproduction paragraph**

Find:
```
The undefended (vanilla) BERT is reproduced
closely.
```
Replace:
```latex
Our undefended (vanilla) BERT is close to the
paper in clean accuracy and under TextFooler, but 3--4 points lower
under PWWS and BERT-Attack.
```

**[11] The PWWS TODO**

Find:
```
% TODO: add one sentence explaining why the PWWS numbers are lower.
```
Replace:
```latex
The gap is likely due to fine-tuning: the authors chose BERT's learning
rate by grid search, while we used a single learning rate, and our
vanilla BERT is also lower under PWWS. Training TMD's InfoGAN with the
paper's own hyperparameters did not close the gap (48.0 / 48.5 vs.\
50.5 / 57.0 under TextFooler / BERT-Attack on validation).
```

**[1] Main results paragraph**

Find:
```
It also beats
TMD, which is the best result reported in the paper, under all three
attacks, while keeping clean accuracy as high as or higher than the
baselines.
```
Replace:
```latex
Compared with
TMD, the best defense in the paper, both seeds are ahead under
TextFooler and BERT-Attack, by more than the sampling noise of about
$\pm4$ points. Under PWWS, seed~1 is ahead (+11.4) and seed~2 ties
(+2.5). Against our own TMD reproduction, which was attacked on the
same sentences, both seeds are better under every attack
($p < 0.0001$, McNemar test). Clean accuracy stays level with the
baselines.
```

**[2] "Both parts are needed" paragraph**

Find:
```
Each part alone improves on TMD, but much less than the full defense;
the large gain appears only when they are combined.
```
Replace:
```latex
Each part alone gives a smaller or mixed gain. The adversarially
trained BERT without a projection is better than TMD under TextFooler
(55.5 vs.\ 50.5) but worse under BERT-Attack (47.5 vs.\ 57.0), and our
encoder on the plain BERT improves on TMD only moderately. The large
gain appears only when both parts are combined.
```

**[5] + [8] Ablation table caption**

Find:
```
  \caption{Ablation: accuracy under TextFooler / BERT-Attack (\%) for
  each combination of BERT and projection.}
```
Replace:
```latex
  \caption{Ablation on 200 validation sentences: accuracy under
  TextFooler / BERT-Attack (\%) for each combination of BERT and
  projection. The plain-BERT encoder comes from an earlier experiment;
  a second seed of it scored only 37.0 / 45.0.}
```

**[5] Controls table caption**

Find:
```
  \caption{Controls: accuracy under TextFooler / BERT-Attack (\%).}
```
Replace:
```latex
  \caption{Controls on 200 validation sentences: accuracy under
  TextFooler / BERT-Attack (\%).}
```

**[3] Unseen-attacks paragraph**

Find:
```
Training on any single attack already beats TMD on all three attacks,
including the ones not used in training, and training on all three gives
the best results.
```
Replace:
```latex
Training on a single attack already beats TMD under the other
word-substitution attack that was not used in training: 65.4 under
BERT-Attack after training on TextFooler, and 68.6 under TextFooler
after training on BERT-Attack. Under the unseen PWWS, the single-attack
models tie TMD (72.3 and 72.8 vs.\ 70.0), and after training on two
attacks PWWS rises to 76.4 (one seed). In this protocol PWWS uses the
same candidate words as TextFooler, so it is the weaker test of
generalisation. Training on all three attacks gives the best results,
but this model also had more training data, so the gain is not only
due to the number of attacks.
```

**[11] The transfer TODO**

Find:
```
% TODO (optional): add the transfer table if the team decides to include it.
```
Replace:
```latex
\paragraph{Transfer from an undefended model.}
We also give each defense the adversarial sentences that already fooled
the undefended BERT, without attacking the defense itself. TMD
classifies only 11.4\% of the PWWS sentences correctly (23.0\% for
TextFooler and 25.9\% for BERT-Attack), although it keeps 63.2\% under
the attack that queries it. Our defense keeps 92.9\%, 93.1\% and
92.0\%. TMD's robustness therefore depends strongly on how the attack
is run, while ours does not.
```

**[1] Cost paragraph, last sentence**

Find:
```
Our method is therefore better than TMD in
both accuracy and speed.
```
Replace:
```latex
Our method is therefore faster than TMD and,
under TextFooler and BERT-Attack, also more accurate.
```

**[15] Cost table caption**

Find:
```
  \caption{Inference time of each defense.}
```
Replace:
```latex
  \caption{Inference time of each defense on 1000 test sentences, and
  relative to vanilla BERT.}
```

---

## limitations.tex

**[9] Last sentence**

Find:
```
Finally, the DeepWordBug comparison is made
against our own reproduction of TMD, not against numbers from the
paper.
```
Replace:
```latex
The DeepWordBug comparison is made against
our own reproduction of TMD, not against numbers from the paper.
Only the encoder was trained with two seeds; BERT's adversarial
fine-tuning was run once. The numbers for ASCC, DNE, SAFER and TMD are
taken from the TMD paper, not re-run by us. Finally, the weights of our
loss terms were not tuned.
```

---

## conclusion.tex

**[1] First sentence**

Find:
```
We showed that a learned, deterministic manifold projection on top of
an adversarially trained BERT is both more robust and more efficient
than TMD.
```
Replace:
```latex
We showed that a learned, deterministic manifold projection on top of
an adversarially trained BERT is far more robust than the paper's
adversarial-training defenses, more robust than TMD under TextFooler
and BERT-Attack (and under PWWS for one of two seeds), and about twice
as fast as TMD.
```

---

## references.bib

**[7] Add these seven entries.** Check that none of the keys already exist in your file.

```bibtex
@inproceedings{dong2021ascc,
  title     = {Towards Robustness Against Natural Language Word Substitutions},
  author    = {Dong, Xinshuai and Luu, Anh Tuan and Ji, Rongrong and Liu, Hong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{zhou2021dne,
  title     = {Defense against Synonym Substitution-based Adversarial Attacks via {D}irichlet Neighborhood Ensemble},
  author    = {Zhou, Yi and Zheng, Xiaoqing and Hsieh, Cho-Jui and Chang, Kai-Wei and Huang, Xuanjing},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2021}
}

@inproceedings{ye2020safer,
  title     = {{SAFER}: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions},
  author    = {Ye, Mao and Gong, Chengyue and Liu, Qiang},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2020}
}

@inproceedings{gao2018deepwordbug,
  title     = {Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers},
  author    = {Gao, Ji and Lanchantin, Jack and Soffa, Mary Lou and Qi, Yanjun},
  booktitle = {IEEE Security and Privacy Workshops (SPW)},
  year      = {2018}
}

@inproceedings{li2021textdefender,
  title     = {Searching for an Effective Defender: Benchmarking Defense against Adversarial Word Substitution},
  author    = {Li, Zongyi and Xu, Jianhan and Zeng, Jiehang and Li, Linyang and Zheng, Xiaoqing and Zhang, Qi and Chang, Kai-Wei and Hsieh, Cho-Jui},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2021}
}

@inproceedings{morris2020textattack,
  title     = {{TextAttack}: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in {NLP}},
  author    = {Morris, John and Lifland, Eli and Yoo, Jin Yong and Grigsby, Jake and Jin, Di and Qi, Yanjun},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year      = {2020}
}

@inproceedings{zhang2019trades,
  title     = {Theoretically Principled Trade-off between Robustness and Accuracy},
  author    = {Zhang, Hongyang and Yu, Yaodong and Jiao, Jiantao and Xing, Eric and El Ghaoui, Laurent and Jordan, Michael I.},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```

---

## After pasting

1. Recompile twice so the new citations and table references resolve.
2. Check the page count with the official AAAI style: at most 6 pages, references excluded. The new text adds roughly a third of a page. If you run over, shorten the controls table first.
