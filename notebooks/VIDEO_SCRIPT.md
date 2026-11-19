# Video narration script (for ElevenLabs)

One block per slide, in deck order. Paste each block into ElevenLabs separately, so each audio clip lines up with its slide when you edit the video. The text is written for speech: acronyms are spelled out with spaces (T M D, P W W S) so the voice reads them letter by letter, and "Bert" and "Text Fooler" are written the way they are said.

Total: about 725 words, roughly 4.5 minutes at ElevenLabs' default pace (the limit is 5). If the clips add up to more than 4:45, shorten slides 3 and 6 first.

---

## Slide 1: cover (about 10 seconds)

This is our project for Deep Learning, A I sixty-one-oh-three. We extend Textual Manifold-based Defense, or T M D, a text defense published at E M N L P twenty twenty-two, and turn it into a deterministic, attack-aware defense.

## Slide 2: a few swapped words flip the label (about 25 seconds)

Text classifiers can be fooled by changing just a few words. An attacker swaps words for near-synonyms. A person still reads the same meaning, but the model changes its answer. Take the A G News dataset, where each news snippet must be sorted into one of four topics. A standard Bert model puts ninety-four percent of normal sentences in the right topic. But when an attacker is allowed to swap a few words, it gets only twenty-nine to thirty-six percent right.

## Slide 3: TMD (about 35 seconds)

T M D defends in three steps. First, Bert turns each sentence into an embedding: a point in space. Second, a generative adversarial network learns where the embeddings of normal, unattacked sentences lie. This region is called the manifold. Third, at test time, T M D draws thirty random points from that network, keeps the one closest to the input, and classifies that point instead. The idea is that an attack pushes a sentence off the manifold, and projecting it back undoes the attack. T M D is trained on normal sentences only. It never sees an attack.

## Slide 4: TMD's score depends on the attack (about 30 seconds)

We first reproduced T M D with the authors' own attack code. Our results are close to the paper on two of the three attacks, and lower on P W W S. Then we found a weakness. Under the P W W S attack, T M D still gets sixty-three percent right when the attacker queries it directly. But when we take sentences that were already attacked on a plain Bert model and give them to T M D, it gets only eleven percent right. So its score depends heavily on how the attack is run.

## Slide 5: our defense, two changes (about 35 seconds)

Our defense makes two changes. First, we replace the random search with a learned encoder. It predicts the right point on the manifold in a single step. The result still goes through T M D's own generator, so it stays on the manifold, and there is no randomness. Second, we also train Bert on attacked training sentences, using the attacks only as black boxes. This is the key difference from T M D: our defense learns from attacks, and T M D does not.

## Slide 6: far ahead of attack-aware defenses (about 40 seconds)

Because our defense learns from attacks, the fair comparison is with the paper's two adversarial-training defenses, A S C C and D N E. Against them, our defense is far ahead on all three attacks, by nineteen to forty-seven points, with both of our training seeds. Compared with T M D, we are ahead on Text Fooler and Bert-Attack with both seeds. On P W W S, our first seed is ahead and our second seed ties. On normal sentences it stays as accurate as before, at about ninety-five percent.

## Slide 7: both changes are needed (about 30 seconds)

Is the gain just from training on attacks? This ablation says no. With the same adversarially trained Bert, with no projection it gets about fifty percent right under attack. Adding T M D's random search raises that to sixty-six. Our encoder raises it to eighty. The training data is the same in all three cases. The projection makes the difference.

## Slide 8: attacks it never saw (about 30 seconds)

We also checked whether it only works on the attacks it was trained on. Each row here was trained on one or two attacks and tested on all three. Trained only on Bert-Attack, it scores sixty-nine on Text Fooler, against fifty for T M D. Trained only on Text Fooler, it scores sixty-five on Bert-Attack, against fifty-five. On P W W S, it ties T M D after one attack, and is above it after two.

## Slide 9: a new kind of attack, typos (about 30 seconds)

Finally, we tried a completely different attack: Deep Word Bug, which makes typos instead of swapping words. None of our defenses was trained on it. T M D gets fifty-five percent right. Our defenses get seventy-two to eighty-one percent right, and every version is significantly better than T M D. So the robustness is not limited to word swaps.

## Slide 10: what to take away (about 35 seconds)

So what did we learn? Replacing T M D's random guess with a learned step, on top of a Bert that has seen attacks, gives a much stronger defense. It is far ahead of the paper's other attack-aware defenses, and it holds up against typos it never saw. It also gives the same answer every time, and runs almost as fast as plain Bert. The limits: it learns from attacks and T M D does not, we tested one dataset and one model, and our two training runs differ by about eight points. Our main lesson: a defense should learn where to project, not guess. Thank you.

---

## Tips for ElevenLabs

- Generate one clip per slide, then place each clip under its slide in your video editor. Hold each slide about half a second after its clip ends.
- If the voice still mispronounces an acronym, add dots between the letters ("T. M. D.").
- For a pause inside a block, models that support it accept `<break time="0.5s" />`.
- Listen through once against the slides: every number in this script matches what is on screen.
