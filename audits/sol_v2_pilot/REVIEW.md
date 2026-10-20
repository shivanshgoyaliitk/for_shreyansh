# Sol v2 pilot audit

The saved provenance identifies this run as **GPT-5.6 Sol, low/Light reasoning, Standard speed**, not GPT-6 Sol. Model identity is recorded from the producing session; this audit verified the saved files and hashes, not the host configuration independently.

All 58 report IDs match the blinded input in order; field sets, allowed states, prompt/schema/input hashes and provenance hashes pass. No pseudo labels were produced. A frozen prediction snapshot and SHA-256 are retained here. Original model outputs were not modified.

652 of 696 finding decisions are observed (93.68%); 44 abstain. Macro AUC on each finding's covered subset is 0.78301. With unsure assigned 0.5 for all-case diagnostic evaluation only, macro AUC is 0.80484. Training must still treat unsure as unobserved, not as a neutral label. Covered AUC for binary decisions equals balanced accuracy when both classes are represented.

The frozen ORIGINAL Gemini pilot had 90.37% coverage, covered macro AUC 0.78163, and neutral-imputed macro AUC 0.81637. On the same non-abstained cases per finding shared between the models, macro AUC is 0.77763 for Sol versus 0.77675 for original Gemini. Thus the available evidence does not establish a meaningful winner. Gemini's corrected pilot, if any, is not represented in these numbers.

Covered-subset sensitivities remain low for lateral OA (0.375), PF OA (0.385), effusion (0.440), and synovitis (0.481). These descriptive results are on only 58 previously used development studies, not an unbiased final test. They may reflect report omissions, the specificity-oriented prompt, image/text disagreement, or interpretation errors; these metrics alone cannot identify the cause.

## Usage

The pilot spans a five-hour reset and an overnight interruption. Primary usage went from 77% in the old window to 3% in a new window: this is not a valid subtraction. The last eight reports had a 1%-to-3% change in one window, including session overhead and potentially shared activity. Weekly usage went from 12% to 16% over the whole interval, but this is account-wide and not task-attributable. Do not project an exact cost or full-run allowance from either change. This pilot does not provide a reliable usage estimate for GPT-6 Sol either, because it used GPT-5.6 Sol.

## Recommendation

Keep the completed Sol pilot and wait for the other v2 pilots/corrected Gemini audit before authorizing all 4,349 reports. Compare frozen predictions with consistent coverage metrics. If Sol remains a candidate, measure a bounded uninterrupted run on the intended exact model/settings in a fresh task while other Codex work is idle. Do not interpret the interrupted pilot as proof that the full dataset fits current limits.
