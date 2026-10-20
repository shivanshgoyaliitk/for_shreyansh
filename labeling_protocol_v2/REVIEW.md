# Prompt review — 2026-09-27

The former prompt is not adequately aligned with the competition targets. This was a substantive criteria error, not just a wording issue. It also contradicted itself on hedging and mixed severity with certainty. Earlier assurances that the inherited prompt was ready to reuse were too strong.

Primary source: [competition anatomy and annotation guide, Label Description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343). Its detailed definitions were retrieved through indexed page text; direct page extraction returned no body. The [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) independently identifies the twelve targets. No human training labels were inspected to construct v2.

The v2 prompt separates official image-target definitions from our explicit text-extraction conventions. This matters because the radiology report may not contain all information available to image annotators. Missing threshold details abstain; genuinely subthreshold or diagnostically hedged findings are target-negative. Unmentioned findings remain weak negatives in usable routine reports. These are choices to evaluate, not claims that text can recover every human image label.

The output uses only definite/absent/unsure, compatible with existing 1/0/NaN conversion. Existing probable/possible values belong to the legacy protocol. Do not combine the two protocols as interchangeable model votes. In particular, do not fill the unfinished legacy Sonnet cache with v2 labels. A larger model cannot repair contradictory or incorrect task definitions on its own.

## Deployment

The old prompt and caches remain intact. Every labeling folder now has a separate v2 prompt, strict v2 schema, manifest, and instructions that take precedence on the next read. New results go to output_v2/, including a new 58-report pilot. A currently running agent may still have the old instructions in context: explicitly stop it and tell it to reread agent.md before continuing under v2. Do not delete its existing work.

No automatic annotation calls or paid experiments were launched during this review. The legacy Kaggle notebook still uses its old prompt; do not resume paid legacy extraction expecting it to follow v2. Updating its generation/consensus path is a separate migration after validation.

## Evaluation required before full relabeling

Freeze predictions before access to human targets. Evaluate the same 58 study IDs for each new model/protocol, reporting per-finding positive/negative counts, abstention coverage and discrimination only where classes exist. Evaluate binary sensitivity/specificity and balanced accuracy as well as AUC; report covered counts so abstention cannot masquerade as improvement. Where both legacy and v2 predictions exist, make paired comparisons on the same IDs and report missingness. Small-sample uncertainty is substantial.

Those 58 cases have already influenced earlier development: they are a development audit, not a pristine final test. Do not repeatedly tune examples against their errors and call the resulting score unbiased. Add independent reviewed cases if available before claiming superiority.

The synthetic cases supplied here check interpretation of the written protocol only. They have not been run through any model and cannot demonstrate clinical accuracy. No prompt can be certified “best possible” before empirical comparison.
