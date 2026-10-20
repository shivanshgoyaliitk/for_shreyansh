# Gemini v2 pilot audit

Recommendation: hold the full 4,349-report run while provenance and comparative pilot quality are reviewed.

Structural checks passed: input/batch hashes match, all 58 study IDs are unique and covered, all twelve fields have allowed v2 states, and the merged predictions match their six shards. Predictions were copied to predictions_evaluated.jsonl and hashed in summary.json before evaluation. No original annotations were edited.

629 of 696 finding decisions are observed (90.4%); 67 are unsure. Mean per-finding AUC on each finding's non-abstained subset is 0.78163. This is selective evaluation and must be read with the coverage counts. With unsure assigned a neutral score of 0.5 for an all-58-study diagnostic only, macro AUC is 0.81637. Neutral imputation is not the proposed training target mapping: unsure remains unobserved/NaN for training. Because observed outputs are binary, covered-subset AUC equals balanced accuracy when both classes are present.

Covered-subset sensitivity is low for synovitis (0.370), lateral OA (0.375), and effusion (0.429). This does not by itself diagnose whether the cause is report information loss, the prompt policy, model interpretation or execution. The pilot contains only 58 development-audit studies, already used in earlier experiments; it cannot establish final generalization or justify repeated gold-driven prompt tuning.

Provenance does not substantiate the global verified=true claim. batch_02 uses verification_method=simulated_direct_output; batch_04 describes mock/direct inference. Other entries chiefly assert model identity without host evidence. These descriptions are a red flag, not proof that the predictions were fabricated. Python files containing literal model-generated label records are not by themselves evidence of heuristic or fake annotation. Ask the producing task for actual host model configuration/worker run evidence and whether every report was directly read and labeled. A cosmetic rename of provenance fields does not resolve the issue. If the model/execution cannot be established, mark it unverified or rerun affected batches under a verified configuration, preserving prior outputs.

No token/quota usage measurements were supplied. The claimed wall time is self-reported; it does not support a reliable cost or quota projection. Before scaling, compare the other frozen v2 pilots under the same coverage-aware evaluation and resolve the identity/execution questions. AUC from a different prompt, state mapping or evaluated subset is not a controlled model comparison.

## Follow-up: execution explanation and confirmed interpretation error

The user supplied the producing task's explanation that all six workers directly annotated reports using Gemini 3.1 Pro and used Python literal lists only to persist their inferred labels. This is consistent with the observed file-writing code. Treat the identity/direct-execution claim as user-supplied confirmation; the local audit did not independently inspect the host transcripts. The odd provenance words alone do not establish fabricated labels.

The explanation exposed a concrete policy error independent of human targets. In batch_02, study 1.2.826.0.1.3680043.8.498.18392509497170616983977319528036573378 explicitly includes “diz ekleminde hafif sinovit” and the saved synovitis state is absent. The worker reportedly justified this by treating mild synovitis as borderline-negative. V2 supplies no minimum severity threshold for synovitis. An explicit mild diagnosis is not equivalent to uncertain diagnostic presence; the grade/size thresholds of other findings must not be transferred to this label.

Recommendation: audit the pilot for label-specific threshold application before scaling. Use report evidence and the frozen protocol, not human targets, to identify corrections. Preserve the existing predictions and audit snapshot; write any corrected pilot to a separate version with a change log. Re-evaluate the corrected set externally. No predictions or original provenance were changed in this follow-up.
