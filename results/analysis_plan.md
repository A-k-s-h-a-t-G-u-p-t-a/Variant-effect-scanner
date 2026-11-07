# Analysis plan (written 2026-11-08, before Phase A-C analyses were run)

Question: does ESMC-300M masked-prediction entropy add information about functional-site residues beyond MSA conservation?

## What was already seen (so it cannot count as confirmatory)
Task 3 on the 126 development proteins: cons+entropy vs cons, pooled AUC +0.009 (95% protein-level interval +0.003 to +0.016). That comparison, and every number in `results/task3_report.md`, is **exploratory**.

## Primary comparison for all new analyses (fixed now)
- Contrast: (best conservation variant + ESMC-300M entropy) minus (best conservation variant alone), logistic regression on standardised features, class-balanced, grouped 5-fold CV by MMseqs2 cluster.
- Metrics: pooled average precision (AP) and pooled AUC, each with a 95% protein-level bootstrap (1,000 resamples, seed 0, paired).
- "Best conservation variant" is chosen from the A3 sweep as the variant with the highest cross-validated score **on the 126 development proteins only**, and is then frozen.

## Confirmatory test
The C1 proteins (new M-CSA-labelled proteins not among the 126) are the held-out set. The model class, features, hyper-parameters and conservation variant are frozen from the development set; C1 is evaluated once, with no refitting of choices. If C1 cannot be completed in time, the draft states that no confirmatory test was run.

## Multiple comparisons
The A1-A3 contrasts, the depth strata (B1) and the label-type split (C2) form one family; Holm correction is applied across them and both raw and adjusted values are reported. Anything not pre-listed here is labelled exploratory.

## Rules
- Features are never tuned on C1 or on the labels of the held-out folds.
- Results that disagree with the headline are reported with equal prominence.
- No clinical claims; scores are research hints.

## C3 (annotation-free labels; added 2026-11-08 before any C3 analysis was run)
- Data: the 9 ProteinGym assays already used in Gate 4 (ProteinGym v1.3 singles only), ESMC-300M scans already cached, alignments fetched from ColabFold.
- Position label: a position is "sensitive" if >= 50% of its measured substitutions are binarised as deleterious (`DMS_score_bin == 0`); positions with fewer than 5 measured substitutions are dropped. No threshold tuning.
- Features: ESMC-300M entropy and the frozen best conservation variant from Phase A (plus the default `cons` for reference).
- Evaluation: per-assay AUC and AP; leave-one-assay-out logistic regression for [cons], [entropy], [cons, entropy]; paired difference (cons+entropy minus cons) with a bootstrap over assays (only 9 proteins, so intervals will be wide and the result is a consistency check, not a test).

## C2 (label types; added before analysis)
- Split the 126-protein UniProt label into active-site residues and binding-site residues (UniProt feature type, same evidence rules). Same frozen features, same grouped CV, paired bootstrap of cons+entropy minus cons within each label type.
