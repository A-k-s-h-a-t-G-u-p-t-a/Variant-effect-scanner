# Does protein language model confidence add to evolutionary conservation for locating functional sites? A pre-registered zero-shot analysis

*Draft v1, 2026-11-08. Items marked [PENDING] depend on the held-out C1 scan and are not yet filled. References are listed at the end as suggestions to verify; no citation details below should be trusted until checked.*

## Abstract

Protein language models (PLMs) predict the amino acid at a masked position, and the entropy of that prediction has been proposed as a signal of functionally important residues. Whether it carries information beyond ordinary evolutionary conservation is untested in our setting. We score all residues of 126 reviewed human proteins (100-350 aa; 32,402 residues, 689 experimentally supported active or binding site residues) with ESM Cambrian 300M, without any training or fine-tuning, and compare masked-prediction entropy with multiple-sequence-alignment (MSA) conservation, AlphaFold structure features, and two larger PLMs, using cross-validation grouped by sequence cluster and protein-level bootstrap intervals. On its own, entropy ranks functional sites about as well as conservation (pooled AUC 0.843 vs 0.848 for Shannon conservation and 0.854 for Jensen-Shannon divergence; difference entropy minus conservation -0.008, 95% interval -0.026 to +0.010). Adding entropy to the strongest conservation score gives a small gain (AUC +0.008, interval +0.002 to +0.013; average precision +0.006), positive in 34 of 37 pre-registered contrasts but significant after Holm correction in only 3. The gain is concentrated in binding-site residues and shallower alignments and absent for catalytic residues, and structure features add nothing. Absolute precision is modest: with 2.1% positives, the best model's top 10% of residues per protein contain about 76% of the sites but only about 13% of those residues are sites. We conclude that PLM entropy mostly re-expresses conservation, plus a small additional signal. [PENDING: held-out confirmatory result on 24 M-CSA proteins.]

## 1. Introduction

Zero-shot variant-effect prediction with PLMs scores a substitution by the change in the model's log-probability at a masked position [REF: ESM-1v, ESM-2]. The same masked distribution has a second use: its entropy is low where the model is confident, which one expects at residues that evolution constrains, such as catalytic and ligand-binding sites. This suggests PLM entropy as a way to locate functional sites. The obvious competitor is conservation computed from an MSA, which the PLM may largely have internalised. If so, the PLM adds convenience (no alignment needed) but no information.

The question we ask is narrow: does masked-prediction entropy from a PLM add information about functionally annotated residues beyond MSA conservation and simple structure features? Any answer is informative: entropy adds signal, entropy is a re-expression of conservation, or conservation wins. We state the primary comparison and the multiple-comparison handling before running the follow-up analyses, and treat only a held-out set as confirmatory.

## 2. Methods

### 2.1 Model and scores
We use ESM Cambrian 300M (Hugging Face `biohub/ESMC-300M-hf`; about 300M parameters, 30 layers) in float32, with no training. For each position i, the residue is replaced by the mask token and the model returns a distribution over the vocabulary; we restrict and renormalise it over the 20 standard amino acids. The entropy is -sum_a p_i(a) ln p_i(a) in nats (maximum ln 20 = 2.996). The substitution score for W to M is log p_i(M) - log p_i(W); one masked forward pass per position (L passes, not 20L). Scores for other models: ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) and ESM Cambrian 600M (`biohub/ESMC-600M-hf`), the latter two on a GPU in float32.

Correctness checks (gates) were kept passing throughout: the tokenizer offset is asserted for every model (one special token before residue 1 and one after the last residue, verified for each of the three models); known haemoglobin mutations have the expected sign (heme histidines H64Y and H93Y strongly negative, conservative K9R near zero); the heatmap shows conserved columns and a dark proline row; and mean Spearman on nine ProteinGym assays (proteins under 400 aa) is 0.455 against 0.454 published for the same model, with a maximum per-assay gap of 0.006. A GPU rescan of a 108-residue protein matched the CPU scan, and a CPU rescan after our code changes was identical to the cached scan.

### 2.2 Proteins and labels
Development set: reviewed human proteins of 100-350 residues in UniProt with at least one experimentally supported active site (149 entries), sorted by accession and shuffled with seed 0; 3 were dropped for non-standard residues and 20 more because their experimental evidence was not coded ECO:0000269 or ECO:0007744 or their annotations spanned more than 5 residues, leaving 126 proteins. No protein was chosen by hand. Positives are residues in Active site or Binding site features with experimental evidence codes ECO:0000269 or ECO:0007744 and spans of at most 5 residues (689 positives among 32,402 residues, 2.1%). Residues with similarity- or prediction-only site annotations are excluded as neither positives nor negatives, to avoid circularity. Of the positives, 181 are active-site residues and 545 binding-site residues; these add up to more than 689 because some residues carry both annotation types.

Sequences were clustered with MMseqs2 at 30% identity, giving 104 clusters (92 singletons; largest cluster 10 proteins).

### 2.3 Baselines
- **Conservation.** One MSA per protein from the public ColabFold MMseqs2 server (UniRef plus environmental sequences; 1,188-23,242 sequences, median 10,856), subsampled to at most 2,000 sequences with the query kept. Sequence weights use an identity threshold theta; frequencies use a pseudocount. Shannon conservation is (ln 20 - H) x (1 - gap fraction). Jensen-Shannon divergence follows Capra and Singh [REF] against a background amino-acid distribution, gap-penalised. We evaluated 12 variants (theta 0.62, 0.8, 0.9; pseudocount 1 and 5; subsample size 100, 500 and 2,000; optional window smoothing), and froze the best variant on the development set by cross-validated average precision (`jsd_t0.62_n2000`) before running the follow-up analyses.
- **Structure.** AlphaFold DB models; per-residue pLDDT and a burial proxy (number of C-beta atoms, C-alpha for glycine, within 10 Angstrom).
- **Other PLMs.** Entropy from ESM-2 650M and ESM Cambrian 600M, scanned as above.
- **Other ESMC-300M scores.** Probability of the true residue (p_wt), mean and minimum log-ratio over the 19 substitutions.

### 2.4 Evaluation
Single-feature scores are evaluated with the sign "low = site" for entropy and log-ratios and "high = site" for conservation, burial and pLDDT. Combined models are class-balanced logistic regressions on standardised features, and out-of-fold decision scores from 5-fold cross-validation grouped by sequence cluster (so homologues never straddle train and test) are pooled into one AUC and one average precision (AP) per feature set. Intervals are 95% percentile bootstraps over proteins (1,000 resamples, seed 0), with paired resamples so that differences between models use the same resamples. With only 2.1% positives we also report AP and, per protein, recall and precision among the top 10% of residues.

Robustness checks vary the classifier (logistic regression or a fixed-setting gradient-boosting model), the features (raw values, ranks within each protein, and +-3-residue window means) and the conservation variant.

### 2.5 Pre-registration and multiple comparisons
`results/analysis_plan.md` was written before the follow-up analyses were run. It fixes the primary comparison (the best conservation variant plus entropy, versus the same conservation variant alone) and the metrics (AUC and AP), states that the Task-3 comparison on the development set is exploratory because it was already seen, and designates a held-out set of proteins as the only confirmatory test. All 37 pre-registered contrasts (classifier and feature robustness, three pretraining-familiarity strata, and the two label types) form a single family with Holm correction. A fourth familiarity proxy (UniProt annotation score) was dropped before reporting because it equals 5 for every protein, so any split would be arbitrary.

### 2.6 Additional analyses
- **Pretraining familiarity (B1).** Terciles of: mean log-probability of the true residues under ESMC-300M (a pseudo-perplexity proxy of how familiar the sequence is), number of alignment sequences at at least 90% identity to the query, and number of PDB entries.
- **Label types (C2).** Active-site residues and binding-site residues analysed separately, with residues of the other type removed.
- **Annotation-free check (C3).** On the nine ProteinGym assays, a position is called sensitive if at least 50% of its measured single substitutions are binarised as deleterious (positions with fewer than 5 measurements are dropped; no threshold tuning). Leave-one-assay-out logistic regression; since there are only 9 proteins this is a consistency check, not a test.
- **Held-out confirmatory set (C1).** Reviewed human proteins of 100-350 residues with catalytic residues in M-CSA [REF], not among the 126 development proteins (24 proteins, 108 M-CSA residues). Models fitted once on the development set; nothing refit or tuned. Residues with experimentally supported UniProt site annotations that M-CSA does not list are excluded from this analysis.

## 3. Results

### 3.1 Entropy and conservation as single scores
Pooled AUC for locating functional sites, with 95% protein-level intervals:

| Score | AUC |
|---|---|
| Jensen-Shannon conservation | 0.854 (0.831-0.878) |
| Shannon conservation | 0.848 (0.825-0.872) |
| ESM Cambrian 600M entropy | 0.845 (0.821-0.870) |
| ESMC-300M entropy | 0.843 (0.818-0.867) |
| ESM-2 650M entropy | 0.819 (0.795-0.846) |
| p_wt (ESMC-300M) | 0.824 (0.798-0.851) |
| Mean log-ratio (ESMC-300M) | 0.823 (0.798-0.847) |
| Minimum log-ratio (ESMC-300M) | 0.791 (0.765-0.816) |
| Burial (contacts) | 0.692 (0.664-0.720) |
| pLDDT | 0.545 (0.508-0.583) |

The 0.843 for entropy replicates on the 101 proteins not used in an earlier 25-protein pilot (0.843 on those alone; 0.844 on the pilot), and is unchanged when one protein per cluster is kept (0.845). Entropy minus Shannon conservation is -0.008 (interval -0.026 to +0.010): the data do not show that entropy beats conservation, and conservation is nominally ahead. The two are strongly related (Spearman about -0.72).

### 3.2 Precision and recall
The average precision of the best conservation score is 0.181 (95% interval 0.157-0.215) against a chance level of 0.021, and the average precision of entropy alone is 0.134. Within each protein's top 10% of residues, the best conservation model recovers on average 76% of its sites, but only about 13% of those residues are sites. AUC alone therefore overstates how useful the ranking is for picking out individual residues.

### 3.3 Does entropy add information to conservation?
In cross-validation grouped by cluster, adding entropy to the strongest conservation score (`jsd_t0.62_n2000`) increases pooled AUC from 0.857 to 0.864 (+0.008, 95% interval +0.002 to +0.013) and AP from 0.181 to 0.187 (+0.006, +0.002 to +0.009). Against the Shannon conservation baseline used in our first analysis, the gain is +0.009 (+0.003 to +0.016). Across classifiers and feature transforms, the gain in AUC is between +0.005 and +0.011 and the gain in AP between +0.006 and +0.021, and the gradient-boosting model with window features reaches AUC 0.873 and AP 0.205 for conservation plus entropy against 0.863 and 0.184 for conservation alone. Of the 37 pre-registered contrasts, 34 have a positive point estimate and 12 have raw p below 0.05, but only 3 remain below 0.05 after Holm correction (recall in the top 10% in two configurations, and the AP gain in proteins with the most close homologues). The effect is consistent in sign but small and only partly robust to correction.

Structure features add nothing: conservation plus pLDDT and burial reaches AUC 0.844, against 0.847 for conservation alone, and the entropy gain is the same with or without them. Against the Shannon conservation baseline, ESM Cambrian 600M entropy gives the same gain as the 300M model (+0.009, +0.002 to +0.016, against +0.009 for the 300M model); ESM-2 650M gives a smaller, non-significant gain (+0.003, -0.001 to +0.008).

### 3.4 Where the gain is
- **Alignment depth** (exploratory; Shannon conservation baseline, depth terciles chosen after seeing the data). The gain in pooled AUC is +0.020 (+0.005 to +0.034) in the shallowest third of alignments (1,188-8,343 sequences), +0.009 in the middle third and +0.002 (-0.007 to +0.011) in the deepest third.
- **Label type.** Active-site residues are located almost perfectly by conservation (AUC 0.962) and entropy adds nothing (+0.0004, -0.006 to +0.004). For binding-site residues conservation reaches 0.824 and entropy adds +0.008 (-0.0003 to +0.016) in AUC and +0.0045 (0.0001 to 0.0094) in AP.
- **Pretraining familiarity.** The gain is small in every tercile of every proxy. It is not concentrated in the least familiar proteins; in AP it is larger in the most familiar tercile of the pseudo-perplexity proxy (+0.0066 against +0.0020) and in proteins with more than 15 close homologues (+0.0082 against about 0 for 6 or fewer). This is consistent with the concern that part of the gain reflects overlap with the PLM's training data, but we did not test the differences between terciles directly, and it cannot be separated from simpler explanations such as better-studied proteins having better annotations.

### 3.5 Annotation-free check on deep mutational scans
Adding entropy to conservation improved AUC in 5 of the 9 assays (mean +0.009, bootstrap over assays -0.012 to +0.031) and changed AP by -0.004 on average (-0.034 to +0.020). The check is inconclusive. The GFP assay has a single sensitive position, so its AUC is uninformative.

### 3.6 Held-out confirmatory test
[PENDING: C1 results on 24 held-out M-CSA proteins (108 residues). Report AUC and AP for conservation, entropy and conservation plus entropy, the paired difference with its interval, and the sensitivity run that drops proteins in a 30%-identity cluster with a development protein.]

## 4. Discussion

On its own, PLM masked-prediction entropy locates experimentally supported functional sites about as well as MSA conservation, and entropy is strongly related to it. Combining the two helps a little, mainly for binding-site residues and for shallower alignments, and not for catalytic residues, which conservation already finds almost perfectly. The most defensible summary is that entropy is largely a re-expression of conservation that carries a small additional signal. The gain is too small to carry weight by itself in a pipeline that already has an alignment, but it supports using entropy as an alignment-free substitute.

The precision results matter for any use of these scores. A pooled AUC of 0.85 sounds strong, but with 2.1% positives only about one in eight of a protein's top-ranked 10% of residues is a site.

## 5. Limitations

- Labels are incomplete: unannotated true sites count as negatives, and experimentally annotated sites favour well-studied proteins. Binding-site annotations from structures are themselves related to conservation.
- Conservation depends on one alignment source and setting; other databases, depths or filters could change the baseline. Alignments are deep (the shallowest has 1,188 sequences), so "shallow" is relative.
- Well-studied proteins probably have close homologues in the PLMs' training data. The familiarity strata are proxies, not a test, and a clean test (sites annotated after the training cutoff, or proteins without homologues) was not possible here.
- The development-set comparison was seen before the analysis plan was written and is exploratory. The multiple-comparison family is conservative and the strata are small.
- The classifier is a plain logistic regression or a fixed gradient-boosting model with no tuning.
- C1 is small (24 proteins, 108 residues), so even a confirmatory test has low power.
- Scores are research hints about functional sites, not statements about any variant's clinical effect, and nothing here supports pathogenic or benign labelling.

## 6. Reproducibility
All code, tables and pinned library versions are in the repository (`scripts/`, `src/`, `results/`, `data/`). Fixed seeds are used for the protein shuffle, subsampling and bootstraps. Model IDs and library versions are recorded with each scan (`results/scan_meta.json`, `results/logprobs_*/meta.json`). Scans are cached and never recomputed unless the model or scoring code changes.

## Suggested references (verify every detail before use)
- Meier et al. 2021, "Language models enable zero-shot prediction of the effects of mutations on protein function" (ESM-1v).
- Lin et al. 2023, "Evolutionary-scale prediction of atomic-level protein structure with a language model" (ESM-2).
- ESM Cambrian model release (EvolutionaryScale, 2024) and the Hugging Face model cards for `biohub/ESMC-300M-hf` and `biohub/ESMC-600M-hf`.
- Notin et al. 2023, "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design".
- Steinegger and Soding 2017, MMseqs2.
- Mirdita et al. 2022, ColabFold.
- Capra and Singh 2007, "Predicting functionally important residues from sequence conservation".
- Ribeiro et al. 2018, Mechanism and Catalytic Site Atlas (M-CSA).
- The UniProt Consortium; Jumper et al. 2021 (AlphaFold); Varadi et al. 2022 (AlphaFold Protein Structure Database).
