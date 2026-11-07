# Does PLM entropy add information about functional sites beyond conservation?

Reproduce: `python scripts/task3_cv.py` (reads `results/site_table.csv`; writes `task3_cv.csv`, `task3_diffs.csv`, `task3_auc.png`).

## Data and method

- 126 reviewed human proteins (100-350 aa, seed-0 shuffle of all qualifying entries, none hand-picked), 32,402 residues, 689 positives (UniProt active/binding-site residues with experimental evidence codes ECO:0000269 / ECO:0007744, annotations spanning <= 5 residues). Residues annotated only by similarity or prediction are excluded.
- Proteins grouped into 104 clusters with MMseqs2 at 0.3 sequence identity.
- Features per residue: ESMC-300M masked entropy (main), ESM-2 650M entropy, ESM-C 600M entropy, MSA conservation (weighted Shannon `cons`, Jensen-Shannon `jsd`; ColabFold UniRef+environmental alignments, 1,188-23,242 sequences per protein, subsampled to 2,000), AlphaFold pLDDT and burial (C-beta neighbours within 10 A).
- Grouped 5-fold cross-validation (groups = cluster), class-balanced logistic regression on standardised features, out-of-fold scores pooled into one AUC. Intervals are 95% protein-level bootstraps (1,000 resamples of proteins, same resamples for every model so differences are paired).
- Single-feature AUCs score entropy and log-ratios with the sign "low = site" and conservation, JSD, burial and pLDDT with "high = site".

## Results

Pooled AUC, all 126 proteins (95% interval):

| Score | AUC |
|---|---|
| JSD conservation | 0.854 (0.831-0.878) |
| Shannon conservation (`cons`) | 0.848 (0.825-0.872) |
| ESM-C 600M entropy | 0.845 (0.821-0.870) |
| ESMC-300M entropy | 0.843 (0.818-0.867) |
| ESM-2 650M entropy | 0.819 (0.795-0.846) |
| p_wt | 0.824 (0.798-0.851) |
| mean log-ratio | 0.823 (0.798-0.847) |
| min log-ratio | 0.791 (0.765-0.816) |
| burial (contacts) | 0.692 (0.664-0.720) |
| pLDDT | 0.545 (0.508-0.583) |

Cross-validated models:

| Features | AUC |
|---|---|
| cons | 0.847 (0.823-0.871) |
| entropy | 0.839 (0.815-0.864) |
| cons + entropy | **0.856 (0.835-0.879)** |
| cons + pLDDT + contacts | 0.844 (0.821-0.869) |
| cons + entropy + pLDDT + contacts | 0.855 (0.833-0.879) |
| cons + ESM-2 entropy | 0.850 (0.827-0.874) |
| cons + ESM-C 600M entropy | 0.856 (0.834-0.879) |
| cons + all three entropies | 0.856 (0.834-0.878) |

Paired AUC differences (protein-level bootstrap):

| Comparison | Difference | 95% interval |
|---|---|---|
| entropy minus cons (single features) | -0.008 | -0.026 to +0.010 |
| (cons + entropy) minus cons | **+0.009** | +0.003 to +0.016 |
| (cons + entropy + structure) minus (cons + structure) | +0.011 | +0.005 to +0.018 |
| (cons + ESM-2 entropy) minus cons | +0.003 | -0.001 to +0.008 |
| (cons + ESM-C 600M entropy) minus cons | +0.009 | +0.002 to +0.016 |

By alignment depth (proteins split into three equal groups; "shallow" is relative, the shallowest alignment still has 1,188 sequences):

| Depth group | Proteins / sites | (cons + entropy) minus cons |
|---|---|---|
| shallow (1,188-8,343) | 42 / 213 | +0.020 (+0.005 to +0.034) |
| middle (8,433-13,275) | 42 / 163 | +0.009 (+0.003 to +0.014) |
| deep (13,453-23,242) | 42 / 313 | +0.002 (-0.007 to +0.011) |

## Conclusion in plain language

1. **On its own, entropy is about as good as conservation, not better.** The single-feature difference is small and its interval spans zero (-0.026 to +0.010). Entropy and conservation are also strongly related (Spearman about -0.72). The data do not show that entropy beats conservation, and conservation is nominally ahead.
2. **Entropy adds a small amount on top of conservation.** Adding it raises the cross-validated AUC from 0.847 to 0.856, a gain of 0.009 whose interval excludes zero but is small (+0.003 to +0.016). This is a real but modest effect, not a large one.
3. **The gain is concentrated where alignments are shallower.** It is about +0.020 in the shallowest third and indistinguishable from zero in the deepest third. This fits the idea that the language model carries some of the information a thinner alignment lacks, but it is a post hoc split and this analysis does not prove that explanation.
4. **Structure features add nothing here.** pLDDT and burial do not improve on conservation (0.844 vs 0.847), and the entropy gain is the same with or without them.
5. **Model choice matters little.** ESM-C 600M gives the same gain as ESMC-300M; ESM-2 650M gives a smaller, non-significant one.

## Caveats

- The labels are literature-based UniProt active and binding sites. They are incomplete (unannotated true sites count as negatives) and they favour well-studied proteins, so absolute AUCs are optimistic or pessimistic in ways we cannot quantify.
- The M-CSA label covers only 16 of the 126 proteins, so it was not used for the main comparison. It is also not independent of UniProt.
- Well-studied proteins probably have close homologues in the language models' training data. We did **not** stratify by pretraining representation (no clean measure was available), so this remains an open confounder.
- Alignments come from one server and one setting (UniRef plus environmental sequences, subsampled to 2,000 sequences, one weighting threshold). Other alignment depths or filtering could change the conservation baseline.
- The depth strata and the number of comparisons are not corrected for multiple testing, and the intervals come from 126 proteins, so the stratified numbers are exploratory.
- The model is a plain logistic regression with no tuning. A stronger classifier might change the small gains in either direction.
- Scores are research hints about functional sites, not clinical statements about any variant.
