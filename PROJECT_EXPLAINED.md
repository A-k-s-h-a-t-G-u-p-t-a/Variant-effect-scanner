# Variant Effect Scanner: What We Built, How We Built It, and Who Does What

**Project title:** Zero-Shot Variant Effect Prediction and Exhaustive Mutational Scanning with Protein Language Models

**One-sentence summary:** We took a protein language model (ESM Cambrian, 300 million parameters), which has only ever read raw protein sequences, and used it without any further training to predict how harmful every possible single mutation in a protein is. We showed that our predictions reproduce published benchmark results exactly, and that the model's confidence picks out experimentally proven functional sites in enzymes, even though it was never shown any such labels.

---

## Table of contents

1. [Biology you need before reading the rest](#1-biology-you-need-before-reading-the-rest)
2. [Computing and statistics you need](#2-computing-and-statistics-you-need)
3. [The problem we are solving](#3-the-problem-we-are-solving)
4. [The core idea: masked-marginal scoring](#4-the-core-idea-masked-marginal-scoring)
5. [What we did, step by step](#5-what-we-did-step-by-step)
6. [Results summary](#6-results-summary)
7. [Limitations we must state honestly](#7-limitations-we-must-state-honestly)
8. [Project files and how to rerun everything](#8-project-files-and-how-to-rerun-everything)
9. [Team roles: four members, equal shares](#9-team-roles-four-members-equal-shares)
10. [Glossary](#10-glossary)

---

## 1. Biology you need before reading the rest

Every team member should be comfortable with this section before presenting any part of the project. It is written for someone with a computing background and no biology beyond school level.

### 1.1 From DNA to protein

- **DNA** is the long molecule that stores genetic instructions. It is written in an alphabet of four letters (A, C, G, T).
- A **gene** is a stretch of DNA that holds the recipe for one protein.
- The cell copies a gene into **messenger RNA**, and a machine called the **ribosome** reads that RNA three letters at a time. Each group of three letters (a **codon**) specifies one **amino acid**.
- The ribosome joins amino acids into a chain. That chain, once folded, is a **protein**.

This flow (DNA → RNA → protein) is called the **central dogma of molecular biology**. Our project works entirely at the protein level: we never look at DNA.

### 1.2 Amino acids: the 20-letter protein alphabet

Proteins are built from 20 standard amino acids. Each has a one-letter code, and a protein sequence is simply a string of these letters, for example `MVHLTPEEKSAVTALWGKV...`.

| Letter | Amino acid | Property that matters for us |
|---|---|---|
| A | Alanine | Small, water-avoiding (hydrophobic) |
| C | Cysteine | Can form sulfur bridges; often catalytic |
| D | Aspartic acid | Negatively charged; common in active sites |
| E | Glutamic acid | Negatively charged; similar to D |
| F | Phenylalanine | Large, ring-shaped, hydrophobic |
| G | Glycine | Smallest; very flexible; often in tight turns |
| H | Histidine | Can gain or lose a charge; very common in active sites and metal binding |
| I | Isoleucine | Hydrophobic; similar to V and L |
| K | Lysine | Positively charged |
| L | Leucine | Hydrophobic |
| M | Methionine | Almost every protein starts with M |
| N | Asparagine | Polar, uncharged |
| P | Proline | Rigid; breaks helices, so substituting it in is often damaging |
| Q | Glutamine | Polar, uncharged; similar to N |
| R | Arginine | Positively charged; often binds phosphate groups |
| S | Serine | Small, polar; catalytic in many enzymes |
| T | Threonine | Polar; similar to S |
| V | Valine | Hydrophobic |
| W | Tryptophan | Largest; rarely tolerated as a replacement |
| Y | Tyrosine | Ring-shaped, polar |

You do not need to memorise this table. You do need to know that some swaps are **conservative** (similar chemistry, for example Isoleucine to Valine, or Aspartic acid to Glutamic acid, usually harmless) and some are **drastic** (for example anything to Proline inside a helix, or removing a Histidine that holds a metal ion).

### 1.3 Protein structure in four levels

1. **Primary structure:** the sequence of letters. This is the only thing our model sees.
2. **Secondary structure:** local shapes, mainly the spiral **alpha helix** and the flat **beta sheet**.
3. **Tertiary structure:** the full three-dimensional fold of one chain.
4. **Quaternary structure:** several chains assembled together (haemoglobin, for example, is four chains).

The key fact: **sequence determines structure, and structure determines function**. A single changed letter can break the fold or destroy a site that does the protein's job.

### 1.4 Mutations and how they are written

- A **mutation** (also called a **variant** or **substitution**) is a change of one amino acid for another.
- We only study **missense mutations**: one amino acid replaced by a different one. We do not study insertions, deletions or premature stop signals.
- **Notation:** `E6V` means "at position 6, the original (wild-type) amino acid Glutamic acid (E) is replaced by Valine (V)".
- **Wild type** means the normal, unmutated, natural sequence.
- A **multi-mutant** has several changes at once, written with colons, for example `A24T:G30S`.

**Numbering trap we hit in this project:** some clinical traditions number residues after removing the starting Methionine. The sickle cell mutation is famously called "E6V" in medicine, but in the UniProt database sequence (which keeps the starting Methionine as position 1) it is **E7V**. Always check which numbering a source uses.

### 1.5 Evolution and conservation

- Proteins that do the same job in different species (for example haemoglobin in humans, horses and fish) are related by evolution and are called **homologues**.
- Over millions of years, positions where mutations would be harmful stay unchanged, because organisms carrying harmful mutations leave fewer offspring. These positions are called **conserved**.
- Positions where mutations are harmless drift freely between species and are called **variable**.
- **Key consequence for our project:** by reading hundreds of millions of natural protein sequences, a model can learn which positions are conserved and which substitutions nature tolerates, without ever being told what any protein does.

### 1.6 Functional sites

Enzymes are proteins that speed up chemical reactions. Particular residues do the essential work:

- **Active site:** the residues that directly carry out the chemical reaction (for example a Cysteine that attacks the substrate, or a Histidine that passes a proton).
- **Binding site:** residues that hold a partner molecule in place, such as the substrate, a cofactor like NAD or NADP, or a metal ion like zinc or iron.

Mutating an active-site or binding-site residue usually destroys the protein's function, so these residues are almost always conserved.

### 1.7 Haemoglobin and sickle cell disease (our worked example)

- **Haemoglobin** carries oxygen in red blood cells. It is made of alpha and beta chains. We scanned the **beta chain**, encoded by the gene *HBB* (UniProt identifier **P68871**, 147 residues including the starting Methionine).
- Each chain holds a **heme** group, an iron-containing ring that actually binds oxygen. Two Histidines hold the heme in place: the **proximal Histidine (H93 in UniProt numbering)** bonds directly to the iron, and the **distal Histidine (H64)** sits on the oxygen side.
- **Sickle cell disease** is caused by a single mutation, Glutamic acid to Valine at clinical position 6 (UniProt position 7). It does not break the protein's fold. Instead it creates a sticky surface patch that makes haemoglobin molecules clump into long fibres, which bends red blood cells into a sickle shape. This detail matters when we interpret our results.

### 1.8 Measuring mutation effects in the laboratory

- **Deep mutational scanning** is an experimental technique in which scientists create thousands of mutants of one protein, all at once, and measure how well each one works (by growth of cells, brightness, binding strength, stability and so on). The measured value is called the **fitness score**.
- **ProteinGym** is a public collection of over 200 such experiments, assembled as a benchmark so that computational predictors can be compared fairly.

### 1.9 Clinical context

- Sequencing a person's genome reveals many missense variants. For most of them, nobody knows whether they cause disease. These are called **variants of uncertain significance**.
- Testing each one in a laboratory is slow and expensive. Computational predictors help prioritise which variants deserve attention. This is the motivation for our project.
- **ClinVar** is the public database where clinical laboratories record whether variants are pathogenic (disease-causing) or benign.

### 1.10 Databases we used

- **UniProt** (specifically its reviewed section, **Swiss-Prot**): the main protein database. Each entry has a sequence plus hand-curated annotations such as active sites and binding sites, and each annotation carries an **evidence code** saying where it came from.
  - `ECO:0000269` = supported by experiments described in a published paper.
  - `ECO:0007744` = supported by an experimentally determined three-dimensional structure.
  - `ECO:0000250` and `ECO:0000255` = inferred by similarity to another protein or by prediction. **We deliberately did not count these as true sites** (explained in section 5.5).

---

## 2. Computing and statistics you need

### 2.1 Protein language models

- A **language model** learns patterns in text. A **protein language model** does the same with protein sequences, treating each amino acid as a "word".
- Our model is trained with **masked language modelling**: during training, some letters are hidden behind a special `<mask>` symbol and the model must guess them from the surrounding sequence. After seeing hundreds of millions of sequences, it gets very good at saying which amino acids fit at a given position.
- Our model: **ESM Cambrian, 300 million parameters** (Hugging Face name `biohub/ESMC-300M-hf`). It has 30 transformer layers and an internal vector size of 960. It runs on an ordinary laptop processor without a graphics card.
- **Zero-shot** means we use the model as it is, with no extra training on mutation data. We never trained anything in this project.

### 2.2 Tokens

The model does not read letters directly. A **tokenizer** converts the sequence into a list of numbers (**tokens**). It also adds a special start token `<cls>` at the beginning and an end token `<eos>` at the end. So the residue at biological position 1 sits at token index 1, not index 0. Getting this offset wrong silently ruins every result, which is why our first checkpoint verifies it.

### 2.3 Probabilities, log probabilities and log-ratios

- For a masked position, the model outputs a score for every token, which a **softmax** function converts into probabilities that sum to 1.
- We work with the **logarithm** of probabilities (**log probabilities**) because they are numerically stable and differences between them are meaningful.
- **Log-ratio** = log probability of the mutant amino acid minus log probability of the wild-type amino acid. Zero means "the model likes them equally"; strongly negative means "the model strongly prefers the original", which we interpret as damaging.

### 2.4 Entropy

**Entropy** measures how spread out a probability distribution is. If the model puts all its probability on one amino acid, entropy is 0 (completely confident). If it spreads probability evenly over all 20, entropy is at its maximum, the natural logarithm of 20, about 3.0. We measure entropy in **nats** (units based on the natural logarithm).

### 2.5 Spearman rank correlation

Compares two lists by their **order** rather than their raw values. A value of 1 means our predicted ranking of mutants matches the laboratory ranking perfectly, 0 means no relationship, and negative means reversed. It is the standard score on ProteinGym. Zero-shot models typically score between 0.3 and 0.5.

### 2.6 Area under the receiver operating characteristic curve

Usually shortened to **AUC** in papers and in our output files. It answers: "if I pick one true functional site and one non-site at random, how often does my score rank the true site higher?" A value of 0.5 is random guessing; 1.0 is perfect.

---

## 3. The problem we are solving

1. Most human missense variants have unknown effects.
2. Laboratory testing does not scale to millions of variants.
3. Protein language models have learned evolutionary constraints from raw sequences.
4. **Question:** can such a model, without any further training, (a) predict which mutations are damaging, (b) do so for every possible mutation of a protein, and (c) reveal which residues are functionally important?

We are **not** claiming state-of-the-art accuracy. We are building a working, validated, reproducible system on a laptop and reporting an honest analysis of what it does and does not capture.

---

## 4. The core idea: masked-marginal scoring

To score the mutation "position *i*, wild type *W*, mutant *M*":

1. Take the wild-type sequence.
2. Replace the residue at position *i* with the `<mask>` token.
3. Run the model once. Read its probability distribution over the 20 amino acids at position *i*.
4. Score = log P(*M*) − log P(*W*).

**Interpretation:** "Having seen the rest of the protein, how much less plausible does the model find the mutant than the original?"

**Efficiency fact:** one masked pass at position *i* gives the scores for all 20 possible amino acids at that position at once. A full scan of a protein of length *L* therefore needs *L* model passes, not *L* × 20. We additionally process 8 masked copies of the sequence per pass (a "batch"), which makes scanning faster on the processor.

The output of a full scan is an **L × 20 matrix**: one row per position, one column per amino acid. The wild-type cell in every row is exactly 0 by construction, which is a free correctness check.

---

## 5. What we did, step by step

The plan was organised as a series of **gates**: checkpoints that must pass before moving on, so that a silent bug cannot contaminate later results.

### 5.1 Environment setup

- Created a Python virtual environment in `.venv` using **Python 3.12** (the Python 3.11 installation on the laptop turned out to be broken).
- Installed the **processor-only** build of PyTorch 2.14.1 (much smaller than the graphics-card build, and we have no graphics card), Hugging Face Transformers 5.19.0, and NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, tqdm and Streamlit. Exact versions are frozen in `requirements.txt`.
- Confirmed the model `biohub/ESMC-300M-hf` exists on Hugging Face and is supported natively by the Transformers library. The first run downloads about 1.2 gigabytes of weights.
- We run in 32-bit floating point on the processor; 16-bit is a graphics-card optimisation and is slow on processors.

### 5.2 Gate 1: verifying the token offset

**Why:** biology counts from 1, Python counts from 0, and the tokenizer adds a start token. Three chances to be off by one, and an off-by-one error produces no error message, only meaningless numbers.

**How:** `scripts/verify.py` tokenizes a test sequence, finds where the first residue actually lands, and asserts that every residue sits where expected.

**Result:**
```
num_hidden_layers: 30
hidden_size:       960
mask token:        <mask> 32
first 5 tokens:    ['<cls>', 'M', 'K', 'T', 'A']
last 3 tokens:     ['R', 'Q', '<eos>']
TOKEN_OFFSET = 1
offset verified OK
```
The constant `TOKEN_OFFSET = 1` is fixed at the top of `src/score.py`, and every scoring function re-checks that the token at the computed index is the expected wild-type letter before masking it.

### 5.3 Gate 2: sanity check on known mutations (haemoglobin beta)

**Why:** before trusting any benchmark, confirm the sign is correct: known damaging mutations must score more negative than harmless ones.

**How:** `scripts/scan.py` downloads the haemoglobin beta sequence from UniProt, scans all 147 positions (about 3 minutes 40 seconds), and looks up known mutations.

**Result** (more negative means more damaging):

| Mutation (UniProt numbering) | Meaning | Score |
|---|---|---|
| H93Y | Proximal heme Histidine; causes the disorder Haemoglobin M Hyde Park | −11.94 |
| H64Y | Distal heme Histidine; causes Haemoglobin M Boston | −10.71 |
| F42S | Heme pocket; causes the unstable Haemoglobin Hammersmith | −9.25 |
| E7V | Sickle cell (clinical name E6V) | −4.55 |
| K9R | Conservative swap on the surface | −0.27 |
| E8D | Conservative swap on the surface | −4.23 |
| V2I | Conservative swap near the start | −4.44 |
| (median of all 2,793 substitutions) | | −6.90 |

**Interpretation:**
- The sign is correct: mutations that destroy heme binding are the most negative, and the most conservative swap is near zero. **Gate 2 passed.**
- Sickle cell scores only moderately. This is expected: the mutation does not break the fold; it causes molecules to stick together, a mechanism a single-sequence model cannot see.
- The model is unusually confident about the whole haemoglobin sequence (the median substitution scores −6.9, and even conservative swaps score around −4). Haemoglobin is one of the most studied and most sequenced proteins, so the model has seen very many close relatives. Scores for famous human proteins are therefore inflated, and we should say so.

### 5.4 Gate 3: the mutational heatmap

**How:** `src/plot.py` draws the L × 20 matrix as a heatmap, 60 positions at a time (a 147-wide heatmap is unreadable). Columns are positions labelled with their wild-type letter, rows are the 20 possible mutant amino acids. Red means damaging, blue means tolerated, white means neutral.

**What we saw (positions 61 to 120):**
- Solid dark red columns at **H64** and **H93**, the two heme-binding Histidines: the model tolerates nothing there.
- The **Proline row** is dark across most positions, because Proline breaks helices.
- The **Tryptophan row** is dark, because Tryptophan is rarely tolerated as a replacement.
- Paler columns at Glycines and surface positions, where many substitutions are tolerated.

This is exactly the structure expected of a correct scan. **Gate 3 passed.** These images are the first slide of the presentation.

Output files: `results/heatmap_P68871_1-60.png`, `results/heatmap_P68871_61-120.png`, `results/heatmap_P68871_121-147.png`, plus the saved matrix `results/scan_P68871.npy`.

### 5.5 Gate 4: benchmark validation on ProteinGym

**How:**
1. Downloaded the ProteinGym version 1.3 substitution data (43 megabytes) and its reference file, which lists each experiment's target sequence.
2. Chose 9 experiments on proteins shorter than 400 residues, covering different measurement types (cell growth fitness, enzyme activity, protein expression level, folding stability) and different organisms (human, bacterium, jellyfish).
3. For each protein: scanned it **once**, saved the result, then looked up every measured mutant in the matrix. We never re-run the model per mutant.
4. For multi-mutants, the score is the **sum** of the individual single-mutant scores (an additive assumption), which is how ProteinGym's official numbers are computed.
5. Computed Spearman correlation against the laboratory fitness scores.
6. Downloaded ProteinGym's **published** per-experiment Spearman table and compared our numbers to the published numbers for the same model (ESM Cambrian 300 million), and to ESM-2 (650 million parameters) and ESM-1v.

**Total compute time:** about 50 minutes on the laptop.

**Results:**

| Protein (experiment) | Measurement | Length | Mutants | **Ours** | Published ESM Cambrian 300M | Published ESM-2 650M | Published ESM-1v |
|---|---|---|---|---|---|---|---|
| SUMO1, human (Weile 2017) | Fitness | 101 | 1,700 | **0.385** | 0.386 | 0.509 | 0.467 |
| Calmodulin 1, human (Weile 2017) | Fitness | 149 | 1,813 | **0.269** | 0.270 | 0.212 | 0.235 |
| UBC9, human (Weile 2017) | Fitness | 159 | 2,563 | **0.499** | 0.499 | 0.473 | 0.477 |
| Dihydrofolate reductase, *E. coli* (Thompson 2019) | Fitness | 159 | 2,363 | **0.503** | 0.501 | 0.480 | 0.416 |
| HRas, human (Bandaru 2017) | Activity | 189 | 3,134 | **0.424** | 0.422 | 0.498 | 0.360 |
| Green fluorescent protein, jellyfish (Sarkisyan 2016) | Activity (brightness) | 238 | 51,714 | **0.124** | 0.118 | 0.108 | 0.098 |
| Thiopurine methyltransferase, human (Matreyek 2018) | Expression | 245 | 3,648 | **0.561** | 0.561 | 0.539 | 0.517 |
| Beta-lactamase, *E. coli* (Stiffler 2015) | Fitness | 286 | 4,996 | **0.706** | 0.706 | 0.731 | 0.668 |
| CBX4 domain, human (Tsuboyama 2023) | Stability | 50 | 2,282 | **0.625** | 0.627 | 0.682 | 0.650 |
| **Mean** | | | | **0.455** | **0.454** | **0.470** | **0.432** |

**Interpretation:**
- **Every protein is within 0.006 of the officially published number for the same model.** This is the strongest possible evidence that our implementation (offset, masking, sign, indexing) is correct. **Gate 4 passed** (threshold was a mean above 0.25).
- Our 300-million-parameter model nearly matches ESM-2 at 650 million parameters (0.455 against 0.470) and beats ESM-1v (0.432).
- Performance varies enormously by protein, from 0.12 to 0.71.

**A debugging lesson worth presenting:** green fluorescent protein first gave 0.056 against a published 0.118. The cause: our first version used only the 1,084 single mutants, while ProteinGym evaluates all 51,714 variants (most are multi-mutants). Scoring multi-mutants additively gave 0.124, matching the published value. Most single mutants of this protein barely change brightness, so ranking them is very hard. Our results file now reports both numbers (`spearman` for all variants and `spearman_singles` for single mutants only).

Output file: `results/proteingym_spearman.csv`.

### 5.6 Phase 3: the finding: does model confidence reveal functional sites?

**The claim being tested:** the model never saw a single biological annotation, only raw sequences. If the positions where it is most confident (lowest entropy) coincide with sites that biologists proved experimentally to be functional, then the model learned something about protein function from evolution alone.

**How we chose the proteins (to avoid bias and circular reasoning):**
1. Queried UniProt for reviewed human proteins of length 100 to 350 that have at least one **experimentally supported** active site. 149 proteins matched.
2. Shuffled them with a fixed random seed (seed 0) and took the first 25 that qualified, so nobody hand-picked favourable examples.
3. **Positive labels:** residues inside active-site or binding-site annotations whose evidence is experimental (`ECO:0000269` or `ECO:0007744`).
4. Skipped annotations spanning more than 5 residues (long sequence motifs contain many residues that do not actually touch the partner molecule).
5. **Excluded entirely** (neither positive nor negative): residues annotated only by similarity or prediction. Similarity-based annotations are themselves derived from sequence conservation, which is exactly what the model learns, so counting them would make the test circular. Counting them as negatives would also be wrong, since they may well be real sites.

This gave 25 proteins, 6,248 residues and 126 experimental site residues.

**How we scored:** for each residue, the entropy of the model's distribution over the 20 amino acids (renormalised to sum to 1) when that residue is masked. Low entropy ranks as "more likely functional". We also tried the mean log-ratio over all 19 substitutions as an alternative score.

**Compute time:** about 3 hours on the laptop.

**Results:**

| Measure | Value |
|---|---|
| Pooled area under the curve (all 6,248 residues together), entropy | **0.844** |
| Pooled area under the curve, mean log-ratio | 0.821 |
| Area under the curve per protein, mean / median | 0.906 / 0.945 |
| Proteins scoring above 0.75 | 23 of 25 |
| Mean entropy at functional sites vs all other residues | 0.263 vs 1.139 nats |

**The control experiment:** functional sites are enriched for particular amino acids (Aspartic acid, Histidine, Arginine, Lysine, Glutamic acid and Cysteine dominate our 126 sites). Perhaps the model is simply confident about those amino acids everywhere, and is not identifying sites at all. We tested this two ways (`scripts/entropy_control.py`):

| Control | Area under the curve |
|---|---|
| Amino acid type alone (rate of being a site per amino acid, learned on held-out proteins with 5-fold cross-validation grouped by protein) | 0.707 |
| Entropy, but comparing each site **only against residues of the same amino acid type** (for example site Histidines against non-site Histidines) | **0.843** |

The second control removes the amino-acid-type effect completely and the result does not drop. **The model's confidence is not a proxy for residue type: it genuinely singles out the functional positions.**

**Figure:** `results/entropy_P30043.png` plots entropy along biliverdin reductase B (the protein with the most sites, 19), with the experimental sites shaded. Most shaded bands sit on dips in the entropy line.

Output files: `results/entropy_sites_per_protein.csv`, `results/entropy_sites_residues.csv`, `results/entropy_P30043.png`.

### 5.7 The interactive web application

**Why:** a live scan takes minutes, so for demonstrations we built a browser application that loads saved scans instantly and lets anyone explore the results without touching code.

**How it is built:** `app.py` uses **Streamlit** (a Python library that turns a script into a web page) and **Plotly** (interactive charts with hover details). It reuses the project's own functions from `src/`, so the numbers in the application are exactly the numbers in this document.

**What it shows:**
- **Sidebar:** choose one of 35 pre-scanned proteins (haemoglobin beta, the 9 ProteinGym proteins, the 25 enzymes) or paste any new sequence. A pasted sequence is scanned with a progress bar (about 1.5 seconds per residue) and saved, so it only runs once. A permanent notice states that this is a research tool, not a diagnosis.
- **Heatmap tab:** the full L × 20 matrix, 60 positions at a time. Hovering a cell shows the mutation, its score and the percentage of the protein's substitutions it is more damaging than.
- **Mutation lookup tab:** type a mutation such as `H93Y` to see its score, its within-protein percentile, the model's uncertainty at that position, and a bar chart of all 20 amino acids at that position. If the wild-type letter is wrong it suggests nearby positions, which catches the clinical-versus-UniProt numbering trap (typing `E6V` for haemoglobin suggests `E7V`).
- **Most damaging tab:** the 20 most damaging and 20 most tolerated substitutions, and a download of every substitution as a spreadsheet file.
- **Position importance tab:** entropy along the protein; for the 25 enzymes, the experimentally proven functional sites are marked, with that protein's area under the curve.
- **Validation tab:** for the 9 ProteinGym proteins, a scatter plot of predicted score against laboratory-measured fitness with the Spearman correlation beside the published value; for every protein, the benchmark summary table.

**Deliberate design choice:** the application never labels a mutation "pathogenic" or "benign". Our scores are not calibrated against clinical labels, so it reports scores and within-protein percentiles only.

**Testing:** every one of the 35 proteins was loaded through Streamlit's built-in test runner with no errors, and the lookup checks (valid mutation, wrong wild type, wrong numbering, malformed input) were exercised.

---

## 6. Results summary

| Checkpoint | Outcome |
|---|---|
| Gate 1: token offset | Verified, offset = 1 |
| Gate 2: sign of known mutations | Correct; heme Histidine mutations about −11, conservative swap −0.27 |
| Gate 3: heatmap | Expected structure visible (conserved columns, Proline row) |
| Gate 4: ProteinGym | Mean Spearman 0.455 against published 0.454 for the same model |
| Phase 3: functional sites | Area under the curve 0.844, holding at 0.843 after controlling for amino acid type |

---

## 7. Limitations we must state honestly

1. **Not state of the art.** Larger models and methods that use multiple sequence alignments or protein structure score higher on ProteinGym. Our goal was a validated, reproducible, laptop-scale system.
2. **Not a diagnostic tool.** A low score is a computational hint, not a clinical classification.
3. **Single mutations only.** Multi-mutants are scored by adding single effects, which ignores interactions between mutations (called **epistasis**).
4. **Conservation is not the same as function.** The model may be tracking evolutionary conservation; positions can be conserved for structural reasons (folding) rather than for activity. Our amino-acid control rules out one confound but not this one.
5. **Mechanisms the model cannot see.** Sickle cell disease acts through molecules sticking together, so it scores only moderately.
6. **Memorisation of famous proteins.** Well-studied human proteins such as haemoglobin have very many relatives in the training data, inflating the model's confidence.
7. **Small numbers of sites.** Nine of the 25 proteins have only one or two experimental sites, so their individual perfect scores are noisy. The pooled value of 0.844 is the number to quote.
8. **Weak cases exist.** Cathepsin L scored only 0.558 and Angiogenin 0.703.

**Future work:** train sparse autoencoders on the model's internal vectors to turn its implicit knowledge into named biological features, turning a score into a mechanism; evaluate on ClinVar disease variants; calibrate scores against clinical labels so the application could report a meaningful risk category.

---

## 8. Project files and how to rerun everything

```
major project/
├── src/
│   ├── model.py              # loads the model and tokenizer
│   ├── score.py              # masking, scoring, batched scan, entropy (TOKEN_OFFSET lives here)
│   ├── plot.py               # heatmap drawing
│   └── evaluate.py           # ProteinGym evaluation and saved-scan caching
├── scripts/
│   ├── verify.py             # Gate 1
│   ├── scan.py               # Gates 2 and 3 (haemoglobin beta by default)
│   ├── proteingym.py         # Gate 4
│   ├── entropy_sites.py      # Phase 3 finding
│   └── entropy_control.py    # Phase 3 amino-acid control
├── data/                     # downloaded sequences, ProteinGym files, UniProt annotations
├── results/                  # saved scans (logprobs/), tables, figures
├── app.py                    # interactive web application (Streamlit)
├── requirements.txt
└── PROJECT_EXPLAINED.md      # this document
```

**To rerun** (from the project folder, in PowerShell):
```powershell
.\.venv\Scripts\python.exe scripts\verify.py            # seconds
.\.venv\Scripts\python.exe scripts\scan.py              # about 4 minutes; add a UniProt identifier to scan another protein
.\.venv\Scripts\python.exe scripts\proteingym.py        # about 50 minutes the first time, seconds afterwards
.\.venv\Scripts\python.exe scripts\entropy_sites.py     # about 3 hours the first time, seconds afterwards
.\.venv\Scripts\python.exe scripts\entropy_control.py   # seconds
.\.venv\Scripts\python.exe -m streamlit run app.py      # opens the web application at http://localhost:8501
```
Every scan is saved in `results/logprobs/`. Scripts reuse saved scans automatically, so only the first run is slow. Delete a saved file to force a fresh scan.

---

## 9. Team roles: four members, equal shares

Each member **owns** one part of the pipeline: they must understand their code completely, be able to rerun it, answer questions about it, write their section of the report and present their part. Each role also has one piece of remaining work, so the remaining effort is shared equally. Everyone should read sections 1 to 4 of this document.

### Member 1: Model and Scoring Engineer

**Owns:** environment, model loading, the token offset check, and the scoring algorithm (sections 2.1 to 2.3, 4, 5.1 and 5.2).

**Done work to understand and explain:**
- Why we use the processor-only PyTorch build and 32-bit floating point.
- How the tokenizer adds start and end tokens, and why the offset assertion (Gate 1) is the most important safety check in the project.
- The masked-marginal formula, why one pass gives all 20 scores at a position, and how batching 8 masked copies speeds up scanning.
- The functions in `src/model.py` and `src/score.py`.

**Remaining work:**
- Own the back end of the web application (`app.py`, section 5.7): understand how it caches the model and scans (`st.cache_resource`, `st.cache_data`), how live scanning of a pasted sequence works, and how the within-protein percentile is computed. Be ready to demonstrate a live scan of a short sequence.
- Measure and record scan speed (seconds per residue) for the report.

**Report section:** "Approach" and "System".

### Member 2: Mutational Scanning and Visualisation Lead

**Owns:** the haemoglobin case study, sanity checks and all figures (sections 1.7, 5.3 and 5.4).

**Done work to understand and explain:**
- The haemoglobin biology: heme, proximal and distal Histidines, sickle cell mechanism, and the clinical versus UniProt numbering difference.
- The Gate 2 table and why sickle cell scores only moderately.
- How to read the heatmap: conserved columns, the Proline and Tryptophan rows, why wild-type cells are exactly zero.
- `scripts/scan.py` and `src/plot.py`.

**Remaining work:**
- Own the front end of the web application (`app.py`, section 5.7): the five tabs, the charts and their colour choices. Lead the live demonstration in the presentation (suggested path: haemoglobin heatmap, look up `H93Y`, then `E6V` to show the numbering warning, then an enzyme's position importance tab).
- Produce final, polished figures for the slides (consistent fonts, titles, colour scale).

**Report section:** "Case study: haemoglobin beta" and all figures.

### Member 3: Benchmark Validation Lead

**Owns:** the ProteinGym evaluation (sections 1.8, 2.5 and 5.5).

**Done work to understand and explain:**
- What deep mutational scanning is and what ProteinGym contains.
- Why we scan once and look up mutants in the matrix, and how multi-mutants are scored additively.
- Spearman correlation and why matching the published numbers proves correctness.
- The green fluorescent protein debugging story (single mutants versus all variants).
- `src/evaluate.py` and `scripts/proteingym.py`.

**Remaining work:**
- Optional extension: ClinVar evaluation. Download `variant_summary.txt.gz` from the National Center for Biotechnology Information, filter to one well-covered gene (for example *TP53* or *BRCA1*), keep only clear Pathogenic and Benign labels, score them and compute the area under the curve (expected range 0.70 to 0.85). If time is short, instead write a short analysis of why performance varies so much across the nine proteins (measurement type, protein length, organism).
- Prepare the comparison bar chart (ours against the published baselines) for the slides.

**Report section:** "Validation".

### Member 4: Functional Site Analysis and Report Lead

**Owns:** the Phase 3 finding and the overall report (sections 1.5, 1.6, 1.10, 2.4, 2.6, 5.6 and 7).

**Done work to understand and explain:**
- Active sites, binding sites, conservation and UniProt evidence codes.
- Why only experimental annotations were used, and why similarity-based ones were excluded rather than counted as negatives.
- Entropy, the area under the curve, pooled versus per-protein results.
- The amino-acid-type control and why the within-type value of 0.843 is the key evidence.
- `scripts/entropy_sites.py` and `scripts/entropy_control.py`.

**Remaining work:**
- Assemble the final report from all four sections, following the structure: Problem, Approach, System, Validation, Finding, Limitations, Future work. State clearly in the abstract that we do not claim state-of-the-art accuracy.
- Assemble the slide deck from everyone's figures and keep a single consistent story.

**Report section:** "Finding", "Limitations", "Future work", plus the abstract and introduction.

### Shared responsibilities

- Every member presents their own part in the final presentation.
- Every member reviews one other member's report section (suggested pairs: 1 reviews 3, 3 reviews 1, 2 reviews 4, 4 reviews 2).
- Every member should be able to run `scripts/verify.py` and `scripts/scan.py` on their own laptop.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Amino acid | One of 20 building blocks of proteins, written as a single letter |
| Active site | Residues that directly perform an enzyme's chemical reaction |
| Area under the curve (AUC) | Probability that a random true site is ranked above a random non-site; 0.5 is chance |
| Binding site | Residues that hold a partner molecule, cofactor or metal ion |
| Conserved position | A position that stays unchanged across related species |
| Deep mutational scanning | Laboratory measurement of thousands of mutants at once |
| Entropy | Spread of the model's probability over the 20 amino acids; low means confident |
| Epistasis | Mutations interacting so that their combined effect is not the sum of individual effects |
| Evidence code | UniProt label saying whether an annotation is experimental or inferred |
| Heme | Iron-containing ring in haemoglobin that binds oxygen |
| Log-ratio | Log probability of mutant minus log probability of wild type; negative means damaging |
| Masked language modelling | Training by hiding tokens and asking the model to guess them |
| Missense mutation | Replacement of one amino acid by another |
| Multi-mutant | A variant with several substitutions at once |
| ProteinGym | Public benchmark of over 200 deep mutational scanning experiments |
| Residue | One amino acid within a protein chain |
| Spearman correlation | Agreement between two rankings; 1 is perfect, 0 is none |
| Token | The number the model uses for one symbol (an amino acid or a special marker) |
| UniProt / Swiss-Prot | The main protein database / its manually reviewed section |
| Variant of uncertain significance | A human genetic variant whose clinical effect is unknown |
| Wild type | The normal, unmutated sequence |
| Zero-shot | Using a model with no task-specific training |
