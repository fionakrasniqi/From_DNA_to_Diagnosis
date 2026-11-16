# From DNA to Diagnosis

Predicting ciprofloxacin and other antibiotic resistance in *Escherichia coli* from whole-genome
sequence, and testing whether the models learn resistance or only recognise bacterial lineage.

BSc Computer Science thesis, UNI – Universum International College (2026).
Author: Fione Krasniqi. Mentor: Prof. Ass. Dr. Besart Prebreza.

## What this repository contains

```
amr/
├── app.py                 Streamlit application (Albanian / English, PDF report)
├── .streamlit/config.toml application theme
├── requirements.txt       exact library versions
├── src/                   the pipeline, one script per stage (run in order)
├── tools/                 helper scripts for checking results (not pipeline stages)
├── tests/test_core.py     39 automatic tests
├── models/                fitted models used by the application
├── results/figures/       all figures of the thesis
└── results/tables/        all result tables
```

The genome data (about 11 GB) is not included. It is downloaded automatically by stages s01, s02
and s11. The list of the 7,554 external NCBI genomes is included in
`data/external/ncbi_cohort_ciprofloxacin.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
cd src
python s01_fetch_phenotypes.py          # ~3 min    laboratory phenotypes from BV-BRC
python s02_fetch_genomes.py             # 2-3 h     genome download with truncation guards
python s03_build_kmers.py               # ~25 min   quality control + hexamer matrix
python s04_build_mechanism_features.py  # ~15 min   CARD + QRDR features
python s05_build_folds.py               # ~1 min    four validation designs
python s06_train_classical.py           # 4-6 h     192 configurations
python s07_train_cnn.py                 # ~2.5 h    convolutional network, two arms
python s08_shap_validation.py           # ~3 min    SHAP attribution and checks
python s09_make_figures.py              # <1 min    charts and tables
python s10_build_app.py                 # ~1 min    models for the application
python s11_external_validation.py       # ~45 min   external test on NCBI genomes
python s12_draw_diagrams.py             # <1 min    explanatory diagrams
python s13_tune_kmer.py                 # ~35 min   nested-CV tuning check for k-mer models
cd ..
streamlit run app.py
```

Tests, from the project root: `python -m pytest tests -q`

Every stage is resumable and reads its settings from `src/config.py`. Random seed: 42.

## Cohort (after quality control)

Laboratory-measured phenotypes only. BV-BRC's own computational predictions and "intermediate"
results are excluded.

| Antibiotic | Genomes | Resistant | Susceptible | % Resistant |
|---|---|---|---|---|
| Ciprofloxacin | 7,266 | 1,837 | 5,429 | 25.3 |
| Ampicillin | 6,435 | 3,616 | 2,819 | 56.2 |
| Tetracycline | 845 | 504 | 341 | 59.6 |
| Gentamicin | 7,284 | 856 | 6,428 | 11.8 |

8,710 genomes downloaded, 8,709 passed the length check, 65 excluded by quality control,
8,644 analysed. 743 MLST sequence types.

## Main results

### 1. The validation design changes the answer

Ciprofloxacin, AUROC under lineage-grouped (MLST) validation. Results are shown for all three
models, so nothing depends on picking the best model on the test folds.

| Features | Logistic regression | Random Forest | XGBoost |
|---|---|---|---|
| K-mer (4,096) | 0.875 | 0.771 | 0.834 |
| CARD (134) | 0.894 | 0.898 | 0.907 |
| QRDR (8) | 0.980 | 0.980 | 0.980 |
| Combined | 0.979 | 0.981 | 0.984 |

Under random splits every feature set scores above 0.93. Once lineages are kept out of the test
fold, k-mer models lose 0.09 to 0.17 AUROC while QRDR loses 0.005. The k-means grouping specified
in the proposal does not separate lineages (fold resistance 24/29/25/23/26%, like random splitting).

A nested cross-validation check (`s13`) shows the drop is not caused by missing tuning: tuned
k-mer models still lose 0.078 (logistic regression) and 0.139 (XGBoost).

### 2. The best features match each drug's mechanism

QRDR features win for ciprofloxacin (point mutations); CARD features win for ampicillin,
tetracycline and gentamicin (acquired genes). This holds for all three models.

### 3. Why k-mers fail on new lineages

Of the 20 most influential canonical hexamers, 15 are palindromic restriction-enzyme sites
(base rate 3.1%, binomial p = 2.8 × 10⁻¹⁹). Lineage explains far more of their variation than
QRDR genotype (epsilon² 0.70 vs 0.04).

### 4. Input matters more than architecture

The same CNN reaches 0.982 ± 0.006 on the gyrA/parC regions and 0.496 ± 0.013 on random genome
segments (± = standard deviation across five folds; epochs were 20 and 8, adjusted to the number
of samples per epoch).

### 5. External validation (NCBI, ciprofloxacin)

7,554 genomes not present in BV-BRC; 7,383 scored (162 "cannot determine", 9 failed QC).

| Model | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|
| QRDR + XGBoost | 0.903 (0.892–0.915) | 0.743 | 0.979 |
| Simple rule (gyrA 83 or parC 80 mutated) | 0.871 | 0.798 | 0.944 |
| K-mer + logistic regression | 0.851 (0.836–0.864) | 0.746 | 0.813 |

At the same false-positive rate the QRDR model and the simple rule find practically the same share
of resistant genomes (79.9% vs 79.8%): the model reproduces known biology rather than surpassing it.
The k-mer model stays below even the simple rule. On USA genomes (72% of the set) the QRDR model
reaches 0.969.

## The application

`streamlit run app.py` opens an application in Albanian and English. It reads the four QRDR
positions from a FASTA file, predicts ciprofloxacin resistance, shows the codons it read, and can
download a plain-language PDF report. If gyrA 83 or parC 80 cannot be read, it reports
"cannot determine" instead of guessing.

It is a research and demonstration tool, not a diagnostic test.

## The defect this rebuild corrects

The BV-BRC API returns 25 records by default. The first version requested genome fragments without
a limit, so every genome was truncated to about a fifth of its length. Two guards now prevent this:
an explicit limit on every request, and a check that each download reaches 95% of the length
BV-BRC reports.

## Deviations from the approved proposal

| Proposal | Implementation | Reason |
|---|---|---|
| Exclude sequences under 100 kb | 95% length check + quality control | The 100 kb rule let truncated genomes through |
| Local CARD RGI | BV-BRC precomputed CARD annotations + QRDR detector | RGI needs 70–140 h; the annotations cannot see gyrA mutations |
| DictVectorizer, sparse matrix | Dense array | Nearly every genome contains all 4,096 hexamers |
| CNN on whole genomes | gyrA/parC regions + random segments | Whole genomes are not trainable on a CPU |
| k-means grouping only | k-means + MLST grouping | k-means does not separate lineages |
| BLAST the top 20 hexamers | Lineage and genotype variance attribution | A 6-letter word occurs over 1,000 times per genome |
| NCBI external validation | Done for ciprofloxacin (7,383 genomes) + country hold-out | CARD features cannot be computed outside BV-BRC |

## Limitations

- 43% of the ciprofloxacin cohort is Norwegian; 72% of the external set comes from the USA.
- Tetracycline has only 845 genomes.
- Laboratory labels come from many laboratories with different cut-offs; NCBI labels are not
  independently verified.
- The QRDR reader needs an exact match to a 15-base reference sequence, and the deployed model
  reads only four positions (no efflux or plasmid-mediated resistance).
- The CNN was tested for ciprofloxacin only, with one configuration.
