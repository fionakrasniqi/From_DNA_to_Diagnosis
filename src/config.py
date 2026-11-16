"""
config.py — single source of truth for paths, constants and thresholds.

Every script imports from here. Nothing else hard-codes a path, a seed or a
threshold, so the whole pipeline can be re-pointed or re-parameterised in one
place. This also satisfies the reproducibility commitment in proposal §3.2.
"""

from __future__ import annotations

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
PHENOTYPES = DATA / "phenotypes"
PROCESSED = DATA / "processed"
GENOMES = DATA / "genomes"          # gzipped FASTAs, one per genome

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = ROOT / "models"

for _d in (PHENOTYPES, PROCESSED, GENOMES, FIGURES, TABLES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# ── reproducibility (proposal §3.2) ───────────────────────────────────────────
SEED = 42

# ── study scope (proposal §2.1) ───────────────────────────────────────────────
TAXON_ID = 562                       # Escherichia coli

PRIMARY_ANTIBIOTIC = "ciprofloxacin"
ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "tetracycline", "gentamicin"]

# Only lab-measured phenotypes. BV-BRC also serves "Computational Method" rows,
# which are its own model's predictions — training on those would mean learning
# to imitate another classifier rather than learning from measured resistance.
REQUIRED_EVIDENCE = "laboratory method"
VALID_PHENOTYPES = ["Resistant", "Susceptible"]
LABEL_MAP = {"Susceptible": 0, "Resistant": 1}

# ── BV-BRC API ────────────────────────────────────────────────────────────────
API = "https://www.bv-brc.org/api"
RQL_HEADERS = {
    "Content-Type": "application/rqlquery+x-www-form-urlencoded",
    "Accept": "text/tsv",            # omitting Accept returns HTTP 400
}
PAGE_SIZE = 5000
METADATA_BATCH = 400
REQUEST_TIMEOUT = 240

# The bug that invalidated the first attempt: the data API returns 25 records by
# default, so contig requests silently truncated every assembly to its first 25
# contigs (median 1.16 Mb against a true ~5.07 Mb genome). Always send limit().
CONTIG_LIMIT = 25_000

# ── quality control ───────────────────────────────────────────────────────────
# Supersedes the ">= 100 kb" rule in proposal §3.2, which was too permissive to
# detect truncation. Each download is now checked against the genome_length that
# BV-BRC reports for that assembly. Logged in the manifest per §3.2.
MIN_COMPLETENESS = 0.95              # observed bp / reported genome_length
MAX_CONTAMINATION = 5.0              # CheckM contamination %, upper bound
MIN_CHECKM_COMPLETENESS = 90.0       # CheckM completeness %, lower bound
# Genomes BV-BRC marks genome_quality == "Poor" are always excluded (stage 03).

# ── features (proposal §2.2) ──────────────────────────────────────────────────
K = 6                                # 4**6 = 4096 hexamers
N_KMER_FEATURES = 4 ** K

# Counts are always converted to within-genome relative frequencies before
# modelling (stage 03). Raw counts scale with assembly length, which would let
# tree models split on genome size rather than composition.

# ── cross-validation (proposal §2.4) ──────────────────────────────────────────
N_SPLITS = 5
N_CLUSTERS = 5                       # MiniBatchKMeans groups for GroupKFold

# ── CNN (proposal §2.3) ───────────────────────────────────────────────────────
# Raw 5 Mb sequences are too large to convolve over directly, so the CNN reads
# fixed-length windows of real sequence. Two arms, for a reason worth stating:
#
#   random    windows sampled genome-wide. At CPU-feasible settings this covers
#             ~0.4% of a genome, and the gyrA QRDR is ~30 bp, so a sampled
#             window almost never contains the causal variant. Expected to
#             perform poorly — that outcome is the measurement, quantifying why
#             whole-genome deep learning is hard when causal variants are this
#             sparse.
#
#   targeted  the gyrA and parC regions, located via the conserved flanks
#             validated in stage 04. Asks whether a CNN can recover the
#             resistance mutations from raw sequence without being told which
#             codons matter — directly comparable to the engineered QRDR
#             features.
CNN_WINDOW_BP = 2_000
CNN_WINDOWS_PER_GENOME = 10
CNN_TARGET_FLANK_BP = 1_500      # extracted either side of each QRDR anchor
CNN_ANTIBIOTIC = PRIMARY_ANTIBIOTIC   # CNN arm is ciprofloxacin only
# Epochs differ by arm. The random arm holds CNN_WINDOWS_PER_GENOME samples per
# genome, so one epoch is ~10x more gradient steps than the targeted arm's, and
# it needs proportionally fewer passes. Batch 64 rather than 128: the targeted
# arm has one sequence per genome, and at batch 128 it was under-trained
# (a planted-variant benchmark reached only 0.69 AUROC, versus 1.00 with more
# steps on the same architecture).
CNN_EPOCHS_TARGETED = 20
CNN_EPOCHS_RANDOM = 8
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3
CNN_EMBED_DIM = 8
CNN_FILTERS = 32
CNN_THREADS = 4                  # torch defaults to 1 on many installs

# QRDR positions and their flanks are defined in s04_build_mechanism_features.py
# (QRDR_SITES), next to the code that reads them.
