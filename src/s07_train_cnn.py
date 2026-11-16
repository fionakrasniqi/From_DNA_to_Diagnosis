"""
s07_train_cnn.py — 1D convolutional network on raw sequence (proposal §2.3).

Bases are integer-encoded and passed through convolutional layers with adaptive
max-pooling, as specified. Two arms, because the naive version cannot work:

  random    Windows sampled genome-wide (10 x 2,000 bp = ~0.4% of a genome).
            The gyrA QRDR is ~30 bp, so a sampled window contains the causal
            variant with probability ~0.004. This arm is expected to perform
            poorly, and that is the measurement: it quantifies why deep
            learning on whole bacterial genomes is hard when the causal
            variants are this sparse. Reporting only this arm would look like
            a failed implementation rather than a finding.

  targeted  The gyrA and parC regions, located with the conserved flanks
            validated in stage 04. This asks whether a CNN can recover the
            resistance mutations from raw sequence without being told which
            codons matter — the direct comparison against the eight engineered
            QRDR features, which reached 0.980 AUROC under MLST-grouped CV.

Both arms are evaluated with MLST-grouped folds. Stage 05 showed the k-means
grouping from §2.4 does not separate lineages, and stage 06 showed random splits
inflate composition-based models by up to 0.09 AUROC, so the strict design is
the honest comparison.

Ciprofloxacin only, per the compute deviation recorded in the README.

Window extraction is cached: the first run reads every FASTA (~10 min), later
runs reuse the cache.

Output:
    data/processed/cnn_windows_random.npy     int8 (n, n_windows, window_bp)
    data/processed/cnn_windows_targeted.npy   int8 (n, target_len)
    data/processed/cnn_genome_ids.npy
    results/tables/cnn_results.csv
    models/cnn_targeted.pt
"""

from __future__ import annotations

import gzip
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import config as C
from s04_build_mechanism_features import QRDR_SITES, _revcomp

log = logging.getLogger("s07")

BASE_CODE = np.full(256, 4, dtype=np.int8)      # 4 = unknown / N
for _i, _b in enumerate(b"ACGT"):
    BASE_CODE[_b] = _i
    BASE_CODE[_b + 32] = _i                      # lowercase

# one region per gene, anchored on the first QRDR site of that gene
TARGET_ANCHORS = {
    "gyrA": QRDR_SITES["gyrA_83"][0],
    "parC": QRDR_SITES["parC_80"][0],
}
TARGET_LEN = 2 * C.CNN_TARGET_FLANK_BP          # per gene
TOTAL_TARGET_LEN = TARGET_LEN * len(TARGET_ANCHORS)


# ── sequence loading and window extraction ────────────────────────────────────
def read_sequence(genome_id: str) -> str | None:
    path = C.GENOMES / f"{genome_id}.fna.gz"
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as fh:
            return "".join(line.strip().upper()
                           for line in fh if not line.startswith(">"))
    except (OSError, EOFError):
        return None


def encode(sequence: str) -> np.ndarray:
    return BASE_CODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def sample_random_windows(sequence: str, rng: np.random.Generator) -> np.ndarray:
    """n_windows random fixed-length windows, encoded."""
    out = np.full((C.CNN_WINDOWS_PER_GENOME, C.CNN_WINDOW_BP), 4, dtype=np.int8)
    if len(sequence) <= C.CNN_WINDOW_BP:
        out[:, :len(sequence)] = encode(sequence)
        return out
    starts = rng.integers(0, len(sequence) - C.CNN_WINDOW_BP,
                          C.CNN_WINDOWS_PER_GENOME)
    for i, start in enumerate(starts):
        out[i] = encode(sequence[start:start + C.CNN_WINDOW_BP])
    return out


def extract_targeted(sequence: str) -> np.ndarray:
    """Concatenated gyrA and parC regions, located by conserved flank.

    Both strands are searched. When a gene's flank is absent the region is left
    as unknown bases rather than dropping the genome, so the arms stay on the
    same cohort and remain comparable.
    """
    pieces = []
    for flank in TARGET_ANCHORS.values():
        region = np.full(TARGET_LEN, 4, dtype=np.int8)
        idx = sequence.find(flank)
        if idx == -1:
            rc = _revcomp(flank)
            idx = sequence.find(rc)
            if idx != -1:
                start = max(0, idx - C.CNN_TARGET_FLANK_BP)
                chunk = _revcomp(sequence[start:start + TARGET_LEN])
                region[:len(chunk)] = encode(chunk)
        else:
            start = max(0, idx - C.CNN_TARGET_FLANK_BP)
            chunk = sequence[start:start + TARGET_LEN]
            region[:len(chunk)] = encode(chunk)
        pieces.append(region)
    return np.concatenate(pieces)


def build_window_cache(genome_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(C.SEED)
    random_windows = np.full(
        (len(genome_ids), C.CNN_WINDOWS_PER_GENOME, C.CNN_WINDOW_BP), 4, np.int8)
    targeted = np.full((len(genome_ids), TOTAL_TARGET_LEN), 4, np.int8)

    started, missing = time.time(), 0
    for i, gid in enumerate(genome_ids):
        sequence = read_sequence(gid)
        if sequence is None:
            missing += 1
            continue
        random_windows[i] = sample_random_windows(sequence, rng)
        targeted[i] = extract_targeted(sequence)

        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - started)
            log.info("windows %d/%d (%.0f genomes/s)", i + 1, len(genome_ids), rate)

    log.info("window extraction complete (%d genomes unreadable)", missing)
    coverage = (targeted != 4).mean(axis=1)
    log.info("targeted region coverage: median %.1f%% of bases resolved",
             np.median(coverage) * 100)
    return random_windows, targeted


# ── model ─────────────────────────────────────────────────────────────────────
class SequenceCNN(nn.Module):
    """Integer-encoded bases -> embedding -> two conv blocks -> adaptive max pool."""

    def __init__(self, embed_dim: int = C.CNN_EMBED_DIM,
                 n_filters: int = C.CNN_FILTERS):
        super().__init__()
        self.embed = nn.Embedding(5, embed_dim)   # A, C, G, T, unknown
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, n_filters, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_filters * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).transpose(1, 2)
        return self.head(self.conv(h).squeeze(-1)).squeeze(-1)


def train_one_fold(X_train, y_train, X_test, y_test, pos_weight: float,
                   is_bagged: bool, n_epochs: int) -> dict:
    """Train the CNN on one fold.

    is_bagged: the random arm holds several windows per genome, so training
    treats windows as independent samples and prediction averages a genome's
    window logits. The targeted arm has one sequence per genome.
    """
    torch.manual_seed(C.SEED)
    model = SequenceCNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=C.CNN_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    if is_bagged:
        n, w, bp = X_train.shape
        flat_train = torch.from_numpy(X_train.reshape(-1, bp)).long()
        flat_labels = torch.from_numpy(np.repeat(y_train, w)).float()
    else:
        flat_train = torch.from_numpy(X_train).long()
        flat_labels = torch.from_numpy(y_train).float()

    n_samples = len(flat_train)
    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(n_samples)
        total = 0.0
        for start in range(0, n_samples, C.CNN_BATCH_SIZE):
            batch = order[start:start + C.CNN_BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(model(flat_train[batch]), flat_labels[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        if (epoch + 1) % 4 == 0 or epoch == 0:
            log.info("    epoch %2d/%d  loss %.4f", epoch + 1, n_epochs,
                     total / n_samples)

    # ── inference ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        if is_bagged:
            n, w, bp = X_test.shape
            flat_test = torch.from_numpy(X_test.reshape(-1, bp)).long()
            logits = []
            for start in range(0, len(flat_test), C.CNN_BATCH_SIZE):
                logits.append(model(flat_test[start:start + C.CNN_BATCH_SIZE]))
            proba = torch.sigmoid(torch.cat(logits)).numpy().reshape(n, w).mean(axis=1)
        else:
            flat_test = torch.from_numpy(X_test).long()
            logits = []
            for start in range(0, len(flat_test), C.CNN_BATCH_SIZE):
                logits.append(model(flat_test[start:start + C.CNN_BATCH_SIZE]))
            proba = torch.sigmoid(torch.cat(logits)).numpy()

    pred = (proba >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_test, proba),
        "auprc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred, zero_division=0),
        "n_test": len(y_test),
        "model": model,
    }


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    torch.set_num_threads(C.CNN_THREADS)

    fold_path = C.PROCESSED / f"folds_{C.CNN_ANTIBIOTIC}.csv"
    if not fold_path.exists():
        log.error("missing %s — run s05_build_folds.py first", fold_path)
        return 1

    fold_table = pd.read_csv(fold_path, dtype={"genome_id": str})
    genome_ids = fold_table["genome_id"].to_numpy()
    y = fold_table["label"].to_numpy()
    log.info("%s: %d genomes, %.1f%% resistant",
             C.CNN_ANTIBIOTIC, len(y), y.mean() * 100)

    # ── window cache ─────────────────────────────────────────────────────────
    random_path = C.PROCESSED / "cnn_windows_random.npy"
    targeted_path = C.PROCESSED / "cnn_windows_targeted.npy"
    ids_path = C.PROCESSED / "cnn_genome_ids.npy"

    cached = (random_path.exists() and targeted_path.exists() and ids_path.exists()
              and np.array_equal(np.load(ids_path, allow_pickle=True), genome_ids))
    if cached:
        log.info("reusing cached windows")
        X_random = np.load(random_path)
        X_targeted = np.load(targeted_path)
    else:
        log.info("extracting windows from %d genomes (~10 min)", len(genome_ids))
        X_random, X_targeted = build_window_cache(genome_ids)
        np.save(random_path, X_random)
        np.save(targeted_path, X_targeted)
        np.save(ids_path, genome_ids)

    log.info("random arm   %s  (%.0f MB)", X_random.shape, X_random.nbytes / 1e6)
    log.info("targeted arm %s  (%.0f MB)", X_targeted.shape, X_targeted.nbytes / 1e6)

    # ── cross-validated training ─────────────────────────────────────────────
    folds = fold_table["fold_mlst"].to_numpy()
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    rows = []

    for arm, X, is_bagged, n_epochs in [
            ("targeted", X_targeted, False, C.CNN_EPOCHS_TARGETED),
            ("random", X_random, True, C.CNN_EPOCHS_RANDOM)]:
        log.info("=== CNN arm: %s ===", arm)
        for fold in sorted(f for f in np.unique(folds) if f >= 0):
            test_mask = folds == fold
            train_mask = ~test_mask
            if len(np.unique(y[test_mask])) < 2:
                continue

            t0 = time.time()
            log.info("  fold %d (train %d / test %d)", fold,
                     train_mask.sum(), test_mask.sum())
            result = train_one_fold(X[train_mask], y[train_mask],
                                    X[test_mask], y[test_mask],
                                    pos_weight, is_bagged, n_epochs)
            log.info("  fold %d AUROC %.3f  (%.1f min)", fold, result["auroc"],
                     (time.time() - t0) / 60)

            rows.append({"arm": arm, "fold": int(fold),
                         "auroc": result["auroc"], "auprc": result["auprc"],
                         "f1": result["f1"], "n_test": result["n_test"]})

            if arm == "targeted" and fold == 0:
                torch.save(result["model"].state_dict(), C.MODELS / "cnn_targeted.pt")

            pd.DataFrame(rows).to_csv(C.TABLES / "cnn_results.csv", index=False)

    results = pd.DataFrame(rows)

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("CNN RESULTS  (ciprofloxacin, MLST-grouped folds)")
    print("=" * 66)
    for arm in ["targeted", "random"]:
        sub = results[results["arm"] == arm]
        if sub.empty:
            continue
        print(f"\n  {arm:9}  AUROC {sub['auroc'].mean():.3f} +/- {sub['auroc'].std():.3f}"
              f"   AUPRC {sub['auprc'].mean():.3f}   F1 {sub['f1'].mean():.3f}")
        print(f"             per fold: "
              f"{', '.join(f'{v:.3f}' for v in sub['auroc'])}")

    print("\n--- comparison against stage 06 (MLST-grouped) ---")
    print("  engineered QRDR features (8 cols)   AUROC 0.980")
    print("  k-mer composition (4,096 cols)      AUROC 0.875")
    targeted_auc = results.loc[results["arm"] == "targeted", "auroc"].mean()
    random_auc = results.loc[results["arm"] == "random", "auroc"].mean()
    print(f"  CNN targeted (raw gyrA/parC)        AUROC {targeted_auc:.3f}")
    print(f"  CNN random windows (~0.4% genome)   AUROC {random_auc:.3f}")

    print("\nINTERPRETATION")
    if targeted_auc > 0.90:
        print("  The targeted CNN recovers resistance signal from raw sequence")
        print("  without being told which codons matter, approaching the")
        print("  engineered QRDR features. Deep learning found the biology.")
    else:
        print("  The targeted CNN falls short of the engineered QRDR features,")
        print("  suggesting the point mutations are hard to learn from raw")
        print("  sequence at this sample size without positional supervision.")
    print("  The random arm quantifies the sparsity problem: causal variants")
    print("  occupy ~30 bp of a 5 Mb genome, so genome-wide window sampling")
    print("  almost never observes them.")

    print(f"\nwrote {C.TABLES / 'cnn_results.csv'}")
    print("\nnext: python src/s08_shap_validation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
