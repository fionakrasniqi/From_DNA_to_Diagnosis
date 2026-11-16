"""
s03_build_kmers.py — hexamer frequency matrix (proposal §2.2).

Decomposes each genome into overlapping substrings of length k = 6, giving the
4,096 possible hexamers described in the proposal.

Two changes from the original implementation, both necessary:

1. Counting is vectorised. The previous per-character Python loop ran at roughly
   1.5 s per megabase; over ~8,700 complete genomes of ~5 Mb that is more than a
   day of compute. Encoding bases to 2 bits and using a strided rolling index
   with np.bincount does the same work in well under a second per genome.

2. Counts are converted to within-genome relative frequencies. Raw counts scale
   with assembly length, which lets tree models split on genome size instead of
   composition, and lets MiniBatchKMeans in stage 05 cluster by length rather
   than by lineage. Both raw counts and frequencies are saved; downstream stages
   use the frequencies.

DictVectorizer is not used. It exists to map arbitrary string keys to column
indices, but the hexamer space is fixed, complete and directly indexable: every
genome contains essentially all 4,096 hexamers, so the matrix is dense and the
sparse representation only adds overhead.

Output:
    data/processed/X_kmer{k}_counts.npy   int64  (n_genomes, 4096)
    data/processed/X_kmer{k}_freq.npy     float32
    data/processed/kmer_genome_ids.npy
    data/processed/kmer_vocabulary.npy    hexamer string per column
"""

from __future__ import annotations

import gzip
import logging
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger("s03")

# 2-bit base encoding; anything else (N, IUPAC ambiguity codes) maps to 255
_BASE_CODE = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _BASE_CODE[_b] = _i
    _BASE_CODE[_b + 32] = _i          # lowercase (soft-masked assemblies)

POWERS = (4 ** np.arange(C.K - 1, -1, -1)).astype(np.int64)


def read_sequence(path) -> str:
    """Concatenate all contigs of a gzipped FASTA into one string.

    Contigs are joined without a separator, which creates at most K-1 spurious
    hexamers per contig junction. Against ~5 million real hexamers per genome
    that is negligible, and any junction containing an N is dropped anyway.
    """
    parts = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts)


def count_kmers(sequence: str, k: int = C.K) -> np.ndarray:
    """Vectorised hexamer counts over the 4**k space."""
    codes = _BASE_CODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if codes.size < k:
        return np.zeros(4 ** k, dtype=np.int64)

    is_valid = codes < 4
    safe = np.where(is_valid, codes, 0).astype(np.int64)

    windows = np.lib.stride_tricks.sliding_window_view(safe, k)
    # a window is usable only if every base in it is A/C/G/T
    usable = np.lib.stride_tricks.sliding_window_view(is_valid, k).all(axis=1)

    indices = windows @ POWERS
    return np.bincount(indices[usable], minlength=4 ** k)


def build_vocabulary(k: int = C.K) -> np.ndarray:
    """Hexamer strings in the same order as the count vector's columns."""
    return np.array(["".join(p) for p in product("ACGT", repeat=k)])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    manifest_path = C.PROCESSED / "download_manifest.csv"
    metadata_path = C.PROCESSED / "genome_metadata.csv"
    if not manifest_path.exists():
        log.error("missing %s — run s02_fetch_genomes.py first", manifest_path)
        return 1

    manifest = pd.read_csv(manifest_path, dtype={"genome_id": str})
    valid = manifest[manifest["status"].isin(["ok", "cached"])].copy()
    log.info("%d genomes with successful download", len(valid))

    # ── assembly QC (proposal §3.2) ──────────────────────────────────────────
    # BV-BRC's `genome_length` field drove *download* validation, which caught
    # the truncation bug. It cannot judge assembly *quality*: a genome record
    # of 275 kb downloads correctly against a 275 kb expected length, but is
    # not a usable E. coli assembly. QC here filters obvious problems using
    # BV-BRC's own quality metadata plus plausible-size bounds.
    if metadata_path.exists():
        md = pd.read_csv(metadata_path, dtype={"genome_id": str})
        merged = valid.merge(md, on="genome_id", how="left")
        excluded_reasons = {}

        size_bad = ((merged["genome_length"] < 4_400_000) |
                    (merged["genome_length"] > 6_000_000))
        quality_bad = merged["genome_quality"].fillna("").eq("Poor")
        completeness_bad = merged["checkm_completeness"].fillna(100) < C.MIN_CHECKM_COMPLETENESS
        contamination_bad = merged["checkm_contamination"].fillna(0) > C.MAX_CONTAMINATION

        excluded_reasons["implausible_size"] = int(size_bad.sum())
        excluded_reasons["poor_quality"] = int(quality_bad.sum())
        excluded_reasons["low_completeness"] = int(completeness_bad.sum())
        excluded_reasons["high_contamination"] = int(contamination_bad.sum())

        keep = ~(size_bad | quality_bad | completeness_bad | contamination_bad)
        excluded = merged.loc[~keep, ["genome_id", "genome_length",
                                       "genome_quality", "checkm_completeness",
                                       "checkm_contamination"]]
        excluded.to_csv(C.PROCESSED / "qc_excluded.csv", index=False)

        log.info("QC excluded %d genomes (%s)",
                 int((~keep).sum()),
                 ", ".join(f"{k}={v}" for k, v in excluded_reasons.items()))
        valid = merged.loc[keep, ["genome_id"]]
    else:
        log.warning("no genome_metadata.csv — skipping QC filter")

    log.info("%d genomes proceeding to k-mer counting", len(valid))

    if valid.empty:
        log.error("no genomes to process — check the manifest and QC results")
        return 1

    counts, kept, skipped = [], [], []
    started = time.time()

    for i, gid in enumerate(valid["genome_id"], start=1):
        path = C.GENOMES / f"{gid}.fna.gz"
        if not path.exists():
            skipped.append(gid)
            continue
        try:
            sequence = read_sequence(path)
            vector = count_kmers(sequence)
            # BV-BRC serves lowercase sequence. Any future change to the base
            # encoder that stops handling case would silently produce all-zero
            # vectors rather than an error, so check the yield explicitly.
            if vector.sum() < 0.9 * max(len(sequence) - C.K + 1, 1):
                log.warning("%s: only %d hexamers from %d bp — possible "
                            "encoding or ambiguity-code problem",
                            gid, int(vector.sum()), len(sequence))
            counts.append(vector)
            kept.append(gid)
        except (OSError, EOFError, ValueError) as exc:
            log.warning("skipping %s: %s", gid, exc)
            skipped.append(gid)

        if i % 500 == 0:
            rate = i / (time.time() - started)
            log.info("%d/%d  (%.1f genomes/s, ~%.0f min remaining)",
                     i, len(valid), rate, (len(valid) - i) / rate / 60)

    X_counts = np.vstack(counts).astype(np.int64)
    genome_ids = np.array(kept)

    # within-genome relative frequencies
    totals = X_counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    X_freq = (X_counts / totals).astype(np.float32)

    np.save(C.PROCESSED / f"X_kmer{C.K}_counts.npy", X_counts)
    np.save(C.PROCESSED / f"X_kmer{C.K}_freq.npy", X_freq)
    np.save(C.PROCESSED / "kmer_genome_ids.npy", genome_ids)
    np.save(C.PROCESSED / "kmer_vocabulary.npy", build_vocabulary())

    print("\n" + "=" * 58)
    print("K-MER MATRIX")
    print("=" * 58)
    print(f"shape             : {X_counts.shape}")
    print(f"skipped           : {len(skipped)}")
    print(f"median total bp   : {np.median(X_counts.sum(axis=1)):,.0f}")
    print(f"hexamers absent in any genome : "
          f"{int((X_counts == 0).any(axis=0).sum())} of {C.N_KMER_FEATURES}")
    print(f"elapsed           : {(time.time() - started) / 60:.1f} min")
    print("\nSANITY CHECK: median total bp should be near 5,000,000.")
    print("\nnext: python src/s04_build_mechanism_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
