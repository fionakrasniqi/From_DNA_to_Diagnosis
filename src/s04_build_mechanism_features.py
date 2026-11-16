"""
s04_build_mechanism_features.py — biologically-grounded features (proposal §2.2).

Two feature sources, both extracted for every genome in the QC-passing cohort:

  1. CARD gene presence/absence, sourced from BV-BRC's precomputed sp_gene
     annotations. Cheap (~5 minutes for the cohort), homology-based, covers
     acquired resistance genes.

  2. Quinolone-resistance-determining-region (QRDR) point mutations in gyrA and
     parC, detected by direct anchor matching against the downloaded FASTAs.
     Homology alone cannot distinguish wild-type gyrA from mutant gyrA — same
     gene at ~99% identity — so these are done separately here.

Design deviation from proposal §2.2 (local CARD RGI): RGI takes 30–60 s per
complete genome. For this cohort that is 70–140 h of compute, well beyond the
project's remaining budget. BV-BRC serves precomputed CARD annotations through
its API at minutes-level cost. Those annotations are homology-only (RGI itself
also has variant models, but they are not part of the precomputed table), so
the QRDR arm below detects the point mutations they cannot see.

Known limitation of the QRDR reader: it needs an exact match to each 15 bp
flank. A harmless synonymous difference inside a flank (E. coli K-12 carries
CGC instead of CGT at gyrA codon 91) makes the position "not found". Such
differences are lineage-specific, which is why "not found" is unevenly
distributed across the cohort. The application treats gyrA 83 and parC 80 as
required and reports "cannot determine" when either is missing.

Output:
    data/processed/X_card.npy               (n_genomes, n_card_products) int8 presence
    data/processed/card_vocabulary.npy      one product string per column
    data/processed/X_qrdr.npy               (n_genomes, 8)   int8   see QRDR_FEATURES
    data/processed/qrdr_feature_names.npy   column names
    data/processed/mechanism_genome_ids.npy row order (aligned to k-mer matrix)
"""

from __future__ import annotations

import gzip
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as C

log = logging.getLogger("s04")

# ── CARD feature extraction ───────────────────────────────────────────────────
CARD_BATCH = 50               # genomes per sp_gene request
CARD_WORKERS = 4              # BV-BRC handles a few concurrent connections fine
CARD_MIN_PREVALENCE = 0.005   # drop products in < 0.5% of genomes


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["POST"]))
    session.mount("https://", HTTPAdapter(max_retries=retry,
                                          pool_connections=16, pool_maxsize=16))
    return session


def fetch_card_batch(session: requests.Session,
                     genome_ids: list[str]) -> pd.DataFrame:
    """Pull CARD presence for one batch of genomes.

    Uses source=CARD as a server-side filter (single-token so it works) rather
    than property=Antibiotic Resistance, which the RQL layer accepts but returns
    zero rows for reasons that were not worth debugging.
    """
    query = (f"in(genome_id,({','.join(genome_ids)}))&eq(source,CARD)"
             f"&select(genome_id,product,antibiotics)&limit(20000)")
    resp = session.post(f"{C.API}/sp_gene/", data=query,
                        headers=C.RQL_HEADERS, timeout=C.REQUEST_TIMEOUT)
    resp.raise_for_status()
    if len(resp.content) < 100:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(resp.text), sep="\t",
                       dtype={"genome_id": str})


def build_card_matrix(genome_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (presence_matrix int8, vocabulary array) aligned to genome_ids."""
    session = make_session()
    batches = [genome_ids[i:i + CARD_BATCH]
               for i in range(0, len(genome_ids), CARD_BATCH)]

    frames, done, start = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=CARD_WORKERS) as pool:
        futures = {pool.submit(fetch_card_batch, session, b): b for b in batches}
        for fut in as_completed(futures):
            frames.append(fut.result())
            done += 1
            if done % 20 == 0:
                log.info("CARD %d/%d batches (%.0f min elapsed)",
                         done, len(batches), (time.time() - start) / 60)

    hits = pd.concat(frames, ignore_index=True)
    hits["product"] = hits["product"].astype(str)
    hits = hits[hits["product"] != "nan"]
    log.info("CARD rows: %d, distinct products: %d",
             len(hits), hits["product"].nunique())

    # presence pivot
    presence = (hits.assign(v=1)
                    .pivot_table(index="genome_id", columns="product",
                                 values="v", aggfunc="max", fill_value=0))
    presence = presence.reindex(genome_ids, fill_value=0).astype(np.int8)

    prevalence = presence.mean(axis=0)
    keep_cols = prevalence[prevalence >= CARD_MIN_PREVALENCE].index
    dropped = presence.shape[1] - len(keep_cols)
    log.info("CARD: %d products retained (%d dropped, prevalence < %.1f%%)",
             len(keep_cols), dropped, CARD_MIN_PREVALENCE * 100)

    return presence[keep_cols].to_numpy(), np.array(keep_cols)


# ── QRDR mutation detection ───────────────────────────────────────────────────
# Detection reads the actual codon rather than testing for anticipated mutants.
#
# A conserved 15 bp flank locates the position, then the adjacent 3 bp are
# translated. This captures every variant, including ones not seen before, and
# reports the amino acid rather than a yes/no against a guessed list.
#
# Flank choice matters. gyrA codons 83 and 87 sit 12 bp apart, so a flank on the
# wrong side of one spans the other: a genome carrying S83L would then fail to
# match the codon-87 flank and be scored "not found" at both positions. Each
# flank below is taken from the side that contains no other variable codon —
# upstream for the first position, downstream for the second.
#
# References: E. coli gyrA and parC coding sequences retrieved from BV-BRC
# feature_sequence, with WT residues verified against the protein translation
# (gyrA S83/D87, parC S80/E84).

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# name -> (flank, side, wild-type residue, resistant residues)
#   side "left"  : flank sits immediately upstream, codon is the next 3 bp
#   side "right" : flank sits immediately downstream, codon is the previous 3 bp
QRDR_SITES = {
    "gyrA_83": ("CATCCCCATGGTGAC", "left",  "S", {"L", "A", "T", "W", "V"}),
    "gyrA_87": ("ACGATCGTCCGTATG", "right", "D", {"N", "G", "Y", "A", "H", "V"}),
    "parC_80": ("CATCCGCACGGCGAT", "left",  "S", {"I", "R"}),
    "parC_84": ("GCGATGGTCCTGATG", "right", "E", {"V", "G", "K", "A"}),
}

QRDR_FEATURES = [
    "gyrA_83_R", "gyrA_83_WT",
    "gyrA_87_R", "gyrA_87_WT",
    "parC_80_R", "parC_80_WT",
    "parC_84_R", "parC_84_WT",
]


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def call_qrdr_site(sequence: str, flank: str, side: str) -> str | None:
    """Return the amino acid at a QRDR position, or None if the flank is absent.

    Both strands are searched because contigs come in arbitrary orientation.
    On the reverse strand the flank's reverse complement is located and the
    codon is taken from the opposite side, then reverse-complemented back.
    """
    # forward strand
    idx = sequence.find(flank)
    if idx != -1:
        codon = (sequence[idx + len(flank):idx + len(flank) + 3] if side == "left"
                 else sequence[idx - 3:idx])
        return CODON_TABLE.get(codon)

    # reverse strand
    rc_flank = _revcomp(flank)
    idx = sequence.find(rc_flank)
    if idx != -1:
        codon = (sequence[idx - 3:idx] if side == "left"
                 else sequence[idx + len(rc_flank):idx + len(rc_flank) + 3])
        return CODON_TABLE.get(_revcomp(codon)) if len(codon) == 3 else None

    return None


def read_sequence(path: Path) -> str:
    """Concatenate all contigs of a gzipped FASTA into one uppercase string.

    Uppercased because BV-BRC serves lowercase sequence, and the reference
    flanks above are stored uppercase.
    """
    parts = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    return "".join(parts)


def build_qrdr_matrix(genome_ids: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Detect QRDR alleles per genome.

    Returns (features int8, feature names, per-genome residue calls) — the third
    return value records the actual amino acid found at each site, which stage 08
    uses as ground truth when validating SHAP-selected hexamers.
    """
    features = np.zeros((len(genome_ids), len(QRDR_FEATURES)), dtype=np.int8)
    calls: dict[str, list] = {site: [] for site in QRDR_SITES}

    start, missing = time.time(), 0
    for i, gid in enumerate(genome_ids):
        path = C.GENOMES / f"{gid}.fna.gz"
        if not path.exists():
            missing += 1
            for site in QRDR_SITES:
                calls[site].append(None)
            continue

        try:
            sequence = read_sequence(path)
        except (OSError, EOFError):
            missing += 1
            for site in QRDR_SITES:
                calls[site].append(None)
            continue

        for col, (site, (flank, side, wt, resistant)) in enumerate(QRDR_SITES.items()):
            residue = call_qrdr_site(sequence, flank, side)
            calls[site].append(residue)
            if residue is None:
                continue
            if residue in resistant:
                features[i, col * 2] = 1
            elif residue == wt:
                features[i, col * 2 + 1] = 1

        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - start)
            log.info("QRDR %d/%d  (%.0f genomes/s)", i + 1, len(genome_ids), rate)

    log.info("QRDR: scan complete (%d genomes missing FASTAs)", missing)

    residues = pd.DataFrame(calls)
    residues.insert(0, "genome_id", genome_ids)
    return features, np.array(QRDR_FEATURES), residues


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    ids_path = C.PROCESSED / "kmer_genome_ids.npy"
    if not ids_path.exists():
        log.error("missing %s — run s03_build_kmers.py first", ids_path)
        return 1

    genome_ids = np.load(ids_path).tolist()
    log.info("%d genomes (aligned to k-mer matrix row order)", len(genome_ids))

    # ── CARD ──────────────────────────────────────────────────────────────────
    card_path = C.PROCESSED / "X_card.npy"
    if card_path.exists():
        # CARD succeeded on the previous run; don't re-pull 1.2M rows
        X_card = np.load(card_path)
        card_vocab = np.load(C.PROCESSED / "card_vocabulary.npy", allow_pickle=True)
        log.info("== stage 04a: reusing cached CARD matrix %s ==", X_card.shape)
    else:
        log.info("== stage 04a: CARD gene presence ==")
        X_card, card_vocab = build_card_matrix(genome_ids)
        np.save(card_path, X_card)
        np.save(C.PROCESSED / "card_vocabulary.npy", card_vocab)
        log.info("CARD matrix: %s", X_card.shape)

    # ── QRDR ──────────────────────────────────────────────────────────────────
    log.info("== stage 04b: QRDR point-mutation detection ==")
    X_qrdr, qrdr_names, residues = build_qrdr_matrix(genome_ids)
    np.save(C.PROCESSED / "X_qrdr.npy", X_qrdr)
    np.save(C.PROCESSED / "qrdr_feature_names.npy", qrdr_names)
    np.save(C.PROCESSED / "mechanism_genome_ids.npy", np.array(genome_ids))
    residues.to_csv(C.PROCESSED / "qrdr_residues.csv", index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("MECHANISM FEATURES")
    print("=" * 62)
    print(f"CARD  matrix : {X_card.shape}  int8 presence/absence")
    print(f"QRDR  matrix : {X_qrdr.shape}  int8 (8 derived features)")

    print("\nQRDR site coverage and observed residues:")
    for site, (_, _, wt, resistant) in QRDR_SITES.items():
        col = residues[site]
        called = col.notna().sum()
        print(f"\n  {site}  (WT={wt}, resistant={'/'.join(sorted(resistant))})")
        print(f"    site located in {called}/{len(col)} genomes ({called / len(col) * 100:.1f}%)")
        counts = col.value_counts()
        for residue, n in counts.head(6).items():
            tag = "WT" if residue == wt else ("R" if residue in resistant else "other")
            print(f"      {residue}  {n:>5}  ({n / len(col) * 100:5.1f}%)  {tag}")

    print("\nSANITY CHECK — site coverage should be well above 90%.")
    print("Low coverage means the conserved flank does not match this cohort.")

    print(f"\nCARD top 10 most common resistance-associated products:")
    top = pd.Series(X_card.sum(axis=0), index=card_vocab).nlargest(10)
    for name, count in top.items():
        print(f"  {count:>5}  {name[:70]}")

    print("\nnext: python src/s05_build_folds.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
