"""
tests/test_core.py — automatic checks of the pipeline's core logic.

Run from the project root:
    python -m pytest tests -q

The first group uses small synthetic inputs and runs anywhere. The last test
reads the real fold files and is skipped if they are not present.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as C                                                # noqa: E402
import s03_build_kmers as S3                                      # noqa: E402
import s04_build_mechanism_features as S4                         # noqa: E402
import s05_build_folds as S5                                      # noqa: E402
import s08_shap_validation as S8                                  # noqa: E402

random.seed(C.SEED)


def random_dna(n: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(n))


# ── k-mer counting (stage 03) ─────────────────────────────────────────────────
def naive_counts(sequence: str) -> np.ndarray:
    s = sequence.upper()
    counts = Counter(s[i:i + C.K] for i in range(len(s) - C.K + 1)
                     if set(s[i:i + C.K]) <= set("ACGT"))
    return np.array([counts.get(k, 0) for k in S3.build_vocabulary()])


def test_kmer_counts_match_naive_counter():
    seq = random_dna(4000) + "NNN" + random_dna(1500)
    assert np.array_equal(S3.count_kmers(seq), naive_counts(seq))


def test_kmer_counts_ignore_case():
    # BV-BRC serves lowercase sequence; an encoder that missed this would
    # silently return all zeros
    seq = random_dna(3000)
    assert np.array_equal(S3.count_kmers(seq.lower()), S3.count_kmers(seq))
    assert S3.count_kmers(seq.lower()).sum() == len(seq) - C.K + 1


def test_kmer_windows_with_ambiguous_bases_are_skipped():
    assert S3.count_kmers("ACGTNACGTA").sum() == 0


# ── QRDR reader (stage 04) ────────────────────────────────────────────────────
@pytest.mark.parametrize("site", list(S4.QRDR_SITES))
@pytest.mark.parametrize("codon,amino_acid", [("TCG", "S"), ("TTG", "L"),
                                              ("GAC", "D"), ("AAC", "N"),
                                              ("GAA", "E"), ("GTA", "V")])
def test_qrdr_reads_codon_on_both_strands(site, codon, amino_acid):
    flank, side, _, _ = S4.QRDR_SITES[site]
    core = flank + codon if side == "left" else codon + flank
    seq = random_dna(400) + core + random_dna(400)
    assert S4.call_qrdr_site(seq, flank, side) == amino_acid
    assert S4.call_qrdr_site(S4._revcomp(seq), flank, side) == amino_acid


def test_qrdr_returns_none_when_flank_absent():
    flank, side, _, _ = S4.QRDR_SITES["gyrA_83"]
    assert S4.call_qrdr_site("A" * 1000, flank, side) is None


def test_revcomp_is_its_own_inverse():
    seq = random_dna(200)
    assert S4._revcomp(S4._revcomp(seq)) == seq


# ── folds (stage 05) ──────────────────────────────────────────────────────────
def test_sequence_type_parsing():
    assert S5.extract_sequence_type("MLST.ecoli_achtman_4.131") == "131"
    assert S5.extract_sequence_type(
        "MLST.ecoli_pasteur.5;MLST.ecoli_achtman_4.73") == "73"
    assert S5.extract_sequence_type(None) is None


def test_country_names_are_merged():
    assert S5.normalise_country("England") == "United Kingdom"
    assert S5.normalise_country(" united kingdom ") == "United Kingdom"
    assert S5.normalise_country("USA") == "United States"
    assert S5.normalise_country("Norway") == "Norway"


def test_country_folds_treat_england_as_united_kingdom():
    df = pd.DataFrame({"isolation_country": ["England"] * 60 + ["United Kingdom"] * 60,
                       "label": [0, 1] * 60})
    folds, eligible = S5.assign_country_folds(df)
    assert eligible == ["United Kingdom"]
    assert (folds == 0).all()


# ── statistics (stage 08) ─────────────────────────────────────────────────────
def test_canonical_hexamers():
    groups = S8.canonical_groups(S3.build_vocabulary())
    assert len(groups) == 2080
    assert sum(1 for g in groups if g[1]) == 64


def test_epsilon_squared_removes_group_count_inflation():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=7000)
    codes = rng.integers(0, 750, 7000)
    eta2, eps2 = S8.variance_explained(noise, codes)
    assert eta2 > 0.08          # pure noise still looks "explained"
    assert abs(eps2) < 0.02     # the correction removes it


def test_binomial_tail_reference_value():
    # 13 of 20 palindromic at 64/4096 was the original single-strand result
    assert S8.binomial_tail(13, 20, 64 / 4096) == pytest.approx(2.3155e-19, rel=1e-3)


# ── real data (skipped if the pipeline has not been run) ──────────────────────
@pytest.mark.parametrize("antibiotic", C.ANTIBIOTICS)
def test_no_sequence_type_in_two_mlst_folds(antibiotic):
    path = C.PROCESSED / f"folds_{antibiotic}.csv"
    if not path.exists():
        pytest.skip("fold files not built")
    folds = pd.read_csv(path, dtype={"sequence_type": str})
    typed = folds[folds["sequence_type"].notna()]
    assert (typed.groupby("sequence_type")["fold_mlst"].nunique() == 1).all()
