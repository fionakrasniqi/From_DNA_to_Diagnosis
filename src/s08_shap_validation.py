"""
s08_shap_validation.py — feature attribution and biological validation (§2.5).

The proposal asks for SHAP on the XGBoost model, the top 20 hexamers, and
validation against gyrA 83/87 and parC 80/84. The mechanism it specifies —
BLAST the top hexamers — cannot work: a 6 bp word occurs more than 1,000 times
per genome by chance (5.07 Mb / 4,096), so a match cannot be tied to one locus.
The intent is kept and the method replaced with tests that are decisive.

Canonical hexamers. Stage 03 counts one strand only, so a non-palindromic
hexamer and its reverse complement are two near-identical columns, while a
palindrome is a single column. Ranking single columns would therefore favour
palindromes for a purely technical reason. SHAP values are summed over each
hexamer and its reverse complement before ranking (2,080 canonical hexamers,
64 of them palindromic, base rate 3.08%). The single-strand ranking is still
written for comparison.

Test 1 — enrichment. Are the top hexamers over-represented inside the
  gyrA/parC regions extracted in stage 07, relative to the genome? A weak test,
  reported for completeness.

Test 2 — lineage attribution. How much of each hexamer's variation is explained
  by MLST sequence type? Eta-squared rises with the number of groups even for
  pure noise (about (k-1)/(N-1), ~0.10 with ~740 sequence types), so epsilon-
  squared, which subtracts that inflation, is reported alongside, together with
  a permutation baseline (sequence types shuffled 200 times). Untyped genomes
  are excluded from this test.

Test 3 — genotype attribution. The same measures against QRDR genotype
  (a resistance allele at gyrA 83 or parC 80).

SHAP is also run on the eight QRDR features as a positive control.

Output:
    results/tables/shap_top_hexamers_canonical.csv   main ranking used in the thesis
    results/tables/shap_palindrome_test.csv           palindrome count and binomial test
    results/tables/shap_top_hexamers.csv              single-strand ranking (comparison)
    results/tables/shap_qrdr_control.csv
    results/figures/08_shap_summary.png
"""

from __future__ import annotations

import logging
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import config as C

log = logging.getLogger("s08")

TOP_N = 20
N_PERMUTATIONS = 200

# Recognition sites verified against the NEB list of restriction enzymes
# (the company that maintains REBASE). Used to label the palindromes found.
ENZYMES = {
    "AGATCT": "BglII", "ATTAAT": "AseI", "CAATTG": "MfeI", "CAGCTG": "PvuII",
    "CATATG": "NdeI", "GCTAGC": "NheI", "GGATCC": "BamHI", "GTGCAC": "ApaLI",
    "GTTAAC": "HpaI", "TTATAA": "PsiI", "TCATGA": "BspHI", "CGGCCG": "EagI",
    "CACGTG": "PmlI", "ACGCGT": "MluI", "GATATC": "EcoRV", "GAATTC": "EcoRI",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def canonical_groups(vocabulary) -> list[tuple[str, bool, list[int]]]:
    """Group each hexamer with its reverse complement.

    Returns (canonical name, is_palindrome, column indices). 4,096 hexamers give
    2,080 groups: 64 palindromes (one column) and 2,016 pairs (two columns).
    """
    index = {str(h): i for i, h in enumerate(vocabulary)}
    groups, seen = [], set()
    for h in map(str, vocabulary):
        if h in seen:
            continue
        rc = revcomp(h)
        seen.update([h, rc])
        members = [index[h]] if rc == h else [index[h], index[rc]]
        groups.append((min(h, rc), rc == h, members))
    return groups


def variance_explained(values: np.ndarray, codes: np.ndarray) -> tuple[float, float]:
    """Return (eta², epsilon²) for a one-way decomposition by integer group codes.

    eta² = SS_between / SS_total; epsilon² = (SS_between - (k-1) MS_within) / SS_total.
    """
    values = np.asarray(values, dtype=float)
    counts = np.bincount(codes)
    present = counts > 0
    k, n = int(present.sum()), len(values)
    sums = np.bincount(codes, weights=values)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=present)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0 or k < 2 or n <= k:
        return float("nan"), float("nan")
    ss_between = float((counts[present] * (means[present] - grand) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    return ss_between / ss_total, (ss_between - (k - 1) * ms_within) / ss_total


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def fit_xgboost(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """Same hyperparameters as stage 06 so attributions match the reported model."""
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.15,
        subsample=0.8, colsample_bytree=0.2, max_bin=64,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=C.SEED,
    )
    model.fit(X, y)
    return model


def count_in_regions(regions: np.ndarray, hexamer: str) -> np.ndarray:
    """Occurrences of a hexamer (both strands) in each genome's gyrA/parC regions."""
    code = {b: i for i, b in enumerate("ACGT")}
    forward = np.array([code[b] for b in hexamer], dtype=np.int8)
    backward = np.array([code[b] for b in revcomp(hexamer)], dtype=np.int8)
    counts = np.zeros(len(regions), dtype=np.int32)
    for pattern in (forward, backward):
        windows = np.lib.stride_tricks.sliding_window_view(regions, len(pattern), axis=1)
        counts += (windows == pattern).all(axis=2).sum(axis=1).astype(np.int32)
        if np.array_equal(forward, backward):
            break
    return counts


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    antibiotic = C.PRIMARY_ANTIBIOTIC
    fold_path = C.PROCESSED / f"folds_{antibiotic}.csv"
    if not fold_path.exists():
        log.error("missing %s — run s05_build_folds.py first", fold_path)
        return 1

    folds = pd.read_csv(fold_path, dtype={"genome_id": str, "sequence_type": str})
    genome_ids = np.load(C.PROCESSED / "kmer_genome_ids.npy", allow_pickle=True)
    id_to_row = {gid: i for i, gid in enumerate(genome_ids)}
    rows = np.array([id_to_row[g] for g in folds["genome_id"]])

    X_kmer = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")[rows]
    X_qrdr = np.load(C.PROCESSED / "X_qrdr.npy").astype(np.float32)[rows]
    qrdr_names = np.load(C.PROCESSED / "qrdr_feature_names.npy", allow_pickle=True)
    vocabulary = [str(v) for v in np.load(C.PROCESSED / "kmer_vocabulary.npy",
                                          allow_pickle=True)]
    y = folds["label"].to_numpy()
    log.info("%s: %d genomes, %.1f%% resistant", antibiotic, len(y), y.mean() * 100)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    log.info("fitting XGBoost on hexamer features and computing SHAP values")
    shap_values = shap.TreeExplainer(fit_xgboost(X_kmer, y)).shap_values(X_kmer)

    log.info("positive control: XGBoost on QRDR features")
    qrdr_shap = shap.TreeExplainer(fit_xgboost(X_qrdr, y)).shap_values(X_qrdr)
    control = (pd.DataFrame({"feature": qrdr_names,
                             "mean_abs_shap": np.abs(qrdr_shap).mean(axis=0)})
               .sort_values("mean_abs_shap", ascending=False))
    control.to_csv(C.TABLES / "shap_qrdr_control.csv", index=False)

    # ── grouping variables ───────────────────────────────────────────────────
    typed = folds["sequence_type"].notna().to_numpy()
    st_codes = pd.factorize(folds.loc[typed, "sequence_type"])[0]
    residues = pd.read_csv(C.PROCESSED / "qrdr_residues.csv", dtype={"genome_id": str})
    residues = folds[["genome_id"]].merge(residues, on="genome_id", how="left")
    genotype = ((residues["gyrA_83"].isin(["L", "A", "T", "W", "V"]))
                | (residues["parC_80"].isin(["I", "R"]))).astype(int).to_numpy()

    regions = None
    targeted_path = C.PROCESSED / "cnn_windows_targeted.npy"
    cnn_ids_path = C.PROCESSED / "cnn_genome_ids.npy"
    if targeted_path.exists() and cnn_ids_path.exists():
        if np.array_equal(np.load(cnn_ids_path, allow_pickle=True),
                          folds["genome_id"].to_numpy()):
            regions = np.load(targeted_path)
    if regions is None:
        log.warning("gyrA/parC regions from s07 not available; enrichment test skipped")

    rng = np.random.default_rng(C.SEED)

    def describe(columns: list[int]) -> dict:
        freq = X_kmer[:, columns].sum(axis=1).astype(float)
        eta_l, eps_l = variance_explained(freq[typed], st_codes)
        null = [variance_explained(freq[typed], rng.permutation(st_codes))[1]
                for _ in range(N_PERMUTATIONS)]
        eta_g, eps_g = variance_explained(freq, genotype)
        record = {"freq_resistant": freq[y == 1].mean(),
                  "freq_susceptible": freq[y == 0].mean(),
                  "lineage_eta2": eta_l, "lineage_eps2": eps_l,
                  "lineage_eps2_null_mean": float(np.mean(null)),
                  "lineage_eps2_null_p95": float(np.percentile(null, 95)),
                  "genotype_eta2": eta_g, "genotype_eps2": eps_g}
        if regions is not None:
            hexamer = vocabulary[columns[0]]
            region_rate = count_in_regions(regions, hexamer).mean() / regions.shape[1]
            record["enrichment_vs_genome"] = (region_rate / freq.mean()
                                              if freq.mean() > 0 else np.nan)
        return record

    # ── canonical ranking (main result) ──────────────────────────────────────
    groups = canonical_groups(vocabulary)
    n_pal = sum(1 for g in groups if g[1])
    base_rate = n_pal / len(groups)
    importance = np.array([np.abs(shap_values[:, m].sum(axis=1)).mean()
                           for _, _, m in groups])
    order = np.argsort(importance)[::-1][:TOP_N]

    log.info("validating top %d canonical hexamers", TOP_N)
    canonical = []
    for rank, gi in enumerate(order, start=1):
        name, palindromic, members = groups[gi]
        canonical.append({"rank": rank, "hexamer": name, "palindromic": palindromic,
                          "enzyme": ENZYMES.get(name, "") if palindromic else "",
                          "mean_abs_shap": importance[gi], **describe(members)})
    canonical = pd.DataFrame(canonical)
    canonical.to_csv(C.TABLES / "shap_top_hexamers_canonical.csv", index=False)

    k = int(canonical["palindromic"].sum())
    p_value = binomial_tail(k, TOP_N, base_rate)
    pd.DataFrame([{"top_n": TOP_N, "palindromic": k, "canonical_hexamers": len(groups),
                   "palindromes_possible": n_pal, "base_rate": base_rate,
                   "binomial_p": p_value}]).to_csv(
        C.TABLES / "shap_palindrome_test.csv", index=False)

    # ── single-strand ranking, kept for comparison ───────────────────────────
    single_importance = np.abs(shap_values).mean(axis=0)
    single = [{"rank": r, "hexamer": vocabulary[i],
               "palindromic": revcomp(vocabulary[i]) == vocabulary[i],
               "mean_abs_shap": single_importance[i]}
              for r, i in enumerate(np.argsort(single_importance)[::-1][:TOP_N], start=1)]
    pd.DataFrame(single).to_csv(C.TABLES / "shap_top_hexamers.csv", index=False)

    # ── figure (Albanian labels) ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    labels = [f"{h} ({e})" if e else h
              for h, e in zip(canonical["hexamer"], canonical["enzyme"])][::-1]
    colours = ["#4C72B0" if p else "#BBBBBB" for p in canonical["palindromic"]][::-1]
    axes[0].barh(range(TOP_N), canonical["mean_abs_shap"][::-1], color=colours)
    axes[0].set_yticks(range(TOP_N))
    axes[0].set_yticklabels(labels, fontsize=8, family="monospace")
    axes[0].set_xlabel("Mesatarja e |SHAP|")
    axes[1].barh(range(TOP_N), canonical["lineage_eps2"][::-1], color="#C44E52")
    axes[1].axvline(canonical["lineage_eps2_null_p95"].max(), color="black",
                    linestyle="--", linewidth=1, label="Niveli i rastësishëm (95%)")
    axes[1].set_yticks(range(TOP_N))
    axes[1].set_yticklabels(canonical["hexamer"][::-1], fontsize=8, family="monospace")
    axes[1].set_xlabel("Varianca e shpjeguar nga linja (ε²)")
    axes[1].set_xlim(0, 1)
    axes[1].legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(C.FIGURES / "08_shap_summary.png", dpi=150)
    plt.close()

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SHAP CONTROL — QRDR model (8 features)")
    print("=" * 78)
    print(control.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print(f"TOP {TOP_N} CANONICAL HEXAMERS")
    print("=" * 78)
    cols = ["rank", "hexamer", "palindromic", "enzyme", "mean_abs_shap",
            "lineage_eps2", "lineage_eps2_null_p95", "genotype_eps2"]
    print(canonical[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\npalindromic: {k}/{TOP_N} | base rate {base_rate:.2%} | binomial p = {p_value:.2e}")
    print(f"mean epsilon²: lineage {canonical['lineage_eps2'].mean():.3f} "
          f"(permutation baseline {canonical['lineage_eps2_null_mean'].mean():.3f}) | "
          f"QRDR genotype {canonical['genotype_eps2'].mean():.3f}")
    if "enrichment_vs_genome" in canonical:
        print(f"enriched in gyrA/parC regions: "
              f"{int((canonical['enrichment_vs_genome'] > 1).sum())}/{TOP_N} (weak test)")

    print(f"\nwrote {C.TABLES / 'shap_top_hexamers_canonical.csv'}")
    print(f"wrote {C.TABLES / 'shap_palindrome_test.csv'}")
    print(f"wrote {C.FIGURES / '08_shap_summary.png'}")
    print("\nnext: python s09_make_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
