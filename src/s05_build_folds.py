"""
s05_build_folds.py — evaluation protocol (proposal §2.4).

Constructs the four cross-validation designs the study compares, and validates
the clustering that the proposal's phylogeny-aware protocol depends on.

Why several designs. A random split places close relatives on both sides of the
fold boundary, so a model can score well by recognising a lineage it has already
seen rather than by detecting resistance. Grouped and geographic splits break
that shortcut in two different ways, and the gap between designs is the finding
the thesis reports.

  1. random      StratifiedKFold — the optimistic baseline
  2. grouped     GroupKFold over MiniBatchKMeans clusters of k-mer profiles,
                 as specified in §2.4
  3. mlst        GroupKFold over MLST sequence types — added because the
                 k-means grouping does not separate lineages
  4. geographic  leave-one-country-out, replacing the NCBI Pathogen external
                 set (see the deviations table in the README)

Country names are normalised before the geographic design is built. BV-BRC
records the same place under several labels ("England" and "United Kingdom"
both occur), and leaving them separate would put English genomes in training
while the United Kingdom is held out, which defeats the purpose of the split.

Clustering validation. §1.4 claims phylogeny-aware evaluation as a contribution,
which rests on the k-means clusters actually tracking lineage. That is testable
here rather than assumed: BV-BRC reports MLST sequence types, so the adjusted
Rand index between clusters and true STs measures it directly.

This matters for a specific reason. The earlier attempt clustered truncated
assemblies whose lengths ranged 0.6-4.9 Mb, so the clusters tracked assembly
fragmentation rather than lineage. On complete genomes with frequency-normalised
profiles that confound is gone, but the claim still needs evidence.

Output:
    data/processed/folds_<antibiotic>.csv    per-genome fold assignment per design
    data/processed/cluster_assignments.csv   k-means cluster + MLST per genome
    results/tables/clustering_validation.csv ARI and cluster composition
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

import config as C

log = logging.getLogger("s05")

MIN_COUNTRY_SIZE = 100      # smallest country cohort worth holding out

# Labels that refer to the same country. Matching is case-insensitive and
# ignores surrounding spaces. Extend this if the country listing printed by
# fix_country_folds.py shows another duplicate.
COUNTRY_ALIASES = {
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "great britain": "United Kingdom",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "united states of america": "United States",
}


def normalise_country(value):
    """Map a raw isolation_country label to one name per country."""
    if not isinstance(value, str) or not value.strip():
        return value
    cleaned = value.strip()
    return COUNTRY_ALIASES.get(cleaned.lower(), cleaned)


def assign_country_folds(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Leave-one-country-out folds on normalised country names.

    One fold per country with at least MIN_COUNTRY_SIZE genomes and both
    classes present; every other genome gets -1 and is used only for training.
    Returns the fold column and the list of held-out countries in fold order.
    """
    country = df["isolation_country"].map(normalise_country)
    folds = pd.Series(-1, index=df.index, dtype=int)
    eligible = [
        name for name, count in country.value_counts().items()
        if count >= MIN_COUNTRY_SIZE
        and df.loc[country == name, "label"].nunique() == 2
    ]
    for fold, name in enumerate(eligible):
        folds[country == name] = fold
    return folds, eligible


def extract_sequence_type(mlst_value) -> str | None:
    """Pull the ST number out of BV-BRC's MLST string.

    Values look like 'MLST.ecoli_achtman_4.95' where the trailing number is the
    sequence type. Some genomes carry several schemes separated by semicolons;
    the first ecoli_achtman entry is the standard one.
    """
    if not isinstance(mlst_value, str):
        return None
    for part in mlst_value.split(";"):
        part = part.strip()
        if "achtman" in part.lower():
            tail = part.rsplit(".", 1)[-1]
            return tail if tail.isdigit() else None
    return None


def validate_clustering(X: np.ndarray, genome_ids: np.ndarray,
                        metadata: pd.DataFrame) -> pd.DataFrame:
    """Cluster k-mer profiles and measure agreement with MLST sequence types."""
    log.info("clustering %d genomes into %d groups", len(genome_ids), C.N_CLUSTERS)

    kmeans = MiniBatchKMeans(n_clusters=C.N_CLUSTERS, random_state=C.SEED,
                             n_init=10, batch_size=1024)
    clusters = kmeans.fit_predict(X)

    assignments = pd.DataFrame({"genome_id": genome_ids, "cluster": clusters})
    assignments = assignments.merge(
        metadata[["genome_id", "mlst", "genome_length", "isolation_country"]],
        on="genome_id", how="left")
    assignments["sequence_type"] = assignments["mlst"].apply(extract_sequence_type)

    typed = assignments[assignments["sequence_type"].notna()]
    log.info("MLST available for %d/%d genomes (%.1f%%)",
             len(typed), len(assignments), len(typed) / len(assignments) * 100)

    ari = adjusted_rand_score(typed["sequence_type"], typed["cluster"])
    ami = adjusted_mutual_info_score(typed["sequence_type"], typed["cluster"])
    log.info("cluster vs MLST:  ARI=%.3f  AMI=%.3f", ari, ami)

    # ARI is structurally unfair here: the cohort holds ~750 sequence types and
    # the protocol uses 5 clusters, so no assignment can score well. Two fairer
    # measures of whether the grouping does its job:
    #
    #   major-ST concentration — for each of the largest lineages, the share
    #   falling in its most common cluster. High values mean big lineages are
    #   kept together rather than smeared across folds.
    #
    #   ST split rate — the share of multi-genome sequence types that appear in
    #   more than one cluster. This is the leakage the grouping exists to stop.
    st_sizes = typed["sequence_type"].value_counts()
    major_sts = st_sizes[st_sizes >= 50].index

    concentrations = {}
    for st in major_sts:
        rows = typed[typed["sequence_type"] == st]
        concentrations[st] = rows["cluster"].value_counts(normalize=True).iloc[0]

    multi = st_sizes[st_sizes >= 2].index
    split = sum(typed[typed["sequence_type"] == st]["cluster"].nunique() > 1
                for st in multi)
    split_rate = split / len(multi) if len(multi) else np.nan

    log.info("major-ST concentration (mean over %d STs with n>=50): %.3f",
             len(major_sts), np.mean(list(concentrations.values())))
    log.info("multi-genome STs split across clusters: %.1f%%", split_rate * 100)

    # The failure mode from the first attempt: clusters tracking assembly size.
    # On complete genomes this correlation should be near zero.
    length_spread = assignments.groupby("cluster")["genome_length"].mean()
    log.info("mean genome length per cluster: %s",
             ", ".join(f"{v/1e6:.2f}Mb" for v in length_spread))

    summary = (assignments.groupby("cluster")
               .agg(n_genomes=("genome_id", "count"),
                    mean_length_mb=("genome_length", lambda s: s.mean() / 1e6),
                    n_sequence_types=("sequence_type", "nunique"),
                    dominant_st=("sequence_type",
                                 lambda s: s.value_counts().idxmax()
                                 if s.notna().any() else None),
                    dominant_st_share=("sequence_type",
                                       lambda s: s.value_counts(normalize=True).iloc[0]
                                       if s.notna().any() else np.nan))
               .reset_index())
    summary["ari_vs_mlst"] = ari
    summary["ami_vs_mlst"] = ami
    summary["major_st_concentration"] = np.mean(list(concentrations.values())) if concentrations else np.nan
    summary["st_split_rate"] = split_rate

    st_detail = pd.DataFrame(
        {"sequence_type": list(concentrations.keys()),
         "n_genomes": [int(st_sizes[st]) for st in concentrations],
         "concentration": list(concentrations.values())}
    ).sort_values("n_genomes", ascending=False)

    return assignments, summary, st_detail


def build_folds(labels: pd.DataFrame, assignments: pd.DataFrame,
                antibiotic: str) -> pd.DataFrame:
    """Assign each genome a fold under each of the four designs."""
    df = labels.merge(assignments, on="genome_id", how="left")
    n = len(df)

    # 1. random stratified
    df["fold_random"] = -1
    skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(n), df["label"])):
        df.iloc[test_idx, df.columns.get_loc("fold_random")] = fold

    # 2. grouped by k-mer cluster (proposal §2.4)
    df["fold_grouped"] = -1
    gkf = GroupKFold(n_splits=C.N_SPLITS)
    for fold, (_, test_idx) in enumerate(
            gkf.split(np.zeros(n), df["label"], groups=df["cluster"])):
        df.iloc[test_idx, df.columns.get_loc("fold_grouped")] = fold

    # 3. grouped by MLST sequence type — the corrected phylogeny-aware design.
    #    The k-means protocol in §2.4 turns out not to separate lineages (major
    #    STs land ~35-45% in their dominant cluster, barely above the ~20%
    #    expected by chance with 5 clusters), so grouping on the sequence type
    #    itself is what actually keeps a lineage on one side of the boundary.
    #    Genomes with no ST call become singleton groups so they cannot leak.
    st = df["sequence_type"].copy()
    missing = st.isna()
    st[missing] = [f"__untyped_{i}" for i in range(int(missing.sum()))]

    df["fold_mlst"] = -1
    gkf_st = GroupKFold(n_splits=C.N_SPLITS)
    for fold, (_, test_idx) in enumerate(
            gkf_st.split(np.zeros(n), df["label"], groups=st)):
        df.iloc[test_idx, df.columns.get_loc("fold_mlst")] = fold

    # 4. leave-one-country-out on normalised country names — a fold per
    #    country large enough to hold out, everything else pooled into fold -1
    #    and used only for training
    df["isolation_country"] = df["isolation_country"].map(normalise_country)
    df["fold_country"], eligible = assign_country_folds(df)

    log.info("%s: %d genomes | %d country folds (%s)",
             antibiotic, n, len(eligible), ", ".join(eligible[:6]))

    return df[["genome_id", "label", "cluster", "sequence_type",
               "isolation_country", "fold_random", "fold_grouped",
               "fold_mlst", "fold_country"]]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    X = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")
    genome_ids = np.load(C.PROCESSED / "kmer_genome_ids.npy")
    metadata = pd.read_csv(C.PROCESSED / "genome_metadata.csv",
                           dtype={"genome_id": str})
    phenotypes = pd.read_csv(C.PHENOTYPES / "phenotypes_long.csv",
                             dtype={"genome_id": str})

    log.info("k-mer matrix %s", X.shape)

    # ── clustering + validation ──────────────────────────────────────────────
    assignments, summary, st_detail = validate_clustering(X, genome_ids, metadata)
    assignments.to_csv(C.PROCESSED / "cluster_assignments.csv", index=False)
    summary.to_csv(C.TABLES / "clustering_validation.csv", index=False)
    st_detail.to_csv(C.TABLES / "major_st_concentration.csv", index=False)

    # ── folds per antibiotic ─────────────────────────────────────────────────
    in_matrix = set(genome_ids)
    for antibiotic in C.ANTIBIOTICS:
        subset = phenotypes[(phenotypes["antibiotic"] == antibiotic)
                            & (phenotypes["genome_id"].isin(in_matrix))]
        if len(subset) < 100:
            log.warning("%s: only %d genomes, skipping", antibiotic, len(subset))
            continue
        folds = build_folds(subset[["genome_id", "label"]], assignments, antibiotic)
        folds.to_csv(C.PROCESSED / f"folds_{antibiotic}.csv", index=False)

    # ── report ───────────────────────────────────────────────────────────────
    ari = summary["ari_vs_mlst"].iloc[0]
    ami = summary["ami_vs_mlst"].iloc[0]
    conc = summary["major_st_concentration"].iloc[0]
    split_rate = summary["st_split_rate"].iloc[0]

    print("\n" + "=" * 66)
    print("CLUSTERING VALIDATION  (does k-means track lineage?)")
    print("=" * 66)
    print(f"adjusted Rand index vs MLST      : {ari:.3f}")
    print(f"adjusted mutual information      : {ami:.3f}")
    print(f"major-ST concentration (n>=50)   : {conc:.3f}")
    print(f"multi-genome STs split           : {split_rate * 100:.1f}%")
    print()
    print("  ARI is structurally limited here: ~750 sequence types cannot be")
    print("  recovered by 5 clusters, so a low value is a granularity artifact")
    print("  rather than evidence the clustering failed. Concentration and split")
    print("  rate measure what the grouping is actually for.")
    print()
    print("top lineages — share falling in their dominant cluster:")
    print(st_detail.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nper-cluster composition:")
    print(summary[["cluster", "n_genomes", "mean_length_mb",
                   "n_sequence_types", "dominant_st", "dominant_st_share"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nINTERPRETATION")
    print("  mean_length_mb near-identical across clusters means the clusters are")
    print("  NOT tracking assembly size — the failure mode of the first attempt.")
    if conc > 0.8:
        print(f"  concentration {conc:.2f}: major lineages stay largely intact within")
        print("  clusters, so grouped folds do separate populations. This supports")
        print("  the phylogeny-aware claim in proposal §1.4.")
    elif conc > 0.6:
        print(f"  concentration {conc:.2f}: partial lineage separation. Describe the")
        print("  grouping as capturing broad population structure rather than")
        print("  resolving individual sequence types.")
    else:
        print(f"  concentration {conc:.2f}: major lineages are spread across clusters,")
        print("  so grouped CV is composition-based rather than phylogeny-aware.")
        print("  Report it as such — and note MLST-grouped CV as an alternative.")

    print("\n" + "=" * 66)
    print("FOLD DESIGNS WRITTEN")
    print("=" * 66)
    for antibiotic in C.ANTIBIOTICS:
        path = C.PROCESSED / f"folds_{antibiotic}.csv"
        if not path.exists():
            continue
        folds = pd.read_csv(path, dtype={"genome_id": str})
        n_country = (folds["fold_country"] >= 0).sum()
        print(f"\n  {antibiotic}  n={len(folds)}  R={int(folds['label'].sum())}  "
              f"country-eligible={n_country}")
        for design in ["fold_random", "fold_grouped", "fold_mlst"]:
            rates = (folds[folds[design] >= 0]
                     .groupby(design)["label"].agg(["size", "mean"]))
            sizes = "/".join(str(int(v)) for v in rates["size"])
            pcts = "/".join(f"{v*100:.0f}%" for v in rates["mean"])
            print(f"      {design:14} sizes {sizes}")
            print(f"      {'':14} R-rate {pcts}")

    print("\nnext: python src/s06_train_classical.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
