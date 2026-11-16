"""
full_check.py — read-only verification for the Full Check.

Reads the pipeline's existing outputs and writes one compact text report,
results/full_check_report.txt, to send back for review. Nothing is refitted
except in section J, which recomputes SHAP for the hexamer model exactly as
s08 does (same hyperparameters, same seed) so the ranking can be redone with
canonical hexamers. No pipeline file is modified.

Run from src/:
    python full_check.py                 # all sections (~3-5 min, SHAP is the slow part)
    python full_check.py --skip-shap     # everything except section J (~20 s)
    python full_check.py --conflicts     # also re-query BV-BRC to count
                                         # conflicting R/S labels (~3 min, needs internet)

Each section is independent: if one fails, the error is written into the
report and the rest still run.
"""

from __future__ import annotations

import argparse
import io
import math
import platform
import sys
import traceback
from contextlib import redirect_stdout
from itertools import product

import numpy as np
import pandas as pd

import config as C

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

SITES = ["gyrA_83", "gyrA_87", "parC_80", "parC_84"]
WILD_TYPE = {"gyrA_83": "S", "gyrA_87": "D", "parC_80": "S", "parC_84": "E"}
RESISTANT = {"gyrA_83": {"L", "A", "T", "W", "V"},
             "gyrA_87": {"N", "G", "Y", "A", "H", "V"},
             "parC_80": {"I", "R"},
             "parC_84": {"V", "G", "K", "A"}}
DESIGNS = ["fold_random", "fold_grouped", "fold_mlst", "fold_country"]
FEATURE_SETS = ["kmer", "card", "qrdr", "combined"]


# ── helpers ───────────────────────────────────────────────────────────────────
def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def read_folds(antibiotic: str) -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / f"folds_{antibiotic}.csv",
                       dtype={"genome_id": str, "sequence_type": str})


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def eta_squared(values: np.ndarray, codes: np.ndarray) -> tuple[float, float, int]:
    """Return (eta², epsilon², number of groups) for a one-way decomposition.

    eta² = SS_between / SS_total rises with the number of groups even for pure
    noise, by roughly (k-1)/(N-1). epsilon² subtracts that expected inflation:
    (SS_between - (k-1) * MS_within) / SS_total.
    """
    n = len(values)
    counts = np.bincount(codes)
    present = counts > 0
    k = int(present.sum())
    sums = np.bincount(codes, weights=values)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=present)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0 or k < 2 or n <= k:
        return float("nan"), float("nan"), k
    ss_between = float((counts[present] * (means[present] - grand) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    eta2 = ss_between / ss_total
    eps2 = (ss_between - (k - 1) * ms_within) / ss_total
    return eta2, eps2, k


# ── A. cohort after QC ────────────────────────────────────────────────────────
def section_a() -> None:
    header("A. COHORT — before QC (s01) versus analysed (fold files)")
    pre = pd.read_csv(C.TABLES / "cohort_summary.csv").set_index("antibiotic")
    rows = []
    for ab in C.ANTIBIOTICS:
        f = read_folds(ab)
        rows.append({"antibiotic": ab,
                     "pre_qc_n": int(pre.loc[ab, "n_genomes"]) if ab in pre.index else None,
                     "pre_qc_pctR": float(pre.loc[ab, "pct_resistant"]) if ab in pre.index else None,
                     "analysed_n": len(f), "analysed_R": int(f["label"].sum()),
                     "analysed_S": int((1 - f["label"]).sum()),
                     "analysed_pctR": round(f["label"].mean() * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    ids = np.load(C.PROCESSED / "kmer_genome_ids.npy", allow_pickle=True)
    print(f"\ngenomes in k-mer matrix: {len(ids):,}")

    f = read_folds(C.PRIMARY_ANTIBIOTIC)
    print(f"\n{C.PRIMARY_ANTIBIOTIC}: MLST typed {f['sequence_type'].notna().mean() * 100:.1f}%"
          f"  | distinct STs {f['sequence_type'].nunique()}")

    by_country = (f.groupby(f["isolation_country"].fillna("Unknown"))["label"]
                  .agg(n="size", pctR="mean").sort_values("n", ascending=False))
    by_country["pctR"] = (by_country["pctR"] * 100).round(1)
    by_country["share_of_cohort_%"] = (by_country["n"] / len(f) * 100).round(1)
    print("\ntop 10 countries (ciprofloxacin cohort):")
    print(by_country.head(10).to_string())

    by_st = f.groupby("sequence_type")["label"].agg(n="size", pctR="mean")
    by_st["pctR"] = (by_st["pctR"] * 100).round(1)
    wanted = ["131", "405", "10", "69", "95", "73"]
    print("\nselected STs (ciprofloxacin cohort):")
    print(by_st.reindex(wanted).to_string())
    print("\ntop 10 STs by size:")
    print(by_st.sort_values("n", ascending=False).head(10).to_string())


# ── B. QC and download ────────────────────────────────────────────────────────
def section_b() -> None:
    header("B. DOWNLOAD AND QC")
    man = pd.read_csv(C.PROCESSED / "download_manifest.csv", dtype={"genome_id": str})
    print("download status:\n" + man["status"].value_counts().to_string())
    valid = man[man["status"].isin(["ok", "cached"])]
    print(f"validated: {len(valid):,}  | median observed bp {valid['observed_bp'].median():,.0f}")

    md = pd.read_csv(C.PROCESSED / "genome_metadata.csv", dtype={"genome_id": str})
    missing_len = int(md["genome_length"].isna().sum() + (md["genome_length"] <= 0).sum())
    print(f"\nmetadata rows: {len(md):,}  | genome_length missing or zero: {missing_len}")
    print(f"thresholds: CheckM completeness >= {C.MIN_CHECKM_COMPLETENESS}%, "
          f"contamination <= {C.MAX_CONTAMINATION}%, download >= {C.MIN_COMPLETENESS:.0%}")

    qc = pd.read_csv(C.PROCESSED / "qc_excluded.csv", dtype={"genome_id": str})
    reasons = pd.DataFrame({
        "size_outside_4.4-6.0Mb": ~qc["genome_length"].between(4_400_000, 6_000_000),
        "poor_quality": qc["genome_quality"].fillna("").eq("Poor"),
        "low_checkm_completeness": qc["checkm_completeness"].fillna(100) < C.MIN_CHECKM_COMPLETENESS,
        "high_contamination": qc["checkm_contamination"].fillna(0) > C.MAX_CONTAMINATION,
    })
    print(f"\nQC excluded: {len(qc)} genomes (criteria can overlap)")
    print(reasons.sum().to_string())
    print(f"excluded by more than one criterion: {int((reasons.sum(axis=1) > 1).sum())}")


# ── C. conflicting labels (optional, network) ─────────────────────────────────
def section_c(run: bool) -> None:
    header("C. CONFLICTING R/S LABELS FOR THE SAME GENOME")
    if not run:
        print("skipped (run with --conflicts to re-query BV-BRC)")
        return
    import requests
    import s01_fetch_phenotypes as S1

    session = requests.Session()
    for ab in C.ANTIBIOTICS:
        raw = S1.fetch_antibiotic(session, ab).rename(columns=S1.RENAME)
        raw["genome_id"] = raw["genome_id"].astype(str).str.strip().str.strip('"')
        raw = raw[raw["evidence"].astype(str).str.strip().str.lower() == C.REQUIRED_EVIDENCE]
        raw = raw[raw["phenotype"].isin(C.VALID_PHENOTYPES)]
        per_genome = raw.groupby("genome_id")["phenotype"].nunique()
        conflicts = int((per_genome > 1).sum())
        print(f"{ab:15} genomes {len(per_genome):>6,}  with conflicting R and S: {conflicts:>5} "
              f"({conflicts / max(len(per_genome), 1) * 100:.2f}%)")


# ── D. fold designs ───────────────────────────────────────────────────────────
def section_d() -> None:
    header("D. FOLD DESIGNS — sizes, resistance rates, leakage")
    for ab in C.ANTIBIOTICS:
        f = read_folds(ab)
        print(f"\n{ab}  n={len(f)}")
        for d in DESIGNS:
            sub = f[f[d] >= 0]
            g = sub.groupby(d)["label"].agg(["size", "mean"])
            sizes = "/".join(str(int(v)) for v in g["size"])
            rates = "/".join(f"{v * 100:.0f}%" for v in g["mean"])
            print(f"  {d:13} folds={len(g):>2}  sizes {sizes}  R {rates}")

        typed = f[f["sequence_type"].notna()]
        leak = int((typed.groupby("sequence_type")["fold_mlst"].nunique() > 1).sum())
        print(f"  leakage check: STs present in more than one MLST fold = {leak} (must be 0)")

    f = read_folds(C.PRIMARY_ANTIBIOTIC)
    countries = (f[f["fold_country"] >= 0].groupby("isolation_country")["label"]
                 .agg(n="size", pctR="mean").sort_values("n", ascending=False))
    countries["pctR"] = (countries["pctR"] * 100).round(1)
    print(f"\n{C.PRIMARY_ANTIBIOTIC}: countries held out in the geographic design")
    print(countries.to_string())
    print(f"genomes only ever in training (fold -1): {int((f['fold_country'] < 0).sum())}")

    rates = f.groupby("fold_mlst")["label"].mean()
    top_fold = int(rates.idxmax())
    in_fold = f[f["fold_mlst"] == top_fold]
    top = (in_fold.groupby("sequence_type")["label"].agg(n="size", pctR="mean")
           .sort_values("n", ascending=False).head(5))
    top["pctR"] = (top["pctR"] * 100).round(1)
    print(f"\nMLST fold with the highest resistance rate: fold {top_fold} "
          f"({rates.max() * 100:.0f}% R, n={len(in_fold)}); largest STs in it:")
    print(top.to_string())


# ── E. clustering validation ──────────────────────────────────────────────────
def section_e() -> None:
    header("E. K-MEANS CLUSTER VALIDATION")
    cv = pd.read_csv(C.TABLES / "clustering_validation.csv")
    first = cv.iloc[0]
    print(f"ARI {first['ari_vs_mlst']:.3f} | AMI {first['ami_vs_mlst']:.3f} | "
          f"major-ST concentration {first['major_st_concentration']:.3f} | "
          f"multi-genome STs split {first['st_split_rate'] * 100:.1f}%")
    print(cv[["cluster", "n_genomes", "mean_length_mb", "n_sequence_types",
              "dominant_st", "dominant_st_share"]].to_string(index=False))
    st = pd.read_csv(C.TABLES / "major_st_concentration.csv")
    print(f"\nSTs with n>=50: {len(st)} | concentration range "
          f"{st['concentration'].min():.2f}-{st['concentration'].max():.2f} "
          f"| chance level {1 / C.N_CLUSTERS:.2f}")
    print(st.head(12).to_string(index=False))


# ── F. QRDR coverage and the unresolved-site question ─────────────────────────
def section_f() -> None:
    header("F. QRDR — coverage, residues, and whether 'not found' predicts resistance")
    res = pd.read_csv(C.PROCESSED / "qrdr_residues.csv", dtype={"genome_id": str})
    print(f"all genomes in matrix: {len(res):,}")
    for s in SITES:
        col = res[s]
        print(f"\n{s}: located {col.notna().mean() * 100:.1f}%")
        print("  " + ", ".join(f"{k} {v / len(res) * 100:.1f}%"
                               for k, v in col.value_counts().head(8).items()))

    f = read_folds(C.PRIMARY_ANTIBIOTIC).merge(res, on="genome_id", how="left")

    def category(site: str, value) -> str:
        if pd.isna(value):
            return "not found"
        if value == WILD_TYPE[site]:
            return "wild type"
        if value in RESISTANT[site]:
            return "resistance allele"
        return "other residue"

    print(f"\n{C.PRIMARY_ANTIBIOTIC} cohort (n={len(f)}): resistance rate by site status")
    for s in SITES:
        cat = f[s].apply(lambda v, s=s: category(s, v))
        tab = f.groupby(cat)["label"].agg(n="size", pctR="mean")
        tab["pctR"] = (tab["pctR"] * 100).round(1)
        print(f"\n{s}\n{tab.to_string()}")

    unresolved_any = f[SITES].isna().any(axis=1)
    all_zero = f[SITES].apply(lambda r: all(category(s, r[s]) in ("not found", "other residue")
                                            for s in SITES), axis=1)
    print(f"\ngenomes with at least one site not found: {int(unresolved_any.sum())} "
          f"({f.loc[unresolved_any, 'label'].mean() * 100:.1f}% R)")
    print(f"genomes whose 8 QRDR features are all zero: {int(all_zero.sum())}"
          + (f" ({f.loc[all_zero, 'label'].mean() * 100:.1f}% R)" if all_zero.any() else ""))


# ── G. classical model results ────────────────────────────────────────────────
def section_g() -> None:
    header("G. CLASSICAL MODEL RESULTS (model_comparison.csv)")
    r = pd.read_csv(C.TABLES / "model_comparison.csv")
    print(f"configurations: {len(r)} (expected 192)")

    for ab in C.ANTIBIOTICS:
        sub = r[r["antibiotic"] == ab]
        idx = sub.groupby(["feature_set", "cv_design"])["auroc_mean"].idxmax()
        best = sub.loc[idx]
        grid = best.pivot(index="feature_set", columns="cv_design", values="auroc_mean")
        winner = best.pivot(index="feature_set", columns="cv_design", values="model")
        cols = [d for d in DESIGNS if d in grid.columns]
        print(f"\n{ab}: best AUROC per cell")
        print(grid.reindex(FEATURE_SETS)[cols].round(3).to_string())
        print("winning model per cell:")
        print(winner.reindex(FEATURE_SETS)[cols].to_string())
        gap = (grid["fold_random"] - grid["fold_mlst"]).reindex(FEATURE_SETS).round(3)
        print("lineage gap (random - MLST): " + ", ".join(f"{k} {v:+.3f}" for k, v in gap.items()))

    strict = r[r["cv_design"] == "fold_mlst"]
    best = strict.loc[strict.groupby("antibiotic")["auroc_mean"].idxmax()]
    print("\nbest configuration per antibiotic (MLST-grouped):")
    print(best[["antibiotic", "feature_set", "model", "auroc_mean", "auroc_std",
                "sensitivity_mean", "specificity_mean"]].round(3).to_string(index=False))

    q = r[(r["feature_set"] == "qrdr") & (r["model"] == "XGBoost")
          & (r["cv_design"].isin(["fold_mlst", "fold_country"]))]
    print("\nQRDR + XGBoost specifically (the deployed model type):")
    print(q[["antibiotic", "cv_design", "auroc_mean", "auroc_std",
             "sensitivity_mean", "specificity_mean"]].round(3).to_string(index=False))

    all_q = r[(r["feature_set"] == "qrdr") & (r["cv_design"] == "fold_mlst")]
    print("\nQRDR, all three models, MLST-grouped:")
    print(all_q[["antibiotic", "model", "auroc_mean", "auroc_std"]]
          .round(3).to_string(index=False))


# ── H. CNN ────────────────────────────────────────────────────────────────────
def section_h() -> None:
    header("H. CNN RESULTS")
    cnn = pd.read_csv(C.TABLES / "cnn_results.csv")
    for arm, g in cnn.groupby("arm"):
        print(f"{arm:9} AUROC {g['auroc'].mean():.3f} ± {g['auroc'].std():.3f} "
              f"| per fold {', '.join(f'{v:.3f}' for v in g['auroc'])}")
    print(f"config: window {C.CNN_WINDOW_BP} bp x {C.CNN_WINDOWS_PER_GENOME}, target flank "
          f"{C.CNN_TARGET_FLANK_BP} bp, epochs {C.CNN_EPOCHS_TARGETED}/{C.CNN_EPOCHS_RANDOM}, "
          f"batch {C.CNN_BATCH_SIZE}, lr {C.CNN_LEARNING_RATE}, embed {C.CNN_EMBED_DIM}, "
          f"filters {C.CNN_FILTERS}")


# ── I. SHAP outputs as saved by s08 ───────────────────────────────────────────
def section_i() -> None:
    header("I. SHAP OUTPUTS SAVED BY s08")
    ctrl = pd.read_csv(C.TABLES / "shap_qrdr_control.csv")
    print("positive control (QRDR model):")
    print(ctrl.round(4).to_string(index=False))

    top = pd.read_csv(C.TABLES / "shap_top_hexamers.csv")
    top["palindromic"] = top["hexamer"].apply(lambda h: revcomp(h) == h)
    in_top = set(top["hexamer"])
    top["revcomp_in_top20"] = top["hexamer"].apply(
        lambda h: revcomp(h) != h and revcomp(h) in in_top)
    cols = [c for c in ["rank", "hexamer", "palindromic", "revcomp_in_top20", "mean_abs_shap",
                        "lineage_eta_squared", "genotype_eta_squared", "enrichment_vs_genome"]
            if c in top.columns]
    print("\ntop 20 hexamers (single-strand, as in the thesis draft):")
    print(top[cols].round(4).to_string(index=False))
    print(f"\npalindromic: {int(top['palindromic'].sum())}/20"
          f" | reverse-complement pairs inside top 20: {int(top['revcomp_in_top20'].sum()) // 2}")
    for c in ["lineage_eta_squared", "genotype_eta_squared"]:
        if c in top:
            print(f"mean {c}: {top[c].mean():.3f}")
    if "enrichment_vs_genome" in top:
        print(f"enriched in gyrA/parC regions (ratio > 1): {int((top['enrichment_vs_genome'] > 1).sum())}/20")


# ── J. canonical SHAP re-ranking and corrected eta² (needs the matrix) ────────
def section_j() -> None:
    header("J. RECOMPUTED — canonical hexamer ranking and corrected variance measures")
    import shap
    import s08_shap_validation as S8

    folds = read_folds(C.PRIMARY_ANTIBIOTIC)
    ids = np.load(C.PROCESSED / "kmer_genome_ids.npy", allow_pickle=True)
    id_to_row = {g: i for i, g in enumerate(ids)}
    rows = np.array([id_to_row[g] for g in folds["genome_id"]])
    X = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")[rows]
    vocab = [str(v) for v in np.load(C.PROCESSED / "kmer_vocabulary.npy", allow_pickle=True)]
    y = folds["label"].to_numpy()
    index = {h: i for i, h in enumerate(vocab)}

    model = S8.fit_xgboost(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)

    # canonical groups: each hexamer merged with its reverse complement
    groups, seen = [], set()
    for h in vocab:
        if h in seen:
            continue
        rc = revcomp(h)
        members = [index[h]] if rc == h else [index[h], index[rc]]
        seen.update([h, rc])
        groups.append((min(h, rc), rc == h, members))
    n_pal = sum(1 for g in groups if g[1])
    print(f"canonical hexamers: {len(groups)} ({n_pal} palindromic, base rate "
          f"{n_pal / len(groups) * 100:.2f}%)")

    importance = np.array([np.abs(sv[:, m].sum(axis=1)).mean() for _, _, m in groups])
    order = np.argsort(importance)[::-1][:20]

    # grouping codes for variance decomposition
    st = folds["sequence_type"]
    typed = st.notna().to_numpy()
    st_codes = pd.factorize(st[typed])[0]
    res = pd.read_csv(C.PROCESSED / "qrdr_residues.csv", dtype={"genome_id": str})
    res = folds[["genome_id"]].merge(res, on="genome_id", how="left")
    genotype = ((res["gyrA_83"].isin(RESISTANT["gyrA_83"]))
                | (res["parC_80"].isin(RESISTANT["parC_80"]))).astype(int).to_numpy()

    rng = np.random.default_rng(C.SEED)
    n_perm = 200
    records = []
    for rank, gi in enumerate(order, start=1):
        name, is_pal, members = groups[gi]
        freq = X[:, members].sum(axis=1).astype(float)
        eta_l, eps_l, k_l = eta_squared(freq[typed], st_codes)
        eta_g, eps_g, _ = eta_squared(freq, genotype)
        null = [eta_squared(freq[typed], rng.permutation(st_codes))[0] for _ in range(n_perm)]
        records.append({"rank": rank, "canonical": name, "palindromic": is_pal,
                        "mean_abs_shap": importance[gi],
                        "lineage_eta2": eta_l, "lineage_eps2": eps_l,
                        "lineage_null_mean": np.mean(null),
                        "lineage_null_p95": np.percentile(null, 95),
                        "genotype_eta2": eta_g, "genotype_eps2": eps_g})
    out = pd.DataFrame(records)
    print(f"\nlineage groups used: {k_l} STs (untyped genomes excluded, n={int(typed.sum())})")
    print(out.round(4).to_string(index=False))

    k = int(out["palindromic"].sum())
    p_can = binomial_tail(k, 20, n_pal / len(groups))
    print(f"\ncanonical top 20: {k} palindromic | binomial P(X>={k}) at "
          f"{n_pal / len(groups) * 100:.2f}% = {p_can:.2e}")
    print(f"means: lineage eta² {out['lineage_eta2'].mean():.3f}, epsilon² "
          f"{out['lineage_eps2'].mean():.3f}, permutation null {out['lineage_null_mean'].mean():.3f} "
          f"| genotype eta² {out['genotype_eta2'].mean():.3f}, epsilon² {out['genotype_eps2'].mean():.3f}")
    out.to_csv(C.TABLES / "shap_top_hexamers_canonical.csv", index=False)
    print(f"wrote {C.TABLES / 'shap_top_hexamers_canonical.csv'}")


# ── K. deployed models ────────────────────────────────────────────────────────
def section_k() -> None:
    header("K. DEPLOYED MODELS")
    import joblib
    qrdr = joblib.load(C.MODELS / f"qrdr_{C.PRIMARY_ANTIBIOTIC}.joblib")
    print(f"qrdr model: {type(qrdr).__name__}  colsample_bytree="
          f"{qrdr.get_params().get('colsample_bytree')} (s06 evaluated 0.2)")
    kmer = joblib.load(C.MODELS / f"kmer_{C.PRIMARY_ANTIBIOTIC}.joblib")
    print(f"kmer model: {type(kmer).__name__} -> {[s[0] for s in kmer.steps]}")

    enc = {"WT": [0, 1], "R": [1, 0], "?": [0, 0]}
    print("\nprobability of resistance for selected inputs (? = site not found):")
    for combo in [("WT",) * 4, ("?",) * 4, ("R", "WT", "WT", "WT"), ("R", "R", "R", "WT"),
                  ("R", "?", "WT", "WT"), ("?", "?", "WT", "WT"), ("WT", "WT", "?", "?")]:
        v = np.array(sum((enc[c] for c in combo), []), dtype=np.float32).reshape(1, -1)
        p = float(qrdr.predict_proba(v)[0, 1])
        print(f"  {dict(zip(SITES, combo))}  P(R)={p:.3f}")


# ── L. environment ────────────────────────────────────────────────────────────
def section_l() -> None:
    header("L. ENVIRONMENT")
    print(f"python {platform.python_version()} | {platform.platform()}")
    for name in ["numpy", "pandas", "sklearn", "xgboost", "torch", "shap",
                 "streamlit", "matplotlib", "requests", "joblib"]:
        try:
            module = __import__(name)
            print(f"  {name:11} {getattr(module, '__version__', '?')}")
        except Exception:                                  # noqa: BLE001
            print(f"  {name:11} not installed")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--conflicts", action="store_true")
    args = ap.parse_args()

    sections = [("A", section_a), ("B", section_b),
                ("C", lambda: section_c(args.conflicts)), ("D", section_d),
                ("E", section_e), ("F", section_f), ("G", section_g),
                ("H", section_h), ("I", section_i)]
    if not args.skip_shap:
        sections.append(("J", section_j))
    sections += [("K", section_k), ("L", section_l)]

    buffer = io.StringIO()
    for name, fn in sections:
        print(f"running section {name} ...", file=sys.stderr)
        with redirect_stdout(buffer):
            try:
                fn()
            except Exception:                              # noqa: BLE001
                print(f"\n[section {name} FAILED]\n{traceback.format_exc(limit=3)}")

    report = buffer.getvalue()
    path = C.RESULTS / "full_check_report.txt"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport written to {path}  ({len(report) / 1024:.1f} KB) — send this file back.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
