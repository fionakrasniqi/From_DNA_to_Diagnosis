"""
inspect_results.py — read stage 06 output and produce the tables for the write-up.

Three things:

  1. AUROC by feature set and CV design, for every antibiotic (stage 06 only
     printed ciprofloxacin).

  2. The random-to-MLST-grouped gap per feature set per drug. This is the
     lineage-memorisation estimate: how much apparent accuracy disappears once
     relatives are kept out of the test fold.

  3. A drug-specificity control on the QRDR features. gyrA and parC alter the
     fluoroquinolone target, so they are mechanistically relevant to
     ciprofloxacin and to nothing else here. If they predict gentamicin (an
     aminoglycoside) or tetracycline well, that is co-resistance — the same
     lineages happening to carry both traits — not mechanism. The control
     matters because it decides how strongly the QRDR result can be worded.

Read-only. Runs in seconds.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config as C

CV_ORDER = ["fold_random", "fold_grouped", "fold_mlst", "fold_country"]
FMT = lambda v: f"{v:.3f}"  # noqa: E731


def main() -> int:
    path = C.TABLES / "model_comparison.csv"
    if not path.exists():
        print(f"missing {path} — run s06_train_classical.py first")
        return 1

    results = pd.read_csv(path)
    print("=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    print(f"configurations: {len(results)}")

    # ── 1. per-antibiotic pivots ─────────────────────────────────────────────
    for antibiotic in C.ANTIBIOTICS:
        sub = results[results["antibiotic"] == antibiotic]
        if sub.empty:
            continue
        print(f"\n--- {antibiotic}: AUROC ---")
        pivot = sub.pivot_table(index=["feature_set", "model"],
                                columns="cv_design", values="auroc_mean")
        cols = [c for c in CV_ORDER if c in pivot.columns]
        print(pivot[cols].to_string(float_format=FMT))

    # ── 2. lineage-memorisation gap ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("LINEAGE-MEMORISATION GAP  (best random AUROC - best MLST-grouped AUROC)")
    print("=" * 78)
    rows = []
    for antibiotic in C.ANTIBIOTICS:
        for feature_set in ["kmer", "card", "qrdr", "combined"]:
            sub = results[(results["antibiotic"] == antibiotic)
                          & (results["feature_set"] == feature_set)]
            if sub.empty:
                continue
            rand = sub[sub["cv_design"] == "fold_random"]["auroc_mean"].max()
            mlst = sub[sub["cv_design"] == "fold_mlst"]["auroc_mean"].max()
            country = sub[sub["cv_design"] == "fold_country"]["auroc_mean"].max()
            rows.append({"antibiotic": antibiotic, "feature_set": feature_set,
                         "random": rand, "mlst_grouped": mlst,
                         "country_holdout": country,
                         "gap_random_minus_mlst": rand - mlst})
    gaps = pd.DataFrame(rows)
    print(gaps.to_string(index=False, float_format=FMT))
    gaps.to_csv(C.TABLES / "lineage_gap.csv", index=False)

    # ── 3. QRDR drug-specificity control ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("QRDR DRUG-SPECIFICITY CONTROL")
    print("=" * 78)
    print("gyrA/parC alter the fluoroquinolone target. Mechanistically they")
    print("should predict ciprofloxacin and NOT the other three drugs.")
    print()
    qrdr = results[(results["feature_set"] == "qrdr")
                   & (results["cv_design"] == "fold_mlst")]
    control = (qrdr.groupby("antibiotic")["auroc_mean"].max()
               .reindex(C.ANTIBIOTICS).dropna())
    for antibiotic, auroc in control.items():
        expected = "expected high" if antibiotic == C.PRIMARY_ANTIBIOTIC \
            else "expected near 0.5"
        print(f"  {antibiotic:15}  AUROC {auroc:.3f}   ({expected})")

    others = control.drop(C.PRIMARY_ANTIBIOTIC, errors="ignore")
    if not others.empty:
        worst = others.max()
        print()
        if worst > 0.75:
            print(f"  WARNING: QRDR reaches {worst:.3f} on a drug it cannot")
            print("  mechanistically affect. That is co-resistance — resistant")
            print("  lineages carrying multiple determinants — so the QRDR")
            print("  features are not purely causal. State this in the")
            print("  discussion and avoid claiming QRDR is lineage-free.")
        elif worst > 0.65:
            print(f"  QRDR reaches {worst:.3f} on non-quinolone drugs: some")
            print("  co-resistance signal, but modest. Note it as a limitation.")
        else:
            print(f"  QRDR stays at {worst:.3f} or below on non-quinolone drugs,")
            print("  which is what mechanistic specificity looks like. This")
            print("  strengthens the causal reading of the ciprofloxacin result.")

    # ── 4. best configuration per antibiotic under strict CV ─────────────────
    print("\n" + "=" * 78)
    print("BEST CONFIGURATION PER ANTIBIOTIC  (MLST-grouped)")
    print("=" * 78)
    strict = results[results["cv_design"] == "fold_mlst"]
    best = strict.loc[strict.groupby("antibiotic")["auroc_mean"].idxmax()]
    print(best[["antibiotic", "feature_set", "model", "auroc_mean", "auroc_std",
                "auprc_mean", "sensitivity_mean", "specificity_mean"]]
          .to_string(index=False, float_format=FMT))
    best.to_csv(C.TABLES / "best_models.csv", index=False)

    print(f"\nwrote {C.TABLES / 'lineage_gap.csv'}")
    print(f"wrote {C.TABLES / 'best_models.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
