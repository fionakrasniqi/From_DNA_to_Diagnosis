"""
sanity_check.py — quick signal check on the k-mer matrix.

Runs before any real modelling to answer three questions cheaply:

  1. Do labels join to the k-mer matrix cleanly for every antibiotic?
  2. Is there a resistance signal in the hexamer frequencies at all?
  3. Is that signal driven by real biology, or largely by lineage / country?

The third question is the point of the whole thesis. If a stratified split
and a leave-one-country-out split give roughly the same AUROC, resistance
is encoded in the sequence in a way that transfers. If leave-one-country-out
collapses, the model is memorising where the isolate came from.

Not a modelling result — a smoke test. Real evaluation lives in stage 07.
Runs in about two minutes.

NOTE ON PREPROCESSING: k-mer frequencies live at ~1/4096, which is small
enough that unscaled logistic regression collapses to constant predictions
(AUROC 0.500 with zero variance). StandardScaler on the training fold fixes
it. This was learned the hard way — the first version of this script had no
scaler and gave 0.500 on every drug.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C


def fit_and_score(X_train, y_train, X_test, y_test) -> float:
    """Standardise then fit — see the module docstring for why the scaler is required."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                           solver="lbfgs"),
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# ── load ──────────────────────────────────────────────────────────────────────
print("=" * 62)
print("SANITY CHECK — hexamer signal on the QC'd matrix")
print("=" * 62)

X = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")
genome_ids = np.load(C.PROCESSED / "kmer_genome_ids.npy")
phenotypes = pd.read_csv(C.PHENOTYPES / "phenotypes_long.csv",
                         dtype={"genome_id": str})
metadata = pd.read_csv(C.PROCESSED / "genome_metadata.csv",
                       dtype={"genome_id": str})

print(f"\nmatrix    : {X.shape}  (frequencies)")
print(f"genomes   : {len(genome_ids)}")

id_to_row = {gid: i for i, gid in enumerate(genome_ids)}

# ── per-antibiotic baseline ───────────────────────────────────────────────────
for antibiotic in C.ANTIBIOTICS:
    sub = phenotypes[phenotypes["antibiotic"] == antibiotic]
    sub = sub[sub["genome_id"].isin(id_to_row)]
    if len(sub) < 100:
        print(f"\n{antibiotic:15}  too few genomes ({len(sub)}) — skipping")
        continue

    rows = np.array([id_to_row[g] for g in sub["genome_id"]])
    Xa, ya = X[rows], sub["label"].to_numpy()

    print(f"\n{antibiotic:15}  n={len(ya):>5}  R={int(ya.sum()):>5}  "
          f"S={int((1 - ya).sum()):>5}  ({ya.mean() * 100:.1f}% R)")

    aucs = []
    t0 = time.time()
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=C.SEED).split(Xa, ya):
        aucs.append(fit_and_score(Xa[tr], ya[tr], Xa[te], ya[te]))
    print(f"  random 5-fold AUROC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
          f"({time.time() - t0:.1f}s)")

# ── the interesting one: does the signal transfer across countries? ──────────
print("\n" + "-" * 62)
print("Geographic transfer test — ciprofloxacin")
print("-" * 62)

cipro = phenotypes[phenotypes["antibiotic"] == C.PRIMARY_ANTIBIOTIC]
cipro = cipro[cipro["genome_id"].isin(id_to_row)]
countries = metadata[["genome_id", "isolation_country"]]
merged = cipro.merge(countries, on="genome_id")

rows = np.array([id_to_row[g] for g in merged["genome_id"]])
Xc, yc = X[rows], merged["label"].to_numpy()
country = merged["isolation_country"].fillna("Unknown").to_numpy()

for held_out in ["Norway", "United States", "United Kingdom"]:
    train_mask = country != held_out
    test_mask = country == held_out
    if test_mask.sum() < 30 or len(set(yc[test_mask])) < 2:
        print(f"  hold out {held_out:>15}  n_test={test_mask.sum():>4}  skipped "
              f"(too few or single-class)")
        continue

    auc = fit_and_score(Xc[train_mask], yc[train_mask],
                        Xc[test_mask], yc[test_mask])
    print(f"  hold out {held_out:>15}  n_test={test_mask.sum():>4}  "
          f"R%={yc[test_mask].mean() * 100:>4.1f}  AUROC={auc:.3f}")

print("\n" + "=" * 62)
print("DONE")
print("=" * 62)
print("If random-split AUROC is well above ~0.6, the signal is real.")
print("If leave-one-country-out AUROC drops sharply, the model is")
print("leaning on geographic patterns — which is exactly what the thesis")
print("is designed to expose. Both outcomes make the study worth writing.")

