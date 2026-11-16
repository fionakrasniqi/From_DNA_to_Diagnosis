"""
s13_tune_kmer.py — does the k-mer model lose only because it was not tuned?

Stage 06 used fixed hyperparameters. A fair objection is that the k-mer models
might do better on new lineages if they were tuned. This stage answers that
with nested cross-validation for ciprofloxacin:

  outer loop   the same five folds as stage 06 (random and MLST-grouped)
  inner loop   three folds inside the training part only, used to pick the
               hyperparameters; for the MLST design the inner folds are also
               grouped by sequence type, so tuning never sees a test lineage

The test fold is never used to choose settings, so the reported AUROC is not
optimistic. Two models are tuned: logistic regression (the best untuned k-mer
model) and XGBoost.

Resumable: each finished outer fold is saved immediately.

Run from src/:
    python s13_tune_kmer.py            # about 2-3 hours on a laptop CPU

Output:
    results/tables/kmer_tuning_folds.csv   one row per outer fold
    results/tables/kmer_tuning.csv         summary: tuned vs untuned AUROC
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config as C

warnings.filterwarnings("ignore")
log = logging.getLogger("s13")

DESIGNS = ["fold_random", "fold_mlst"]
INNER_FOLDS = 3

GRIDS = {
    "LogisticRegression": [{"C": c} for c in [0.001, 0.01, 0.1, 1.0]],
    "XGBoost": [{"max_depth": d, "learning_rate": lr}
                for d, lr in itertools.product([3, 6], [0.05, 0.15])],
}


def make_model(name: str, params: dict, pos_weight: float):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced",
                                                solver="lbfgs", **params))
    return XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.2, max_bin=64,
                         scale_pos_weight=pos_weight, eval_metric="logloss",
                         tree_method="hist", n_jobs=-1, random_state=C.SEED, **params)


def inner_splits(design: str, y: np.ndarray, groups: np.ndarray):
    if design == "fold_mlst":
        return list(GroupKFold(n_splits=INNER_FOLDS).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                random_state=C.SEED).split(np.zeros(len(y)), y))


def pick_params(name: str, X, y, groups, design: str) -> tuple[dict, float]:
    """Choose hyperparameters using the training part only."""
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    splits = inner_splits(design, y, groups)
    best, best_score = None, -1.0
    for params in GRIDS[name]:
        scores = []
        for tr, va in splits:
            if len(np.unique(y[va])) < 2:
                continue
            model = make_model(name, params, pos_weight).fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]))
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid, for testing only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.quick:
        GRIDS["LogisticRegression"] = GRIDS["LogisticRegression"][-2:]
        GRIDS["XGBoost"] = GRIDS["XGBoost"][:2]

    folds = pd.read_csv(C.PROCESSED / f"folds_{C.PRIMARY_ANTIBIOTIC}.csv",
                        dtype={"genome_id": str, "sequence_type": str})
    ids = np.load(C.PROCESSED / "kmer_genome_ids.npy", allow_pickle=True)
    row = {g: i for i, g in enumerate(ids)}
    X = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")[[row[g] for g in folds["genome_id"]]]
    y = folds["label"].to_numpy()
    st = folds["sequence_type"].fillna("").to_numpy().astype(object)
    untyped = st == ""
    st[untyped] = [f"__untyped_{i}" for i in range(int(untyped.sum()))]
    log.info("ciprofloxacin: %d genomes, %d k-mer features", len(y), X.shape[1])

    out_path = C.TABLES / "kmer_tuning_folds.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    finished = {(r["design"], r["model"], r["fold"]) for r in rows}
    if finished:
        log.info("resuming: %d outer folds already done", len(finished))

    started = time.time()
    for design in DESIGNS:
        outer = folds[design].to_numpy()
        for name in GRIDS:
            for fold in sorted(f for f in np.unique(outer) if f >= 0):
                if (design, name, int(fold)) in finished:
                    continue
                test = outer == fold
                train = ~test
                params, inner_auc = pick_params(name, X[train], y[train], st[train], design)
                pos_weight = float((y[train] == 0).sum() / max((y[train] == 1).sum(), 1))
                model = make_model(name, params, pos_weight).fit(X[train], y[train])
                auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
                rows.append({"design": design, "model": name, "fold": int(fold),
                             "params": str(params), "inner_auroc": round(inner_auc, 4),
                             "test_auroc": round(float(auc), 4)})
                pd.DataFrame(rows).to_csv(out_path, index=False)
                log.info("%-12s %-18s fold %d  %s  test AUROC %.3f  (%.0f min so far)",
                         design.replace("fold_", ""), name, fold, params, auc,
                         (time.time() - started) / 60)

    res = pd.DataFrame(rows)
    tuned = (res.groupby(["model", "design"])["test_auroc"].agg(["mean", "std"])
             .reset_index().rename(columns={"mean": "tuned_auroc", "std": "tuned_std"}))
    comp_path = C.TABLES / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp = comp[(comp["antibiotic"] == C.PRIMARY_ANTIBIOTIC) & (comp["feature_set"] == "kmer")]
        tuned = tuned.merge(comp[["model", "cv_design", "auroc_mean"]]
                            .rename(columns={"cv_design": "design", "auroc_mean": "untuned_auroc"}),
                            on=["model", "design"], how="left")
    tuned.round(3).to_csv(C.TABLES / "kmer_tuning.csv", index=False)

    print("\n" + "=" * 70)
    print("K-MER MODELS, CIPROFLOXACIN: TUNED (NESTED CV) VS UNTUNED")
    print("=" * 70)
    print(tuned.round(3).to_string(index=False))
    for name in GRIDS:
        t = tuned[tuned["model"] == name].set_index("design")
        if {"fold_random", "fold_mlst"} <= set(t.index):
            gap = t.loc["fold_random", "tuned_auroc"] - t.loc["fold_mlst", "tuned_auroc"]
            line = f"\n{name}: drop random -> MLST after tuning = {gap:.3f}"
            if "untuned_auroc" in t:
                ugap = t.loc["fold_random", "untuned_auroc"] - t.loc["fold_mlst", "untuned_auroc"]
                line += f" (untuned: {ugap:.3f})"
            print(line)
    print(f"\nwrote {C.TABLES / 'kmer_tuning.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
