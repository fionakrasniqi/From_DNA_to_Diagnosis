"""
s06_train_classical.py — classical model comparison (proposal §2.3, §2.4).

Evaluates the full factorial the study is built around:

    3 models        Logistic Regression, Random Forest, XGBoost
  x 4 feature sets  k-mer, CARD, QRDR, combined
  x 4 CV designs    random, k-means-grouped, MLST-grouped, leave-one-country-out
  x 4 antibiotics   ciprofloxacin, ampicillin, tetracycline, gentamicin

The comparison across CV designs is the point. Random splits place close
relatives on both sides of the fold boundary, so a model can score well by
recognising a lineage rather than by detecting resistance. Stage 05 showed the
k-means grouping from §2.4 fails to separate lineages (fold resistance rates
were indistinguishable from random), while MLST grouping does. The gap between
random and MLST-grouped performance is therefore an estimate of how much of the
apparent accuracy is lineage memorisation.

Feature sets serve the same question from the other side. QRDR is eight columns
of known causal biology; k-mer is 4,096 columns of whole-genome composition. If
k-mer only matches QRDR under random splits and falls behind under MLST
grouping, the extra columns were encoding population structure.

Preprocessing note: k-mer frequencies sit at ~1/4096. Unscaled, logistic
regression collapses to constant predictions and returns AUROC 0.500 exactly.
Scaling is fitted inside each fold, never on the full dataset.

Output:
    results/tables/model_comparison.csv    one row per configuration
    results/tables/per_fold_results.csv    per-fold detail for error bars

Deployable models are fitted separately in s10_build_app.py, which imports
build_models() from here so the deployed hyperparameters match these exactly.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config as C

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("s06")

CV_DESIGNS = ["fold_random", "fold_grouped", "fold_mlst", "fold_country"]
FEATURE_SETS = ["kmer", "card", "qrdr", "combined"]


def build_models(pos_weight: float) -> dict:
    """Model zoo. Class weighting is set per-fold from the training split."""
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0,
                               class_weight="balanced", solver="lbfgs"),
        ),
        # Tree hyperparameters are tuned for throughput as well as accuracy.
        # With 4,096 highly correlated hexamer columns, evaluating most of them
        # at every split is nearly all cost and almost no benefit: reducing
        # colsample_bytree from 0.8 to 0.2 and max_bin from 256 to 64 made
        # XGBoost roughly nine times faster with no measurable AUROC loss on a
        # planted-signal benchmark (0.791 -> 0.785, inside fold noise). At the
        # original settings the full factorial was projected at ~35 hours.
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=3,
            max_features=64, class_weight="balanced_subsample",
            n_jobs=-1, random_state=C.SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.15,
            subsample=0.8, colsample_bytree=0.2, max_bin=64,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=C.SEED,
        ),
    }


def load_features(genome_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Load every feature matrix, all aligned to the k-mer matrix row order."""
    kmer = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")
    card = np.load(C.PROCESSED / "X_card.npy").astype(np.float32)
    qrdr = np.load(C.PROCESSED / "X_qrdr.npy").astype(np.float32)

    mech_ids = np.load(C.PROCESSED / "mechanism_genome_ids.npy")
    if not np.array_equal(mech_ids, genome_ids):
        raise RuntimeError("mechanism features are not aligned to the k-mer "
                           "matrix — re-run s04 after any change to s03")

    return {
        "kmer": kmer,
        "card": card,
        "qrdr": qrdr,
        "combined": np.hstack([kmer, card, qrdr]),
    }


def evaluate_fold(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on one training split and score on its held-out split."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y_test, proba),
        "auprc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "n_test": len(y_test),
    }


def run_configuration(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                      model_name: str, pos_weight: float) -> list[dict]:
    """Evaluate one model over every fold of one CV design."""
    results = []
    for fold in sorted(f for f in np.unique(folds) if f >= 0):
        test_mask = folds == fold
        # Under leave-one-country-out, genomes in countries too small to hold
        # out are marked -1. They still belong in training — excluding them
        # would shrink the training set for no reason.
        train_mask = ~test_mask

        y_train, y_test = y[train_mask], y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue  # a single-class split cannot produce an AUROC

        model = build_models(pos_weight)[model_name]
        try:
            scores = evaluate_fold(model, X[train_mask], y_train,
                                   X[test_mask], y_test)
        except Exception as exc:                       # noqa: BLE001
            log.warning("fold %s failed for %s: %s", fold, model_name, exc)
            continue

        scores["fold"] = int(fold)
        results.append(scores)
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    genome_ids = np.load(C.PROCESSED / "kmer_genome_ids.npy")
    features = load_features(genome_ids)
    id_to_row = {gid: i for i, gid in enumerate(genome_ids)}

    for name, matrix in features.items():
        log.info("feature set %-9s %s", name, matrix.shape)

    # Resume support. The full factorial takes hours; a crash or a closed laptop
    # should not cost the whole run. Completed configurations are reloaded and
    # skipped rather than recomputed.
    checkpoint = C.TABLES / "model_comparison.csv"
    fold_checkpoint = C.TABLES / "per_fold_results.csv"
    summary_rows, fold_rows, done = [], [], set()

    if checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        summary_rows = previous.to_dict("records")
        done = set(zip(previous["antibiotic"], previous["feature_set"],
                       previous["cv_design"], previous["model"]))
        log.info("resuming: %d configurations already complete", len(done))
        if fold_checkpoint.exists():
            fold_rows = pd.read_csv(fold_checkpoint).to_dict("records")

    started = time.time()

    for antibiotic in C.ANTIBIOTICS:
        fold_path = C.PROCESSED / f"folds_{antibiotic}.csv"
        if not fold_path.exists():
            log.warning("%s: no fold file, skipping", antibiotic)
            continue

        fold_table = pd.read_csv(fold_path, dtype={"genome_id": str})
        rows = np.array([id_to_row[g] for g in fold_table["genome_id"]])
        y = fold_table["label"].to_numpy()
        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

        log.info("=== %s: n=%d, %.1f%% resistant ===",
                 antibiotic, len(y), y.mean() * 100)

        for feature_set in FEATURE_SETS:
            X = features[feature_set][rows]

            for design in CV_DESIGNS:
                folds = fold_table[design].to_numpy()
                if (folds >= 0).sum() < 50:
                    continue

                for model_name in build_models(pos_weight):
                    if (antibiotic, feature_set, design, model_name) in done:
                        continue
                    t0 = time.time()
                    per_fold = run_configuration(X, y, folds,
                                                 model_name, pos_weight)
                    if not per_fold:
                        continue

                    for record in per_fold:
                        fold_rows.append({
                            "antibiotic": antibiotic, "feature_set": feature_set,
                            "cv_design": design, "model": model_name, **record,
                        })

                    aurocs = [r["auroc"] for r in per_fold]
                    summary_rows.append({
                        "antibiotic": antibiotic,
                        "feature_set": feature_set,
                        "cv_design": design,
                        "model": model_name,
                        "n_folds": len(per_fold),
                        "auroc_mean": np.mean(aurocs),
                        "auroc_std": np.std(aurocs),
                        "auprc_mean": np.mean([r["auprc"] for r in per_fold]),
                        "f1_mean": np.mean([r["f1"] for r in per_fold]),
                        "sensitivity_mean": np.nanmean([r["sensitivity"] for r in per_fold]),
                        "specificity_mean": np.nanmean([r["specificity"] for r in per_fold]),
                        "seconds": time.time() - t0,
                    })

                    log.info("  %-9s %-13s %-18s AUROC %.3f +/- %.3f  (%.0fs)",
                             feature_set, design.replace("fold_", ""),
                             model_name, np.mean(aurocs), np.std(aurocs),
                             time.time() - t0)

                    # flush after every configuration so progress survives a crash
                    pd.DataFrame(summary_rows).to_csv(checkpoint, index=False)
                    pd.DataFrame(fold_rows).to_csv(fold_checkpoint, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(C.TABLES / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(C.TABLES / "per_fold_results.csv", index=False)

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("MODEL COMPARISON")
    print("=" * 76)
    print(f"configurations evaluated : {len(summary)}")
    print(f"total runtime            : {(time.time() - started) / 60:.1f} min")

    cipro = summary[summary["antibiotic"] == C.PRIMARY_ANTIBIOTIC]
    if not cipro.empty:
        print(f"\n--- {C.PRIMARY_ANTIBIOTIC}: AUROC by feature set and CV design ---")
        pivot = cipro.pivot_table(index=["feature_set", "model"],
                                  columns="cv_design", values="auroc_mean")
        order = [d for d in CV_DESIGNS if d in pivot.columns]
        print(pivot[order].to_string(float_format=lambda v: f"{v:.3f}"))

        print("\n--- the headline comparison ---")
        for feature_set in FEATURE_SETS:
            sub = cipro[cipro["feature_set"] == feature_set]
            rand = sub[sub["cv_design"] == "fold_random"]["auroc_mean"].max()
            mlst = sub[sub["cv_design"] == "fold_mlst"]["auroc_mean"].max()
            if np.isnan(rand) or np.isnan(mlst):
                continue
            print(f"  {feature_set:9}  random {rand:.3f}  ->  MLST-grouped "
                  f"{mlst:.3f}   drop {rand - mlst:+.3f}")

        print("\nINTERPRETATION")
        print("  A large drop from random to MLST-grouped means the model was")
        print("  scoring on lineage rather than on resistance mechanism.")
        print("  QRDR is eight columns of known causal biology, so it should")
        print("  hold up best. If k-mer drops toward QRDR under MLST grouping,")
        print("  its extra 4,088 columns were encoding population structure.")

    print("\n--- best configuration per antibiotic (MLST-grouped) ---")
    strict = summary[summary["cv_design"] == "fold_mlst"]
    if not strict.empty:
        best = strict.loc[strict.groupby("antibiotic")["auroc_mean"].idxmax()]
        print(best[["antibiotic", "feature_set", "model",
                    "auroc_mean", "auroc_std", "sensitivity_mean",
                    "specificity_mean"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nwrote {C.TABLES / 'model_comparison.csv'}")
    print(f"wrote {C.TABLES / 'per_fold_results.csv'}")
    print("\nnext: python src/s07_train_cnn.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
