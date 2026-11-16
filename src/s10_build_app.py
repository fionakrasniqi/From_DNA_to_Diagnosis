"""
s10_build_app.py — fit and persist the deployable models.

Stage 06 evaluated models under cross-validation; nothing was saved, because a
cross-validated score belongs to a protocol rather than to a single fitted
object. This stage refits on the full cohort and writes the artefacts the
application loads.

The models are built with the exact hyperparameters from stage 06
(build_models), so the deployed object is the one whose cross-validated scores
are reported in the model card, not a variant of it.

Which models are deployable is not a free choice:

  QRDR   computable from a FASTA alone — locate a conserved flank, read the
         codon. No network, no database identifier, milliseconds per genome.
         It also generalised best in stage 06 (0.980 AUROC under MLST-grouped
         folds).

  k-mer  computable from a FASTA alone, but generalised markedly worse (0.875).
         Retained so the application can show both predictions side by side:
         the disagreement between them is the thesis result made concrete on a
         single genome.

  CARD   requires BV-BRC's precomputed annotations, which are keyed by genome
         identifier. That works for genomes already in the database and not for
         a newly sequenced isolate, so it is deliberately excluded from the
         deployed path despite scoring well on the other three antibiotics.

The best-generalising feature set therefore also happens to be the only one
that can run on a novel genome.

Output:
    models/qrdr_<antibiotic>.joblib
    models/kmer_ciprofloxacin.joblib
    models/model_card.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd

import config as C
from s06_train_classical import build_models

log = logging.getLogger("s10")

QRDR_MODEL = "XGBoost"
KMER_MODEL = "LogisticRegression"


def pos_weight_of(y: np.ndarray) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def cv_metrics(reported: pd.DataFrame | None, antibiotic: str,
               feature_set: str, model: str) -> dict:
    """Cross-validated scores of exactly this feature set and model."""
    out: dict = {}
    if reported is None:
        return out
    for design, suffix in [("fold_mlst", "mlst_grouped"),
                           ("fold_country", "country_holdout")]:
        row = reported[(reported["antibiotic"] == antibiotic)
                       & (reported["feature_set"] == feature_set)
                       & (reported["model"] == model)
                       & (reported["cv_design"] == design)]
        if row.empty:
            continue
        row = row.iloc[0]
        out[f"cv_auroc_{suffix}"] = round(float(row["auroc_mean"]), 3)
        out[f"sensitivity_{suffix}"] = round(float(row["sensitivity_mean"]), 3)
        out[f"specificity_{suffix}"] = round(float(row["specificity_mean"]), 3)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    genome_ids = np.load(C.PROCESSED / "kmer_genome_ids.npy")
    id_to_row = {gid: i for i, gid in enumerate(genome_ids)}
    X_kmer_all = np.load(C.PROCESSED / f"X_kmer{C.K}_freq.npy")
    X_qrdr_all = np.load(C.PROCESSED / "X_qrdr.npy").astype(np.float32)
    qrdr_names = np.load(C.PROCESSED / "qrdr_feature_names.npy", allow_pickle=True)

    comparison_path = C.TABLES / "model_comparison.csv"
    reported = pd.read_csv(comparison_path) if comparison_path.exists() else None
    if reported is None:
        log.warning("no model_comparison.csv — the model card will have no scores")

    card: dict = {
        "project": "From DNA to Diagnosis",
        "built": date.today().isoformat(),
        "seed": C.SEED,
        "k": C.K,
        "qrdr_features": [str(n) for n in qrdr_names],
        "deployable_note": (
            "QRDR features are computed from sequence alone. CARD features "
            "require a BV-BRC genome identifier and are therefore excluded "
            "from the deployed path."
        ),
        "decision_rule": (
            "Resistant if predicted probability >= 0.5. If gyrA 83 or parC 80 "
            "cannot be read, the application reports 'cannot determine'."
        ),
        "models": {},
    }

    for antibiotic in C.ANTIBIOTICS:
        fold_path = C.PROCESSED / f"folds_{antibiotic}.csv"
        if not fold_path.exists():
            continue

        folds = pd.read_csv(fold_path, dtype={"genome_id": str})
        rows = np.array([id_to_row[g] for g in folds["genome_id"]])
        y = folds["label"].to_numpy()

        model = build_models(pos_weight_of(y))[QRDR_MODEL]
        model.fit(X_qrdr_all[rows], y)
        joblib.dump(model, C.MODELS / f"qrdr_{antibiotic}.joblib")

        card["models"][f"qrdr_{antibiotic}"] = {
            "feature_set": "qrdr",
            "model": QRDR_MODEL,
            "n_train": int(len(y)),
            "pct_resistant": round(float(y.mean() * 100), 1),
            **cv_metrics(reported, antibiotic, "qrdr", QRDR_MODEL),
        }
        log.info("qrdr_%s fitted on %d genomes", antibiotic, len(y))

    # hexamer model, primary antibiotic only — used for the side-by-side contrast
    folds = pd.read_csv(C.PROCESSED / f"folds_{C.PRIMARY_ANTIBIOTIC}.csv",
                        dtype={"genome_id": str})
    rows = np.array([id_to_row[g] for g in folds["genome_id"]])
    y = folds["label"].to_numpy()
    kmer_model = build_models(pos_weight_of(y))[KMER_MODEL]
    kmer_model.fit(X_kmer_all[rows], y)
    joblib.dump(kmer_model, C.MODELS / f"kmer_{C.PRIMARY_ANTIBIOTIC}.joblib")

    card["models"][f"kmer_{C.PRIMARY_ANTIBIOTIC}"] = {
        "feature_set": "kmer",
        "model": KMER_MODEL,
        "n_train": int(len(y)),
        **cv_metrics(reported, C.PRIMARY_ANTIBIOTIC, "kmer", KMER_MODEL),
    }
    log.info("kmer_%s fitted on %d genomes", C.PRIMARY_ANTIBIOTIC, len(y))

    (C.MODELS / "model_card.json").write_text(json.dumps(card, indent=2))

    print("\n" + "=" * 70)
    print("DEPLOYABLE MODELS")
    print("=" * 70)
    for name, meta in card["models"].items():
        print(f"  {name:22} {meta['model']:18} n={meta['n_train']:>5}  "
              f"AUROC {meta.get('cv_auroc_mlst_grouped', 'n/a')}  "
              f"sens {meta.get('sensitivity_mlst_grouped', 'n/a')}  "
              f"spec {meta.get('specificity_mlst_grouped', 'n/a')}")
    print(f"\nwrote {C.MODELS / 'model_card.json'}")
    print("\nStart the application from the project root with:")
    print("    streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())