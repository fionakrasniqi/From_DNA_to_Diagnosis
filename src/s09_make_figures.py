"""
s09_make_figures.py — charts and tables for the dissertation.

Reads the outputs of stages 05 to 08 and 11 and draws every chart in the shared
theme (theme.py): thesis typeface, one colour per feature set, labels next to
the data, Albanian text, no titles inside the figure (the caption is in the
thesis), 16 cm wide at 300 dpi. No model is fitted here.

Results are shown for all three models, so no chart depends on picking the
best model with the test folds.

Figures:
  01_lineage_gap.png             AUROC by validation design, one panel per model
  03_cluster_validation.png      k-means clusters versus MLST lineages
  04_cnn_comparison.png          CNN arms against the classical models
  05_palindrome_signature.png    canonical top hexamers: palindromes, lineage vs genotype
  06_qrdr_prevalence.png         QRDR residue prevalence
  07_external_validation.png     ROC curves on the NCBI genomes
  08_shap_summary.png            appendix: SHAP ranking and lineage attribution

Tables (CSV): T1 cohort, T2 main results, T3 validation designs, T4 QRDR control
"""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
import theme as T

log = logging.getLogger("s09")
T.apply()

FEATURES = ["kmer", "card", "qrdr", "combined"]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
DESIGNS = ["fold_random", "fold_grouped", "fold_mlst", "fold_country"]


def end_label(ax, x, y, text, colour, dy=0.0):
    ax.text(x + 0.12, y + dy, text, color=colour, va="center", fontsize=8.5)


# ── 01 ────────────────────────────────────────────────────────────────────────
def figure_01(results: pd.DataFrame) -> None:
    cipro = results[results["antibiotic"] == C.PRIMARY_ANTIBIOTIC]
    fig, axes = plt.subplots(1, 3, figsize=(T.TEXT_WIDTH_IN, 2.9), sharey=True)
    x = np.arange(len(DESIGNS))
    for ax, model in zip(axes, MODELS):
        sub = cipro[cipro["model"] == model]
        finals = []
        for feature in FEATURES:
            s = sub[sub["feature_set"] == feature].set_index("cv_design")["auroc_mean"]
            y = [s.get(d, np.nan) for d in DESIGNS]
            ax.plot(x, y, marker="o", color=T.FEATURE_COLOURS[feature], markersize=3.5)
            finals.append((y[-1], feature))
        ax.set_xticks(x)
        ax.set_xticklabels(["Rastësisht", "K-means", "MLST", "Vendi"], fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(T.MODEL_SQ[model], fontsize=9.5)
        ax.set_ylim(0.75, 1.0)
        ax.set_xlim(-0.3, 3.3)
    # labels once, on the right panel, spaced so they do not collide
    finals.sort()
    placed = []
    for value, feature in finals:
        y = value
        for p in placed:
            if abs(y - p) < 0.024:
                y = p + 0.024
        placed.append(y)
        axes[-1].text(3.4, y, T.FEATURE_SQ[feature], color=T.FEATURE_COLOURS[feature],
                      va="center", fontsize=8.5, clip_on=False)
    axes[0].set_ylabel("AUROC")
    fig.supxlabel("Skema e vlerësimit", fontsize=9.5, y=-0.02)
    fig.subplots_adjust(wspace=0.12, right=0.85, bottom=0.27)
    T.save(fig, C.FIGURES / "01_lineage_gap.png")


# ── 03 ────────────────────────────────────────────────────────────────────────
def figure_03() -> None:
    path = C.TABLES / "major_st_concentration.csv"
    if not path.exists():
        return
    detail = pd.read_csv(path).nlargest(12, "n_genomes")
    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 3.2))
    y = np.arange(len(detail))
    ax.barh(y, detail["concentration"], color=T.GREY, height=0.62)
    chance = 1 / C.N_CLUSTERS
    ax.axvline(chance, color=T.INK, ls=(0, (3, 2)), lw=0.9)
    ax.text(chance + 0.01, -0.75, "niveli i rastësishëm (20%)", fontsize=8.5, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"ST{int(s)} (n = {int(n):,})" for s, n
                        in zip(detail["sequence_type"], detail["n_genomes"])], fontsize=8.5)
    ax.set_ylim(len(detail) - 0.5, -1.2)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Pjesa e linjës që ndodhet në grupin e saj kryesor")
    T.save(fig, C.FIGURES / "03_cluster_validation.png")


# ── 04 ────────────────────────────────────────────────────────────────────────
def figure_04(results: pd.DataFrame) -> None:
    cnn_path = C.TABLES / "cnn_results.csv"
    if not cnn_path.exists():
        return
    cnn = pd.read_csv(cnn_path)
    strict = results[(results["antibiotic"] == C.PRIMARY_ANTIBIOTIC)
                     & (results["cv_design"] == "fold_mlst")]
    q = strict[strict["feature_set"] == "qrdr"]["auroc_mean"]
    k = strict[strict["feature_set"] == "kmer"]["auroc_mean"]
    bars = [
        ("Veçoritë QRDR\n(3 modele)", q.max(), q.min(), q.max(), T.BLUE),
        ("Heksamerët\n(3 modele)", k.max(), k.min(), k.max(), T.RED),
    ]
    for arm, label, colour in [("targeted", "CNN: gjenet\ngyrA dhe parC", T.DARK),
                               ("random", "CNN: segmente\ntë rastësishme", T.GREY)]:
        s = cnn[cnn["arm"] == arm]["auroc"]
        bars.append((label, s.mean(), s.mean() - s.std(), s.mean() + s.std(), colour))

    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 3.0))
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[4] for b in bars], width=0.58)
    for xi, (label, h, lo, hi, _) in zip(x, bars):
        if hi - lo > 0.002:
            ax.plot([xi, xi], [lo, hi], color=T.INK, lw=0.9)
            ax.plot([xi - 0.06, xi + 0.06], [lo, lo], color=T.INK, lw=0.9)
            ax.plot([xi - 0.06, xi + 0.06], [hi, hi], color=T.INK, lw=0.9)
        text = f"{lo:.3f}–{hi:.3f}" if label.startswith("Heks") else f"{h:.3f}"
        ax.text(xi, max(h, hi) + 0.012, text, ha="center", fontsize=8.5)
    ax.axhline(0.5, color=T.INK, ls=":", lw=0.9)
    ax.text(-0.42, 0.508, "niveli i rastësishëm", fontsize=8, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("AUROC (grupim sipas MLST)")
    ax.set_ylim(0.4, 1.05)
    T.save(fig, C.FIGURES / "04_cnn_comparison.png")


# ── 05 and 08 ─────────────────────────────────────────────────────────────────
def _canonical():
    path = C.TABLES / "shap_top_hexamers_canonical.csv"
    if not path.exists():
        log.warning("no canonical SHAP table — run s08_shap_validation.py first")
        return None, None
    top = pd.read_csv(path).fillna({"enzyme": ""})
    test_path = C.TABLES / "shap_palindrome_test.csv"
    test = pd.read_csv(test_path).iloc[0] if test_path.exists() else None
    return top, test


def _shap_bars(ax, top, test):
    colours = [T.BLUE if p else T.GREY for p in top["palindromic"]]
    y = np.arange(len(top))
    ax.barh(y, top["mean_abs_shap"], color=colours, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{h}  {e}" if e else h for h, e in zip(top["hexamer"], top["enzyme"])],
                       fontsize=7.8, family="monospace")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Mesatarja e |SHAP|")
    if test is not None:
        ax.text(0.0, 1.02, f"blu: palindromike, {int(test['palindromic'])} nga "
                f"{int(test['top_n'])} (pritet {test['base_rate']:.1%})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=T.BLUE)


def figure_05() -> None:
    top, test = _canonical()
    if top is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(T.TEXT_WIDTH_IN, 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    _shap_bars(axes[0], top, test)
    ax = axes[1]
    colours = [T.BLUE if p else T.GREY for p in top["palindromic"]]
    ax.scatter(top["lineage_eps2"], top["genotype_eps2"], c=colours, s=22,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=T.INK, ls=(0, (3, 2)), lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Varianca e shpjeguar nga linja (ε²)")
    ax.set_ylabel("Varianca e shpjeguar nga gjenotipi (ε²)")
    ax.text(0.04, 0.95, "mbi vijë: më shumë gjenotip\nnën vijë: më shumë linjë",
            transform=ax.transAxes, va="top", fontsize=8, color="#555555")
    fig.subplots_adjust(wspace=0.5)
    T.save(fig, C.FIGURES / "05_palindrome_signature.png")


def figure_08() -> None:
    top, test = _canonical()
    if top is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(T.TEXT_WIDTH_IN, 3.6), sharey=False)
    _shap_bars(axes[0], top, test)
    ax = axes[1]
    y = np.arange(len(top))
    ax.barh(y, top["lineage_eps2"], color=T.RED, height=0.68)
    null = top["lineage_eps2_null_p95"].max()
    ax.axvline(null, color=T.INK, ls=(0, (3, 2)), lw=0.9)
    ax.text(null + 0.02, -0.8, "niveli i rastësishëm (95%)", fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(top["hexamer"], fontsize=7.8, family="monospace")
    ax.set_ylim(len(top) - 0.5, -1.4)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Varianca e shpjeguar nga linja (ε²)")
    fig.subplots_adjust(wspace=0.55)
    T.save(fig, C.FIGURES / "08_shap_summary.png")


# ── 06 ────────────────────────────────────────────────────────────────────────
def figure_06() -> None:
    path = C.PROCESSED / "qrdr_residues.csv"
    if not path.exists():
        return
    residues = pd.read_csv(path, dtype={"genome_id": str})
    sites = {"gyrA_83": ("S", ["L", "A", "T", "V", "W"]),
             "gyrA_87": ("D", ["N", "G", "Y", "A", "H", "V"]),
             "parC_80": ("S", ["I", "R"]),
             "parC_84": ("E", ["V", "G", "K", "A"])}
    fig, axes = plt.subplots(1, 4, figsize=(T.TEXT_WIDTH_IN, 2.6), sharey=True)
    total = len(residues)
    for ax, (site, (wt, resistant)) in zip(axes, sites.items()):
        counts = residues[site].value_counts()
        shown = [r for r in [wt] + resistant if counts.get(r, 0) / total >= 0.0005]
        values = [counts[r] / total * 100 for r in shown]
        ax.bar(range(len(shown)), values, color=[T.BLUE if r == wt else T.RED for r in shown],
               width=0.65)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels(shown)
        gene, pos = site.split("_")
        ax.set_xlabel(f"{gene} {pos}", fontstyle="italic" if False else "normal")
        for i, v in enumerate(values):
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    axes[0].set_ylabel("% e gjenomeve")
    axes[0].set_ylim(0, 100)
    fig.text(0.5, 1.0, "blu: aminoacidi i zakonshëm     kuq: aminoacid i lidhur me rezistencën",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.subplots_adjust(wspace=0.12)
    T.save(fig, C.FIGURES / "06_qrdr_prevalence.png")


# ── 07 ────────────────────────────────────────────────────────────────────────
def figure_07() -> None:
    path = C.TABLES / "external_predictions.csv"
    if not path.exists():
        return
    p = pd.read_csv(path)
    p = p[p["qrdr_determinable"]]
    y = p["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for column, label, colour in [("p_qrdr", "QRDR", T.BLUE), ("p_kmer", "K-mer", T.RED)]:
        fpr, tpr, _ = roc_curve(y, p[column])
        auc = roc_auc_score(y, p[column])
        ax.plot(fpr, tpr, color=colour, lw=1.6)
        xy = (0.06, 0.90) if label == "QRDR" else (0.14, 0.46)
        ax.text(*xy, f"{label}\nAUROC {auc:.3f}", color=colour, fontsize=8.5, va="bottom")
    ax.plot([0, 1], [0, 1], color=T.INK, ls=":", lw=0.8)
    ax.text(0.62, 0.55, "niveli i rastësishëm", rotation=40, fontsize=7.8, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("1 − specificiteti")
    ax.set_ylabel("Ndjeshmëria")
    T.save(fig, C.FIGURES / "07_external_validation.png")


# ── tables ────────────────────────────────────────────────────────────────────
def write_tables(results: pd.DataFrame) -> None:
    rows = []
    for antibiotic in C.ANTIBIOTICS:
        path = C.PROCESSED / f"folds_{antibiotic}.csv"
        if not path.exists():
            continue
        y = pd.read_csv(path, usecols=["label"])["label"]
        rows.append({"antibiotic": antibiotic, "antibiotiku": T.ANTIBIOTIC_SQ[antibiotic],
                     "n_genomes": len(y), "n_resistant": int(y.sum()),
                     "n_susceptible": int((1 - y).sum()), "pct_resistant": round(y.mean() * 100, 1)})
    if rows:
        pd.DataFrame(rows).to_csv(C.TABLES / "T1_cohort.csv", index=False)

    (results[results["cv_design"] == "fold_mlst"]
     [["antibiotic", "feature_set", "model", "auroc_mean", "auroc_std", "auprc_mean",
       "sensitivity_mean", "specificity_mean"]]
     .sort_values(["antibiotic", "feature_set", "model"])
     .to_csv(C.TABLES / "T2_main_results.csv", index=False))

    comparison = results.pivot_table(index=["antibiotic", "model", "feature_set"],
                                     columns="cv_design", values="auroc_mean")
    comparison = comparison[[d for d in DESIGNS if d in comparison.columns]]
    if {"fold_random", "fold_mlst"} <= set(comparison.columns):
        comparison["lineage_gap"] = comparison["fold_random"] - comparison["fold_mlst"]
    comparison.round(4).to_csv(C.TABLES / "T3_cv_designs.csv")

    control = results[(results["feature_set"] == "qrdr")
                      & results["cv_design"].isin(["fold_mlst", "fold_country"])]
    (control.pivot_table(index=["antibiotic", "model"], columns="cv_design", values="auroc_mean")
     .round(3).to_csv(C.TABLES / "T4_qrdr_specificity.csv"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    path = C.TABLES / "model_comparison.csv"
    if not path.exists():
        log.error("missing %s — run s06_train_classical.py first", path)
        return 1
    results = pd.read_csv(path)
    builders = [("01_lineage_gap", lambda: figure_01(results)),
                ("03_cluster_validation", figure_03),
                ("04_cnn_comparison", lambda: figure_04(results)),
                ("05_palindrome_signature", figure_05),
                ("06_qrdr_prevalence", figure_06),
                ("07_external_validation", figure_07),
                ("08_shap_summary", figure_08)]
    for name, build in builders:
        try:
            build()
            log.info("%s %s", "wrote" if (C.FIGURES / f"{name}.png").exists() else "skipped", name)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("figure %s failed: %s", name, exc)
    write_tables(results)
    print("\nnext: python s12_draw_diagrams.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
