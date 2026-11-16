"""
theme.py — one visual style for every figure, diagram and the application.

Colours carry meaning and keep it everywhere:
    QRDR features / wild-type residue ........ blue
    CARD features ............................ teal
    k-mer features / resistance residue ...... brick red
    combined features / CNN on gyrA-parC ..... dark grey
    background elements / chance level ....... light grey

Charts use the thesis typeface (Times New Roman, with metric-compatible
fallbacks), no box around the plot, light horizontal grid lines only, and
labels written next to the data instead of a separate legend where possible.
Figures are sized to the A4 text width (16 cm) and saved at 300 dpi.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2B5C8A"
TEAL = "#2E8B7A"
RED = "#B04A3C"
DARK = "#4D4D4D"
GREY = "#A6A6A6"
LIGHT = "#E6E6E6"
INK = "#222222"

FEATURE_COLOURS = {"qrdr": BLUE, "card": TEAL, "kmer": RED, "combined": DARK}
FEATURE_SQ = {"kmer": "K-mer", "card": "CARD", "qrdr": "QRDR", "combined": "E kombinuar"}
MODEL_SQ = {"LogisticRegression": "Regresioni logjistik",
            "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
DESIGN_SQ = {"fold_random": "E rastësishme", "fold_grouped": "K-means",
             "fold_mlst": "Sipas MLST", "fold_country": "Sipas vendit"}
ANTIBIOTIC_SQ = {"ciprofloxacin": "Ciprofloksacina", "ampicillin": "Ampicilina",
                 "tetracycline": "Tetraciklina", "gentamicin": "Gentamicina"}

TEXT_WIDTH_IN = 6.3          # 16 cm
DPI = 300


def apply() -> None:
    """Set the shared matplotlib style. Call once before drawing."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#555555",
        "axes.labelcolor": INK,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def save(fig, path) -> None:
    fig.savefig(path, facecolor="white")
    plt.close(fig)
