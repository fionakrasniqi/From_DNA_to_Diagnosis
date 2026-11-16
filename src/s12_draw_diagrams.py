"""
s12_draw_diagrams.py — the six explanatory diagrams of the dissertation.

Needs no data. Drawn in the shared theme (theme.py) so they match the charts.

    diagram_01_cv_schemes.png       four ways of splitting genomes into folds
    diagram_02_codon.png            DNA -> codons -> amino acids, gyrA S83L
    diagram_03_palindrome.png       GAATTC on both strands, EcoRI cut
    diagram_04_pipeline.png         the eleven pipeline stages
    diagram_05_qrdr_flanks.png      reading gyrA codons 83 and 87 with flanks
    diagram_06_cnn.png              convolutional network architecture

Run from src/:  python s12_draw_diagrams.py
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import config as C
import theme as T

T.apply()
OUT = C.FIGURES


def box(ax, x, y, w, h, text="", fc="white", ec=T.DARK, lw=0.9, fs=9, color=T.INK,
        weight="normal", family=None, rounded=True):
    style = "round,pad=0,rounding_size=0.06" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=color, weight=weight, family=family)


def arrow(ax, x1, y1, x2, y2, color=T.DARK, lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=9))


def blank(ax):
    ax.set_axis_off()
    ax.grid(False)


# ── 1. cross-validation schemes ───────────────────────────────────────────────
def diagram_01():
    rng = np.random.default_rng(4)
    lineage_colours = [T.BLUE, T.TEAL, T.RED, T.DARK, T.GREY]
    lineages = np.repeat(np.arange(5), [9, 7, 6, 5, 3])          # 30 genomes
    countries = rng.integers(0, 5, len(lineages))

    rows = [
        ("E rastësishme", rng.permutation(np.arange(30) % 5)),
        ("K-means", None),
        ("Sipas MLST", lineages),
        ("Sipas vendit", None),
    ]
    km = rng.permutation(np.arange(30) % 5)
    rows[1] = ("K-means", km)

    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 3.4))
    blank(ax)
    ax.set_xlim(-2.4, 11.2)
    ax.set_ylim(-0.6, 4.3)
    for r, (label, assign) in enumerate(rows):
        y = 3.3 - r * 1.05
        ax.text(-2.35, y + 0.35, label, ha="left", va="center", fontsize=9.5)
        for f in range(5):
            x = 1.0 + f * 2.0
            if label == "Sipas vendit":
                is_test = f == 2
                box(ax, x, y, 1.8, 0.7, fc="#F3F3F3" if not is_test else "white",
                    ec=T.RED if is_test else T.GREY, lw=1.2 if is_test else 0.8)
                idx = np.where(countries == f)[0]
            else:
                box(ax, x, y, 1.8, 0.7, ec=T.GREY, lw=0.8)
                idx = np.where(assign == f)[0]
            for k, g in enumerate(idx[:12]):
                cx = x + 0.18 + (k % 6) * 0.29
                cy = y + 0.48 - (k // 6) * 0.27
                ax.add_patch(plt.Circle((cx, cy), 0.09, color=lineage_colours[lineages[g]], lw=0))
        if r == 0:
            for f in range(5):
                ax.text(1.0 + f * 2.0 + 0.9, y + 0.88, f"Pjesa {f + 1}", ha="center", fontsize=8.5,
                        color="#555555")
    ax.text(4.4, -0.45, "Ngjyra tregon linjën. Te ndarja sipas vendit, pjesa me kufi të kuq "
            "është vendi që testohet; pjesët gri përdoren për trajnim.",
            ha="center", fontsize=8.5, color="#555555")
    T.save(fig, OUT / "diagram_01_cv_schemes.png")


# ── 2. DNA, codons, amino acids, S83L ─────────────────────────────────────────
def diagram_02():
    codons = ["CAT", "GGT", "GAC", "TCG", "GCG", "GTC"]
    aminos = ["H", "G", "D", "S", "A", "V"]
    positions = [80, 81, 82, 83, 84, 85]
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}

    fig, axes = plt.subplots(2, 1, figsize=(T.TEXT_WIDTH_IN, 3.9))
    for panel, (ax, mutant) in enumerate(zip(axes, [False, True])):
        blank(ax)
        ax.set_xlim(-1.6, 18.4)
        ax.set_ylim(-1.3, 3.0)
        title = "Gjeni normal" if not mutant else "Gjeni me mutacionin S83L"
        ax.text(-1.55, 2.75, title, fontsize=10, weight="bold", va="center")
        seq = "".join(codons)
        if mutant:
            seq = seq[:10] + "T" + seq[11:]
        for i, base in enumerate(seq):
            changed = mutant and i == 10
            colour = T.RED if changed else T.INK
            ax.text(i + 0.5, 1.95, base, ha="center", va="center", fontsize=10.5,
                    family="monospace", color=colour, weight="bold" if changed else "normal")
            ax.text(i + 0.5, 1.35, comp[base], ha="center", va="center", fontsize=10.5,
                    family="monospace", color="#888888")
        ax.text(-0.3, 1.95, "5′", ha="right", va="center", fontsize=8.5, color="#555555")
        ax.text(-0.3, 1.35, "3′", ha="right", va="center", fontsize=8.5, color="#555555")
        for c in range(6):
            x = c * 3
            focus = positions[c] == 83
            ec = (T.RED if mutant else T.BLUE) if focus else T.GREY
            ax.add_patch(Rectangle((x + 0.08, 1.65), 2.84, 0.6, fill=False, ec=ec,
                                   lw=1.3 if focus else 0.7))
            aa = aminos[c] if not (mutant and focus) else "L"
            fc = (T.RED if mutant else T.BLUE) if focus else "#F2F2F2"
            tc = "white" if focus else T.INK
            box(ax, x + 0.55, -0.15, 1.9, 0.62, aa, fc=fc, ec=fc if focus else T.GREY,
                fs=10, color=tc, weight="bold")
            arrow(ax, x + 1.5, 1.08, x + 1.5, 0.52, color="#888888", lw=0.7)
            ax.text(x + 1.5, -0.62, str(positions[c]), ha="center", fontsize=8, color="#555555")
        note = ("Kodoni 83: TCG → serinë (S)" if not mutant
                else "Kodoni 83: TTG → leucinë (L); ndryshoi vetëm një shkronjë")
        ax.text(9, -1.12, note, ha="center", fontsize=9, color=T.RED if mutant else T.BLUE)
    fig.subplots_adjust(hspace=0.15)
    T.save(fig, OUT / "diagram_02_codon.png")


# ── 3. palindrome on both strands ─────────────────────────────────────────────
def diagram_03():
    top, bottom = "GAATTC", "CTTAAG"
    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 2.5))
    blank(ax)
    ax.set_xlim(-3, 11)
    ax.set_ylim(-1.6, 2.6)
    x0 = 1.5
    flank_l, flank_r = "TA", "AG"
    for i, b in enumerate(flank_l + top + flank_r):
        inside = 2 <= i < 8
        ax.text(x0 + i * 0.75, 1.0, b, ha="center", va="center", fontsize=13, family="monospace",
                color=T.INK if inside else "#AAAAAA", weight="bold" if inside else "normal")
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    for i, b in enumerate(flank_l + top + flank_r):
        inside = 2 <= i < 8
        ax.text(x0 + i * 0.75, 0.0, comp[b], ha="center", va="center", fontsize=13,
                family="monospace", color=T.INK if inside else "#AAAAAA",
                weight="bold" if inside else "normal")
    ax.add_patch(Rectangle((x0 + 1.5 - 0.37, -0.42), 6 * 0.75, 1.84, fill=False, ec=T.BLUE, lw=1.2))
    ax.text(x0 - 0.7, 1.0, "5′", ha="right", va="center", fontsize=9, color="#555555")
    ax.text(x0 + 10 * 0.75 - 0.05, 1.0, "3′", ha="left", va="center", fontsize=9, color="#555555")
    ax.text(x0 - 0.7, 0.0, "3′", ha="right", va="center", fontsize=9, color="#555555")
    ax.text(x0 + 10 * 0.75 - 0.05, 0.0, "5′", ha="left", va="center", fontsize=9, color="#555555")
    arrow(ax, x0 + 1.6, 1.75, x0 + 5.6, 1.75, color=T.BLUE)
    ax.text(x0 + 3.6, 2.05, "lexohet: GAATTC", ha="center", fontsize=9, color=T.BLUE)
    arrow(ax, x0 + 5.6, -0.75, x0 + 1.6, -0.75, color=T.BLUE)
    ax.text(x0 + 3.6, -1.15, "lexohet: GAATTC", ha="center", fontsize=9, color=T.BLUE)
    # staggered EcoRI cut: G^AATTC on top, CTTAA^G on bottom
    cx_top = x0 + 2.5 * 0.75 + 0.375 - 0.375
    xs = [x0 + 2 * 0.75 + 0.37, x0 + 2 * 0.75 + 0.37, x0 + 7 * 0.75 - 0.37, x0 + 7 * 0.75 - 0.37]
    ax.plot([xs[0], xs[1], xs[2], xs[3]], [1.4, 0.5, 0.5, -0.4], color=T.RED, lw=1.2, ls="--")
    ax.text(9.6, 0.5, "EcoRI e pret\nADN-në këtu", ha="left", va="center", fontsize=9, color=T.RED)
    T.save(fig, OUT / "diagram_03_palindrome.png")


# ── 4. pipeline architecture ──────────────────────────────────────────────────
def diagram_04():
    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 4.6))
    blank(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.4)
    groups = [
        ("Të dhënat", 8.0, [("s01", "Fenotipet"), ("s02", "Gjenomet*"), ("s03", "Cilësia,\nheksamerët")]),
        ("Veçoritë", 5.9, [("s04", "CARD, QRDR"), ("s05", "Skemat e\nvlerësimit")]),
        ("Modelet", 3.8, [("s06", "Modelet\nklasike"), ("s07", "Rrjeti\nneural"), ("s08", "SHAP")]),
        ("Rezultatet", 1.7, [("s09", "Figurat,\ntabelat"), ("s10", "Modelet e\naplikacionit"),
                             ("s11", "Validimi i\njashtëm")]),
    ]
    centres = {}
    for title, y, stages in groups:
        ax.text(0.15, y + 0.42, title, fontsize=9.5, weight="bold", va="center")
        for k, (code, label) in enumerate(stages):
            x = 3.0 + k * 3.0
            fc = "white"
            ec = T.RED if code == "s02" else T.DARK
            box(ax, x, y, 2.5, 0.95, "", fc=fc, ec=ec, lw=1.1 if code == "s02" else 0.9)
            ax.text(x + 0.12, y + 0.72, code, fontsize=7.5, color="#777777", va="center")
            ax.text(x + 1.25, y + 0.40, label, ha="center", va="center", fontsize=8.8)
            centres[code] = (x, y)
    def link(a, b):
        xa, ya = centres[a]; xb, yb = centres[b]
        if abs(ya - yb) < 0.01:
            arrow(ax, xa + 2.5, ya + 0.47, xb, yb + 0.47)
        else:
            arrow(ax, xa + 1.25, ya, xb + 1.25, yb + 0.95)
    for a, b in [("s01", "s02"), ("s02", "s03"), ("s03", "s04"), ("s04", "s05"),
                 ("s05", "s06"), ("s05", "s07"), ("s05", "s08"), ("s06", "s09"), ("s08", "s09"),
                 ("s09", "s10"), ("s10", "s11")]:
        link(a, b)
    ax.text(6.0, 0.45, "* Dy mbrojtje kundër cungimit: kufi i qartë i rreshtave dhe "
            "verifikimi i gjatësisë (≥ 95%)", ha="center", fontsize=8.5, color=T.RED)
    T.save(fig, OUT / "diagram_04_pipeline.png")


# ── 5. reading QRDR codons with flanks ────────────────────────────────────────
def diagram_05():
    flank83, codon83, middle, codon87, flank87 = ("CATCCCCATGGTGAC", "TCG", "GCGGTCTAT",
                                                  "GAC", "ACGATCGTCCGTATG")
    seq = flank83 + codon83 + middle + codon87 + flank87
    w = 0.3
    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 2.5))
    blank(ax)
    ax.set_xlim(-0.2, len(seq) * w + 0.2)
    ax.set_ylim(-1.7, 2.3)
    spans = {"f83": (0, 15), "c83": (15, 18), "c87": (27, 30), "f87": (30, 45)}
    for i, b in enumerate(seq):
        colour = T.INK
        if spans["c83"][0] <= i < spans["c83"][1] or spans["c87"][0] <= i < spans["c87"][1]:
            colour = T.RED
        ax.text(i * w + w / 2, 0.5, b, ha="center", va="center", fontsize=8.6, family="monospace",
                color=colour, weight="bold" if colour == T.RED else "normal")
    def bracket(a, b, y, text, colour, below=False):
        x1, x2 = a * w + 0.03, b * w - 0.03
        ax.plot([x1, x1, x2, x2], [y, y + (0.15 if not below else -0.15)] * 2 if False else
                [y, y + (-0.15 if below else 0.15), y + (-0.15 if below else 0.15), y],
                color=colour, lw=1.0)
        ax.text((x1 + x2) / 2, y + (-0.45 if below else 0.42), text, ha="center", fontsize=8.6,
                color=colour)
    bracket(*spans["f83"], 0.85, "sekuenca referuese për kodonin 83 (15 baza)", T.BLUE)
    bracket(*spans["f87"], 0.85, "sekuenca referuese për kodonin 87 (15 baza)", T.BLUE)
    bracket(*spans["c83"], 0.15, "kodoni 83", T.RED, below=True)
    bracket(*spans["c87"], 0.15, "kodoni 87", T.RED, below=True)
    y = -0.95
    ax.annotate("", xy=(27 * w, y), xytext=(15 * w, y),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8))
    ax.text(21 * w, y - 0.35, "12 baza", ha="center", fontsize=8.6, color="#555555")
    ax.text(len(seq) * w / 2, -1.65, "Secila sekuencë referuese merret nga ana që nuk përmban "
            "kodonin tjetër, kështu që një mutacion në 83 nuk e pengon leximin e 87.",
            ha="center", fontsize=8.4, color="#555555")
    T.save(fig, OUT / "diagram_05_qrdr_flanks.png")


# ── 6. CNN architecture ───────────────────────────────────────────────────────
def diagram_06():
    layers = [
        ("Hyrja", "6,000 baza", 1.0, 3.0, "#F2F2F2"),
        ("Kodimi", "6,000 × 8", 1.0, 3.0, "white"),
        ("Konvolucion\n32 filtra, k = 9\n+ ReLU", "6,000 × 32", 1.0, 2.6, "white"),
        ("Max pooling\n÷ 4", "1,500 × 32", 1.0, 1.8, "white"),
        ("Konvolucion\n64 filtra, k = 9\n+ ReLU", "1,500 × 64", 1.0, 1.8, "white"),
        ("Maksimumi\nglobal", "64", 1.0, 1.0, "white"),
        ("Dropout 30%\n+ dalja", "P(R)", 1.0, 0.7, "#F2F2F2"),
    ]
    fig, ax = plt.subplots(figsize=(T.TEXT_WIDTH_IN, 2.7))
    blank(ax)
    ax.set_xlim(-0.2, 13.2)
    ax.set_ylim(-1.4, 3.6)
    for k, (name, dim, wdt, hgt, fc) in enumerate(layers):
        x = k * 1.9
        y = 1.5 - hgt / 2
        ec = T.BLUE if "Konvolucion" in name else T.DARK
        box(ax, x, y, 1.35, hgt, "", fc=fc, ec=ec, lw=1.1 if ec == T.BLUE else 0.9, rounded=False)
        ax.text(x + 0.675, -0.45, name, ha="center", va="top", fontsize=7.8, linespacing=1.15)
        ax.text(x + 0.675, 1.5, dim, ha="center", va="center", fontsize=7.6, color="#444444",
                rotation=90 if hgt > 1.5 else 0)
        if k < len(layers) - 1:
            arrow(ax, x + 1.38, 1.5, x + 1.87, 1.5, color="#777777", lw=0.8)
    T.save(fig, OUT / "diagram_06_cnn.png")


def main() -> int:
    for fn in [diagram_01, diagram_02, diagram_03, diagram_04, diagram_05, diagram_06]:
        fn()
        print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
