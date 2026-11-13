"""Draw the schematic figures used in the notebook (saved in figures/).

    python scripts/make_figures.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

os.makedirs("figures", exist_ok=True)

TOP_RED, QCD_BLUE, GREY, INK = "#c0392b", "#2c6fbb", "#9aa0a6", "#202124"

# The first jet of the training set: its 4 leading constituents and its last real one (23 in total)
REAL_ROWS = [
    (0, (474.1, -250.3, -223.7, -334.7)),
    (1, (103.2, -48.9, -56.8, -71.0)),
    (2, (105.3, -55.4, -50.0, -74.2)),
    (3, (40.2, -21.8, -18.7, -28.1)),
    (None, None),
    (22, (0.4, -0.3, -0.1, -0.3)),
]
PADDED_ROWS = [(23, (0, 0, 0, 0)), (None, None), (199, (0, 0, 0, 0))]


def jet_constituents_figure(path="figures/jet_constituents.png"):
    fig = plt.figure(figsize=(14, 6.2))

    # ---------------- left: a jet is a spray of particles
    ax = fig.add_axes([0.01, 0.08, 0.40, 0.84])
    ax.set_xlim(-1.6, 10.5)
    ax.set_ylim(-5.3, 4.2)
    ax.axis("off")
    ax.set_title("A jet: a spray of particles (its constituents)", fontsize=14, color=INK, loc="left")

    half_angle = np.deg2rad(17)
    length = 9.6
    cone = Polygon([(0, 0), (length, length*np.tan(half_angle)), (length, -length*np.tan(half_angle))],
                   closed=True, facecolor="#f3e9d2", edgecolor="#b8a46a", linewidth=1.5, alpha=0.8)
    ax.add_patch(cone)
    ax.plot([0, length], [0, 0], linestyle="--", color="#8a7a4a", linewidth=1)
    ax.text(length + 0.05, 0.12, "jet axis", fontsize=10, color="#6b5d33", va="bottom", ha="right")

    rng = np.random.default_rng(3)
    angles = np.concatenate([[0.02, -0.06, 0.09], rng.uniform(-0.85, 0.85, 14)*half_angle])
    lengths = np.concatenate([[8.6, 6.4, 5.8], rng.uniform(1.5, 4.5, 14)])
    for k, (a, l) in enumerate(zip(angles, lengths)):
        lead = k == 0
        arrow = FancyArrowPatch((0, 0), (l*np.cos(a), l*np.sin(a)), arrowstyle="-|>", mutation_scale=14 if lead else 9,
                                linewidth=2.6 if lead else (1.8 if k < 3 else 1.0), color=TOP_RED if lead else INK,
                                alpha=1 if k < 3 else 0.65)
        ax.add_patch(arrow)
    ax.annotate("constituent 0:\nthe highest-$p_T$ particle", xy=(8.0, 8.6*np.sin(0.02)+0.05), xytext=(5.6, 3.35),
                fontsize=10.5, color=TOP_RED, arrowprops=dict(arrowstyle="->", color=TOP_RED))
    ax.annotate("softer particles\n(shorter arrows = less $p_T$)", xy=(3.4, -0.75), xytext=(0.2, -3.3), va="top",
                fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK))
    ax.plot(0, 0, marker="*", markersize=20, color="#f2b705", markeredgecolor=INK)
    ax.text(0, -0.6, "proton-proton\ncollision", fontsize=10, ha="center", va="top", color=INK)
    ax.text(length - 0.2, -length*np.tan(half_angle) - 0.35, "cone of radius R = 0.8", fontsize=10,
            ha="right", va="top", color="#6b5d33")

    ax.text(-1.5, -5.25, "Each constituent has a four-momentum $(E, p_x, p_y, p_z)$.\n"
                         "The jet's four-momentum is the sum over its constituents.",
            fontsize=11, color=INK, va="bottom")

    # ---------------- right: how a jet is stored in the dataset
    ax = fig.add_axes([0.45, 0.08, 0.54, 0.84])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("How one jet is stored in the dataset", fontsize=14, color=INK, loc="left")

    columns = ["constituent", "$E$", "$p_x$", "$p_y$", "$p_z$"]
    x_cols = [0.9, 2.3, 3.45, 4.6, 5.75]
    x_pt = 6.65
    row_h = 0.78
    y = 9.2
    for x, name in zip(x_cols, columns):
        ax.text(x, y, name, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(x_pt, y, "$p_T$\n(not stored)", ha="center", va="center", fontsize=9.5, color="#5f6368", style="italic")
    ax.plot([0.1, 7.0], [y - 0.42, y - 0.42], color=INK, linewidth=1)

    def draw_rows(rows, y, real):
        for index, values in rows:
            y -= row_h
            if index is None:
                for x in x_cols:
                    ax.text(x, y, "⋮", ha="center", va="center", fontsize=13, color=GREY if not real else INK)
                continue
            ax.add_patch(Rectangle((0.15, y - row_h/2 + 0.06), 6.85, row_h - 0.12,
                                   facecolor="#fde8e4" if real else "#eeeeee", edgecolor="none"))
            ax.text(x_cols[0], y, str(index), ha="center", va="center", fontsize=11.5, color=INK if real else GREY)
            for x, v in zip(x_cols[1:], values):
                ax.text(x, y, f"{v:.1f}" if real else "0", ha="center", va="center", fontsize=11.5,
                        color=INK if real else GREY, family="monospace")
            ax.text(x_pt, y, f"{np.hypot(values[1], values[2]):.1f}" if real else "0", ha="center", va="center",
                    fontsize=10.5, color="#5f6368", family="monospace", style="italic")
        return y

    y_top = y
    y = draw_rows(REAL_ROWS, y, real=True)
    y_real_bottom = y
    y = draw_rows(PADDED_ROWS, y, real=False)

    def brace(y1, y2, text, color):
        x = 7.2
        ax.annotate("", xy=(x, y1), xytext=(x, y2), arrowprops=dict(arrowstyle="-", color=color, linewidth=2))
        for yy in (y1, y2):
            ax.plot([x - 0.12, x], [yy, yy], color=color, linewidth=2)
        ax.text(x + 0.25, (y1 + y2)/2, text, va="center", fontsize=11, color=color)

    brace(y_top - row_h/2 + 0.05, y_real_bottom - row_h/2 + 0.1,
          "real particles\n(here 23 of them),\nsorted by $p_T$:\nhighest first", TOP_RED)
    brace(y_real_bottom - row_h/2 - 0.05, y - row_h/2 + 0.1, "zero-padding,\nup to 200 rows", "#5f6368")

    ax.text(0.15, 0.55, "In the file each jet is one row of 200 × 4 = 800 numbers:\n"
                        "E_0, PX_0, PY_0, PZ_0, E_1, PX_1, ..., PZ_199,  plus the label is_signal_new (1 = top, 0 = QCD).\n"
                        "A real particle always has $E > 0$, so the mask $E > 0$ tells real particles from padding.",
            fontsize=10.5, color=INK, va="bottom")

    fig.savefig(path, dpi=150, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    jet_constituents_figure()
