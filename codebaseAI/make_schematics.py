"""
Vẽ figures sơ đồ cho paper (không cần data):
  Figure 1: Pipeline overview   -> paper/figures/fig1_pipeline.png
  Figure 2: Spiking U-Net arch  -> paper/figures/fig2_architecture.png
  Figure 4: Decision tree       -> paper/figures/fig4_decision_tree.png
Palette colorblind-friendly (Paul Tol), 300 DPI.
    python make_schematics.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../paper/figures"
os.makedirs(OUT, exist_ok=True)

# Paul Tol bright (colorblind-safe)
BLUE, RED, GREEN, YELLOW, PURPLE, GREY = "#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#BBBBBB"


def box(ax, x, y, w, h, text, fc, fontsize=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="black", lw=1.2, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="black", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw))


# ---------- Figure 1: Pipeline ----------
def fig1():
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2)
    y, h = 1.1, 1.1
    box(ax, 0.2, y, 2.0, h, "Sentinel-1 SAR\n(VV, VH; dB)", BLUE, tc="white")
    box(ax, 2.7, y, 2.4, h, "Preprocessing\nclip [-50,0] dB · norm [0,1]\nNaN→ignore · JRC 3-class", YELLOW)
    box(ax, 5.6, y, 2.4, h, "Model\nSpiking U-Net / CNN /\nTransformer (shared protocol)", GREEN, tc="white")
    box(ax, 8.5, y, 2.3, h, "3-class output\nbackground / permanent\nwater / flood", RED, tc="white")
    for x in (2.2, 5.1, 8.0):
        arrow(ax, x, y + h / 2, x + 0.5, y + h / 2)
    ax.set_title("Figure 1. Pipeline overview: SAR → preprocessing → segmentation model → 3-class flood map",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_pipeline.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/fig1_pipeline.png")


# ---------- Figure 2: Spiking U-Net architecture ----------
def fig2():
    fig, ax = plt.subplots(figsize=(11, 6)); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    enc = [("e1\n32", 1.0, 4.6), ("e2\n64", 2.2, 3.7), ("e3\n128", 3.4, 2.8), ("e4\n256", 4.6, 1.9)]
    dec = [("d4\n128", 6.4, 1.9), ("d3\n64", 7.6, 2.8), ("d2\n32", 8.8, 3.7), ("d1\n32", 10.0, 4.6)]
    bw, bh = 0.95, 0.7
    for name, x, yy in enc:
        box(ax, x, yy, bw, bh, name, BLUE, 8, "white")
    box(ax, 5.5, 1.0, bw, bh, "bott\n512", PURPLE, 8, "white")
    for name, x, yy in dec:
        box(ax, x, yy, bw, bh, name, GREEN, 8, "white")
    # input / output
    box(ax, 0.0, 5.2, 0.9, 0.7, "SAR\n2×H×W", GREY, 7)
    box(ax, 10.05, 5.2, 0.9, 0.7, "3×H×W\nlogits", RED, 7, "white")
    # encoder path arrows (down)
    pts = [(1.475, 5.9)] + [(x + bw / 2, yy) for _, x, yy in enc] + [(6.0, 1.7)]
    chain = [(0.45, 5.55, 1.0, 4.95)]
    arrow(ax, 0.9, 5.55, 1.2, 5.3)
    for i in range(len(enc) - 1):
        arrow(ax, enc[i][1] + bw / 2, enc[i][2], enc[i + 1][1] + bw / 2, enc[i + 1][2] + bh)
    arrow(ax, enc[-1][1] + bw / 2, enc[-1][2], 5.9, 1.7)               # e4 -> bott
    arrow(ax, 5.5 + bw, 1.35, dec[0][1], dec[0][2] + bh / 2)          # bott -> d4
    for i in range(len(dec) - 1):
        arrow(ax, dec[i][1] + bw, dec[i][2] + bh, dec[i + 1][1], dec[i + 1][2])
    arrow(ax, dec[-1][1] + bw, dec[-1][2] + bh / 2, 10.05, 5.4)       # d1 -> out
    # skip connections (dashed)
    for (ne, xe, ye), (nd, xd, yd) in zip(enc, reversed(dec)):
        arrow(ax, xe + bw, ye + bh / 2, xd, yd + bh / 2, style="-[", color=RED, lw=1.2)
    ax.text(5.5, 5.4, "Every block = Conv → tdBN(over T,B) → LIF (ATan surrogate, τ=2, detach-reset)",
            ha="center", fontsize=9, style="italic")
    ax.text(5.5, 0.4, "Direct encoding: input repeated over T timesteps → membrane potential averaged → logits;  "
                      "red dashed = skip connections", ha="center", fontsize=8.5, style="italic")
    ax.set_title("Figure 2. Spiking U-Net (4-level encoder–decoder)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_architecture.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/fig2_architecture.png")


# ---------- Figure 4: Decision tree ----------
def fig4():
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    box(ax, 3.5, 5.1, 3.0, 0.8, "Deploy target?", PURPLE, 10, "white")
    box(ax, 0.4, 3.3, 3.0, 0.9, "Neuromorphic HW\n(e.g., Loihi)?", BLUE, 9, "white")
    box(ax, 6.2, 3.3, 3.2, 0.9, "Standard edge GPU/\nMCU with INT8?", BLUE, 9, "white")
    box(ax, 0.2, 1.3, 3.0, 0.9, "Use Spiking U-Net\n(event-driven, low E)", GREEN, 9, "white")
    box(ax, 3.6, 1.3, 2.7, 0.9, "Energy budget\n< 5 mJ / chip?", YELLOW, 9)
    box(ax, 6.7, 1.3, 3.1, 0.9, "Use MobileNet-INT8\n(best acc/mJ)", RED, 9, "white")
    box(ax, 3.3, 0.0, 3.3, 0.9, "Consider SNN\n(accept small acc. drop, d≈0.24)", GREEN, 8.5, "white")
    arrow(ax, 4.5, 5.1, 2.2, 4.2); ax.text(3.1, 4.75, "—", fontsize=8)
    arrow(ax, 5.5, 5.1, 7.6, 4.2)
    arrow(ax, 1.9, 3.3, 1.7, 2.2); ax.text(1.4, 2.75, "yes", fontsize=8, color=GREEN)
    arrow(ax, 2.4, 3.3, 4.6, 2.2); ax.text(3.4, 2.8, "no", fontsize=8, color=RED)
    arrow(ax, 7.8, 3.3, 8.2, 2.2); ax.text(8.3, 2.75, "yes", fontsize=8, color=GREEN)
    arrow(ax, 7.2, 3.3, 5.4, 2.2); ax.text(6.0, 2.8, "no", fontsize=8, color=RED)
    arrow(ax, 4.7, 1.3, 4.8, 0.9); ax.text(5.0, 1.05, "yes", fontsize=8, color=GREEN)
    ax.set_title("Figure 4. Decision guide: Spiking U-Net vs INT8 MobileNet-UNet", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_decision_tree.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/fig4_decision_tree.png")


if __name__ == "__main__":
    fig1(); fig2(); fig4()
    print("XONG schematics.")
