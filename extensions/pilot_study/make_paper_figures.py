"""Generate paper-ready figures for the v4 T5 fine-tune + v6 downstream
contrastive pretraining + ReClor + held-out PARARULE experiments.

Outputs four PNGs under extensions/reports/figures/:
  fig1_t5_trajectory.png       — v1->v4 T5 self-check pass rate (subset + full)
  fig2_v6_cross_eval.png       — v5/v6 contrastive cross-eval 2x2 heatmap
  fig3_reclor_trajectory.png   — ReClor dev_acc across training steps
  fig4_heldout_pararule.png    — by-rule held-out pass rate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("extensions/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_t5_trajectory():
    versions = ["stock", "v1", "v2", "v3", "v4"]
    # Pass rate on the 15-failure subset (23 generator-tested items)
    subset = [8 / 23, 12 / 23, 13 / 23, 16 / 23, 17 / 23]
    subset_pct = [v * 100 for v in subset]
    # Pass rate on full 49-sentence pilot (90 items)
    full = [62 / 90, None, None, 67 / 90, 71 / 90]
    full_pct = [v * 100 if v is not None else None for v in full]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = list(range(len(versions)))
    ax.plot(x, subset_pct, marker="o", linewidth=2,
            label="15 known polarity-flips (n=23 gen-tested)", color="#1f77b4")
    full_x = [i for i, v in enumerate(full_pct) if v is not None]
    full_y = [v for v in full_pct if v is not None]
    ax.plot(full_x, full_y, marker="s", linewidth=2, linestyle="--",
            label="Full 49-sentence pilot (n=90)", color="#ff7f0e")

    for i, v in enumerate(subset_pct):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")
    for i, v in zip(full_x, full_y):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_xlabel("T5wtense fine-tune version")
    ax.set_ylabel("Self-check pass rate (%)")
    ax.set_title("T5wtense fine-tune trajectory — polarity-preservation pass rate")
    ax.set_ylim(30, 85)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    out = OUT_DIR / "fig1_t5_trajectory.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def fig2_v6_cross_eval():
    # Rows: trained on; Cols: evaluated on
    matrix = np.array([
        [99.31, 83.86],   # v5-trained
        [94.49, 98.43],   # v6-trained
    ])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=80, vmax=100, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v5 val (stock T5)", "v6 val (v4 T5)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["v5-trained", "v6-trained"])
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained on")
    ax.set_title("DeBERTa-large contrastive cross-eval (% accuracy)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="black")
    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    out = OUT_DIR / "fig2_v6_cross_eval.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def fig3_reclor_trajectory():
    steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1930]
    v5 = [31.6, 48.2, 53.0, 55.4, 58.4, 59.8, 60.8, 62.8, 62.8, 62.8]
    v6 = [38.6, 55.2, 54.8, 58.0, 58.8, 62.2, 61.8, 63.6, 62.8, 63.4]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, v5, marker="o", linewidth=2,
            label="v5 backbone (stock T5)", color="#1f77b4")
    ax.plot(steps, v6, marker="s", linewidth=2,
            label="v6 backbone (v4 T5)", color="#d62728")
    ax.axhline(62.8, color="#1f77b4", linestyle=":", alpha=0.5)
    ax.axhline(63.6, color="#d62728", linestyle=":", alpha=0.5)
    ax.annotate(f"v5 best 62.8%", xy=(1900, 62.8),
                xytext=(1600, 67), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax.annotate(f"v6 best 63.6%", xy=(1600, 63.6),
                xytext=(1100, 70), fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("Training step")
    ax.set_ylabel("ReClor dev accuracy (%)")
    ax.set_title("ReClor downstream fine-tune — v5 vs v6 backbone trajectory")
    ax.set_ylim(28, 72)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    out = OUT_DIR / "fig3_reclor_trajectory.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def fig4_heldout_pararule():
    rules = ["double_negation", "contraposition", "implication", "commutative"]
    stock = [49 / 60, 22 / 37, 21 / 37, 9 / 9]
    v4    = [55 / 60, 23 / 37, 18 / 37, 7 / 9]
    stock_pct = [v * 100 for v in stock]
    v4_pct = [v * 100 for v in v4]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    x = np.arange(len(rules))
    width = 0.36
    b1 = ax.bar(x - width / 2, stock_pct, width, label="stock T5", color="#7f7f7f")
    b2 = ax.bar(x + width / 2, v4_pct,    width, label="v4 T5",     color="#2ca02c")

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}%",
                        xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.set_ylabel("Self-check pass rate (%)")
    ax.set_title("Held-out PARARULE Depth5 (60 sentences) — by-rule pass rate")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    out = OUT_DIR / "fig4_heldout_pararule.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def main():
    fig1_t5_trajectory()
    fig2_v6_cross_eval()
    fig3_reclor_trajectory()
    fig4_heldout_pararule()


if __name__ == "__main__":
    main()
