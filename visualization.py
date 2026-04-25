"""
Helix Scientific Visualization Suite
Ported from ROUND visualization_utils.py with ROUND's 3-panel dark mode standard.

Provides standardized 3-panel diagnostic plots for all Helix benchmarks:
  Panel A: Persistence Map   — heatmap of crystallization over training
  Panel B: Survival Rate     — per-neuron lock retention over epochs
  Panel C: Hypertorus Projection — phase state mapped to 2D manifold

Used as the standard output format for all benchmark scripts.
"""

import torch
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# Dark mode palette (ROUND "Forest Wall" standard)
DARK_BG    = "#0A0B10"
COLOR_HIT  = "forestgreen"
COLOR_MISS = "maroon"
COLOR_HELIX = "#4fc3f7"
COLOR_GRU  = "#ef5350"
COLOR_WHITE = "white"
COLOR_DIM  = "#555566"


def helix_figure(title="Helix Benchmark", figsize=(15, 5)):
    """Create a dark-mode figure with standard Helix styling."""
    fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
    fig.suptitle(title, color=COLOR_WHITE, fontsize=14, y=1.01)
    return fig


def style_axis(ax):
    """Apply standard Helix dark-mode styling to an axis."""
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.tick_params(colors=COLOR_WHITE)
    ax.xaxis.label.set_color(COLOR_WHITE)
    ax.yaxis.label.set_color(COLOR_WHITE)
    ax.title.set_color(COLOR_WHITE)
    return ax


def three_panel_diagnostic(
    persistence_data,
    survival_data,
    phase_coords,
    title="Helix Diagnostic",
    save_path=None
):
    """
    Standard 3-panel diagnostic for Helix benchmarks.

    Args:
        persistence_data: 2D array (epochs x neurons) of per-neuron accuracy
                          Used for Panel A persistence heatmap
        survival_data:    1D or 2D array of lock retention rates over epochs
                          Used for Panel B survival rate curve
        phase_coords:     List of (h_cos, h_sin) tuples from HelixModel.forward
                          Used for Panel C hypertorus projection
        title:            Figure title
        save_path:        If provided, save to this path
    """
    fig = helix_figure(title, figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel A: Persistence Heatmap ──────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    style_axis(ax_a)
    ax_a.set_title("Panel A: Persistence Map", color=COLOR_WHITE, fontsize=10)

    if persistence_data is not None:
        data = np.array(persistence_data)
        # Forest Wall palette: green = crystallized, red = failed
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "forest_wall", [COLOR_MISS, "#222", COLOR_HIT]
        )
        im = ax_a.imshow(
            data.T, aspect="auto", cmap=cmap, vmin=0, vmax=1,
            origin="lower", interpolation="nearest"
        )
        ax_a.set_xlabel("Epoch", color=COLOR_WHITE)
        ax_a.set_ylabel("Neuron", color=COLOR_WHITE)
        plt.colorbar(im, ax=ax_a, label="Accuracy")
    else:
        ax_a.text(0.5, 0.5, "No data", ha="center", va="center",
                  color=COLOR_DIM, transform=ax_a.transAxes)

    # ── Panel B: Survival Rate ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    style_axis(ax_b)
    ax_b.set_title("Panel B: Survival Rate", color=COLOR_WHITE, fontsize=10)

    if survival_data is not None:
        data = np.array(survival_data)
        if data.ndim == 1:
            ax_b.plot(data, color=COLOR_HELIX, linewidth=2, label="Helix")
        else:
            for i, row in enumerate(data):
                ax_b.plot(row, alpha=0.7, linewidth=1)
        ax_b.set_xlabel("Epoch", color=COLOR_WHITE)
        ax_b.set_ylabel("Locked Neurons (%)", color=COLOR_WHITE)
        ax_b.set_ylim(0, 1.05)
        ax_b.axhline(1.0, color=COLOR_WHITE, linestyle="--", alpha=0.3)
    else:
        ax_b.text(0.5, 0.5, "No data", ha="center", va="center",
                  color=COLOR_DIM, transform=ax_b.transAxes)

    # ── Panel C: Hypertorus Projection ─────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    style_axis(ax_c)
    ax_c.set_title("Panel C: Phase Manifold (2D)", color=COLOR_WHITE, fontsize=10)

    if phase_coords is not None and len(phase_coords) > 0:
        # Project all (h_cos, h_sin) pairs to 2D via PCA-equivalent
        cos_vals = torch.cat([c[0].flatten() for c in phase_coords]).numpy()
        sin_vals = torch.cat([c[1].flatten() for c in phase_coords]).numpy()

        # Plot as scatter on the unit circle manifold
        # Normalize to fit within Logic Plane scaling
        cos_n = cos_vals / (np.abs(cos_vals).max() + 1e-8)
        sin_n = sin_vals / (np.abs(sin_vals).max() + 1e-8)

        n_points = min(len(cos_n), 2000)
        idx = np.random.choice(len(cos_n), n_points, replace=False)
        sc = ax_c.scatter(
            cos_n[idx], sin_n[idx],
            c=np.arange(n_points), cmap="viridis",
            s=3, alpha=0.6
        )
        # Draw unit circle reference
        theta = np.linspace(0, 2 * np.pi, 100)
        ax_c.plot(np.cos(theta), np.sin(theta), color=COLOR_DIM,
                  linewidth=0.5, linestyle="--")
        ax_c.set_xlabel("h_cos", color=COLOR_WHITE)
        ax_c.set_ylabel("h_sin", color=COLOR_WHITE)
        ax_c.set_aspect("equal")
        ax_c.set_xlim(-1.2, 1.2)
        ax_c.set_ylim(-1.2, 1.2)
    else:
        ax_c.text(0.5, 0.5, "No phase data", ha="center", va="center",
                  color=COLOR_DIM, transform=ax_c.transAxes)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def accuracy_vs_epochs(
    helix_accs,
    gru_accs=None,
    title="Accuracy vs Epochs",
    save_path=None,
    ylabel="Accuracy (%)"
):
    """Standard H2H accuracy comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=DARK_BG)
    style_axis(ax)

    epochs_h = [x[0] for x in helix_accs]
    accs_h   = [x[1] * 100 for x in helix_accs]
    ax.plot(epochs_h, accs_h, color=COLOR_HELIX, linewidth=2, label="Helix")

    if gru_accs is not None:
        epochs_g = [x[0] for x in gru_accs]
        accs_g   = [x[1] * 100 for x in gru_accs]
        ax.plot(epochs_g, accs_g, color=COLOR_GRU, linewidth=2,
                label="GRU", linestyle="--")

    ax.axhline(100, color=COLOR_WHITE, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=COLOR_WHITE)
    ax.set_ylim(0, 110)
    ax.legend(facecolor=DARK_BG, labelcolor=COLOR_WHITE)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Saved: {save_path}")
    plt.close()


def bar_comparison(
    helix_val, gru_val,
    title="Helix vs GRU",
    metric_label="Score (%)",
    save_path=None
):
    """Standard final score bar chart."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    style_axis(ax)

    bars = ax.bar(
        ["Helix", "GRU"],
        [helix_val, gru_val],
        color=[COLOR_HIT, COLOR_MISS],
        width=0.4
    )
    for bar, val in zip(bars, [helix_val, gru_val]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", color=COLOR_WHITE,
            fontsize=13, fontweight="bold"
        )

    ax.set_ylabel(metric_label)
    ax.set_title(title, color=COLOR_WHITE)
    ax.set_ylim(0, 115)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Saved: {save_path}")
    plt.close()
