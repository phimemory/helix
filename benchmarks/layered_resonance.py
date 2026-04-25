"""
Helix Layered Resonance Benchmark
Ported from ROUND Prism Stack test.

Tests whether Helix can use one piece of information to change how it
interprets another — multi-step conditional logic via stacked geometric rotations.

ROUND result: solved via physical adjustment of internal focus on the
information circle (stacked phase rotations).

This benchmark validates:
1. Helix can encode conditional context (IF this THEN read that differently)
2. Stacking works: each layer adds a phase rotation that modulates the next
3. Helix vs GRU on hierarchical reasoning tasks
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helix_neuron import HelixModel, HARMONICS_STANDARD

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def generate_prism_data(n_samples=1000, seq_len=8, context_dim=4, input_dim=4):
    """
    Layered Resonance task:
    - First token is a "context key" (0-3)
    - Remaining tokens are data
    - Correct output depends on context key
    - e.g., key=0: sum all, key=1: sum odd positions, key=2: XOR, key=3: max

    This requires the network to:
    1. Remember the context key throughout the sequence
    2. Apply a different transformation based on that key
    """
    inputs = []
    targets = []

    for _ in range(n_samples):
        context_key = torch.randint(0, 4, (1,)).item()
        context_vec = torch.zeros(context_dim + input_dim)
        context_vec[context_key] = 1.0  # one-hot context

        data = torch.randint(0, 2, (seq_len - 1, input_dim)).float()
        data_seq = [context_vec]

        for t in range(seq_len - 1):
            token = torch.zeros(context_dim + input_dim)
            token[context_dim:] = data[t]
            data_seq.append(token)

        seq = torch.stack(data_seq)

        # Target depends on context key
        if context_key == 0:
            target = (data.sum() > seq_len).float().unsqueeze(0)
        elif context_key == 1:
            target = (data[::2].sum() > data[1::2].sum()).float().unsqueeze(0)
        elif context_key == 2:
            target = (data.long().sum(dim=0) % 2).float().mean().unsqueeze(0)
        else:
            target = (data.max() > 0.5).float().unsqueeze(0)

        inputs.append(seq)
        targets.append(target)

    return torch.stack(inputs), torch.stack(targets)


def run_layered_resonance_benchmark(
    epochs=400,
    lr=2**-9,
    hidden_size=32,
    verbose=True
):
    print("=" * 60)
    print("HELIX LAYERED RESONANCE BENCHMARK")
    print(f"  hidden={hidden_size} | epochs={epochs}")
    print("=" * 60)

    feature_dim = 8  # context_dim + input_dim

    X_train, y_train = generate_prism_data(1000, seq_len=8,
                                            context_dim=4, input_dim=4)
    X_test, y_test   = generate_prism_data(200, seq_len=8,
                                            context_dim=4, input_dim=4)

    def train_and_eval(model_class, model_name, **kwargs):
        if model_class == "helix":
            model = HelixModel(
                input_size=feature_dim,
                hidden_size=hidden_size,
                output_size=1,
                harmonics=HARMONICS_STANDARD,
                **kwargs
            )
        else:
            # GRU baseline with 4x capacity
            gru = nn.GRU(feature_dim, hidden_size * 4,
                         batch_first=True, num_layers=1)
            readout = nn.Linear(hidden_size * 4, 1)
            model = nn.Sequential()
            model.gru = gru
            model.readout = readout

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_accs, test_accs = [], []

        for epoch in range(epochs):
            optimizer.zero_grad()

            if model_class == "helix":
                preds, _ = model(X_train)
            else:
                out, _ = model.gru(X_train)
                preds = model.readout(out[:, -1, :])

            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                with torch.no_grad():
                    if model_class == "helix":
                        train_pred, _ = model(X_train)
                        test_pred, _  = model(X_test)
                    else:
                        out_tr, _ = model.gru(X_train)
                        train_pred = model.readout(out_tr[:, -1, :])
                        out_te, _ = model.gru(X_test)
                        test_pred = model.readout(out_te[:, -1, :])

                    tr_acc = ((train_pred > 0) == y_train.bool()).float().mean().item()
                    te_acc = ((test_pred > 0) == y_test.bool()).float().mean().item()
                    train_accs.append((epoch, tr_acc))
                    test_accs.append((epoch, te_acc))

                if verbose and epoch % 100 == 0:
                    print(f"  {model_name} E{epoch:3d}: train={tr_acc:.3f} test={te_acc:.3f}")

        return train_accs, test_accs

    print("\n--- Training Helix ---")
    helix_tr, helix_te = train_and_eval("helix", "Helix")

    print("\n--- Training GRU (4x capacity) ---")
    gru_tr, gru_te = train_and_eval("gru", "GRU")

    final_helix_acc = helix_te[-1][1]
    final_gru_acc   = gru_te[-1][1]
    print(f"\n  FINAL: Helix {final_helix_acc*100:.1f}% | GRU {final_gru_acc*100:.1f}%")

    _plot_results(helix_tr, helix_te, gru_tr, gru_te)

    return {
        "helix_accuracy": final_helix_acc,
        "gru_accuracy": final_gru_acc,
        "helix_params": hidden_size * 3,
        "gru_params": hidden_size * 4,
    }


def _plot_results(helix_tr, helix_te, gru_tr, gru_te):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0A0B10")
    for ax in axes:
        ax.set_facecolor("#0A0B10")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Panel A: Helix accuracy over time
    ax = axes[0]
    epochs_h = [x[0] for x in helix_te]
    accs_h   = [x[1] for x in helix_te]
    ax.plot(epochs_h, accs_h, color="forestgreen", linewidth=2, label="Helix")
    ax.axhline(1.0, color="white", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Test Accuracy", color="white")
    ax.set_title("Helix: Layered Resonance", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#0A0B10", labelcolor="white")

    # Panel B: GRU accuracy over time
    ax = axes[1]
    epochs_g = [x[0] for x in gru_te]
    accs_g   = [x[1] for x in gru_te]
    ax.plot(epochs_g, accs_g, color="maroon", linewidth=2, label="GRU (4x params)")
    ax.axhline(1.0, color="white", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Test Accuracy", color="white")
    ax.set_title("GRU Baseline: Layered Resonance", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#0A0B10", labelcolor="white")

    # Panel C: Final comparison
    ax = axes[2]
    final_h = helix_te[-1][1] * 100
    final_g = gru_te[-1][1] * 100
    bars = ax.bar(["Helix", "GRU (4x)"], [final_h, final_g],
                  color=["forestgreen", "maroon"], width=0.4)
    for bar, val in zip(bars, [final_h, final_g]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", color="white")
    ax.set_title("Final Accuracy: Helix vs GRU", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 110)

    plt.suptitle("Helix Layered Resonance Benchmark", color="white", fontsize=14)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "layered_resonance_benchmark.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0A0B10")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    run_layered_resonance_benchmark()
