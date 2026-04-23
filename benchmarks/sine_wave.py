"""
Sine Wave Benchmark: tests continuous signal tracking.
Helix vs GRU on predicting sine waves of varying frequencies.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from helix import HelixModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SineModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.helix = HelixModel(
            input_size=1, hidden_size=hidden_size, output_size=1,
            num_layers=1, harmonics=[1, 2, 4, 8],
            quantization_strength=0.125, persistence=0.95
        )

    def forward(self, x):
        out, conf = self.helix(x, return_sequence=True)
        return out, conf


class GRUSineModel(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)


def generate_sine_data(n_samples, seq_len=100):
    freqs = torch.rand(n_samples, 1, 1) * 3 + 0.5
    t = torch.linspace(0, 4 * np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
    t = t.expand(n_samples, -1, -1)
    y = torch.sin(freqs * t)
    x = y[:, :-1, :]
    target = y[:, 1:, :]
    return x, target


def train_model(model, name, is_helix=False, epochs=300, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        x, y = generate_sine_data(64, seq_len=100)
        optimizer.zero_grad()
        if is_helix:
            pred, _ = model(x)
        else:
            pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x_test, y_test = generate_sine_data(200, seq_len=100)
                if is_helix:
                    pred_test, _ = model(x_test)
                else:
                    pred_test = model(x_test)
                mse = criterion(pred_test, y_test).item()
                history.append((epoch + 1, mse))
                print(f"  {name} | Epoch {epoch+1:3d} | MSE: {mse:.6f}", flush=True)

    return history


def main():
    print("=" * 50)
    print("HELIX SINE WAVE BENCHMARK")
    print("continuous signal tracking: helix vs gru")
    print("=" * 50, flush=True)

    helix_model = SineModel(hidden_size=32)
    gru_model = GRUSineModel(hidden_size=256)

    h_params = sum(p.numel() for p in helix_model.parameters())
    g_params = sum(p.numel() for p in gru_model.parameters())
    print(f"Helix: {h_params} params (32 hidden)")
    print(f"GRU:   {g_params} params (256 hidden)\n", flush=True)

    print("--- Training Helix ---", flush=True)
    h_hist = train_model(helix_model, "Helix", is_helix=True, epochs=300)

    print("\n--- Training GRU ---", flush=True)
    g_hist = train_model(gru_model, "GRU", is_helix=False, epochs=300)

    # generate chart
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0b10')
    ax.set_facecolor('#0a0b10')

    h_epochs = [h[0] for h in h_hist]
    h_mse = [h[1] for h in h_hist]
    g_epochs = [g[0] for g in g_hist]
    g_mse = [g[1] for g in g_hist]

    ax.semilogy(h_epochs, h_mse, color='#00ff88', linewidth=3, marker='s',
                markersize=4, label=f'Helix ({h_params} params)')
    ax.semilogy(g_epochs, g_mse, color='#ff4444', linewidth=2, marker='o',
                markersize=3, label=f'GRU ({g_params} params)', alpha=0.8)

    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('MSE (log scale)', color='white', fontsize=12)
    ax.set_title('Sine Wave Tracking: Helix vs GRU',
                 color='white', fontsize=14, fontweight='bold', pad=15)
    ax.legend(facecolor='#1a1b20', edgecolor='#333', labelcolor='white', fontsize=10)
    ax.tick_params(colors='white')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')
    ax.grid(True, alpha=0.08, color='white')
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'sine_benchmark.png')
    plt.savefig(out_path, dpi=150, facecolor='#0a0b10')
    print(f"\nChart saved to {out_path}")

    print(f"\n{'='*50}")
    print(f"FINAL: Helix MSE {h_mse[-1]:.6f} | GRU MSE {g_mse[-1]:.6f}")
    print(f"{'='*50}", flush=True)


if __name__ == '__main__':
    main()
