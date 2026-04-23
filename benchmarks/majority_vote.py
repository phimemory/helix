"""
Majority Vote Benchmark: Helix (8 neurons) vs GRU (128 neurons)
Task: Given 8 sequential bits, output 1 if count(1s) > 4, else 0.
Helix solves counting/logic natively via circular arithmetic.
GRU needs 16x more neurons and still may fail.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from helix import HelixModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(n, seq_len=8):
    X = torch.randint(0, 2, (n, seq_len, 1)).float().to(DEVICE)
    Y = (X.sum(1) > 4).float().to(DEVICE)
    return X, Y

class GRUModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

def train(model, X, Y, Xt, Yt, is_helix=False, epochs=1000, lr=0.001953125):
    opt  = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for e in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(X)[0] if is_helix else model(X)
        loss = loss_fn(logits, Y)
        loss.backward(); opt.step()
        if (e + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(Xt)[0] if is_helix else model(Xt)
                acc = ((torch.sigmoid(vl) > 0.5).float() == Yt).float().mean().item()
                history.append((e + 1, acc))
                print(f"  {'Helix' if is_helix else 'GRU '} | E{e+1:4d} | Acc: {acc:.1%}", flush=True)
            if acc == 1.0 and e > 100:
                print(f"  LOCKED at epoch {e+1}", flush=True)
                break
    return history

def main():
    print("=" * 55)
    print("HELIX MAJORITY VOTE BENCHMARK")
    print("8 neurons vs 128 GRU neurons | 8-bit majority")
    print("=" * 55, flush=True)

    X, Y   = generate_data(2000)
    Xt, Yt = generate_data(1000)

    helix = HelixModel(input_size=1, hidden_size=8, output_size=1,
                       num_layers=1, harmonics=[1, 2, 4, 8],
                       quantization_strength=0.125, persistence=1.0).to(DEVICE)
    gru   = GRUModel(hidden=128).to(DEVICE)

    hp = sum(p.numel() for p in helix.parameters())
    gp = sum(p.numel() for p in gru.parameters())
    print(f"\nHelix: {hp:,} params (8 neurons)")
    print(f"GRU:   {gp:,} params (128 neurons)\n", flush=True)

    print("--- Training Helix ---", flush=True)
    h_hist = train(helix, X, Y, Xt, Yt, is_helix=True, epochs=1000)
    print("\n--- Training GRU ---", flush=True)
    g_hist = train(gru,   X, Y, Xt, Yt, is_helix=False, epochs=1000)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0b10'); ax.set_facecolor('#0a0b10')

    if h_hist:
        ax.plot([h[0] for h in h_hist], [h[1]*100 for h in h_hist],
                color='#00ff88', lw=3, marker='s', markersize=4,
                label=f'Helix (8 neurons, {hp:,} params)')
    if g_hist:
        ax.plot([g[0] for g in g_hist], [g[1]*100 for g in g_hist],
                color='#4488ff', lw=2, marker='o', markersize=3, alpha=0.8,
                label=f'GRU (128 neurons, {gp:,} params)')

    ax.axhline(50, color='#ffaa00', ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Accuracy %', color='white', fontsize=12)
    ax.set_title('8-Bit Majority Vote: Helix (8 neurons) vs GRU (128 neurons)',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1a1b20', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white'); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.08, color='white')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#333')
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'majority_vote_benchmark.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor='#0a0b10')
    print(f"\nSaved to {out}", flush=True)

    h_final = h_hist[-1][1] * 100 if h_hist else 0
    g_final = g_hist[-1][1] * 100 if g_hist else 0
    print(f"\nFINAL: Helix {h_final:.1f}% | GRU {g_final:.1f}%", flush=True)

if __name__ == '__main__':
    main()
