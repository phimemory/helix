"""
Color Algebra Benchmark: Helix vs GRU
Task: Semantic algebra on colors — the model must learn that
  RED + YELLOW = ORANGE, ORANGE - RED = YELLOW, etc.
Tests if Helix can perform reversible arithmetic on a circular color wheel,
which is a naturally circular/geometric task perfectly suited to phase rotation.
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

# 8 colors on a wheel — each separated by 45 degrees (pi/4)
COLORS = ['RED', 'ORANGE', 'YELLOW', 'CHARTREUSE',
          'GREEN', 'TEAL', 'BLUE', 'VIOLET']
N_COLORS = len(COLORS)

def color_to_angle(c): return (2 * np.pi * c) / N_COLORS
def angle_to_color(a): return int(round(a * N_COLORS / (2 * np.pi))) % N_COLORS

def generate_algebra_data(n):
    """
    Generate: [color_A_onehot, op_onehot, color_B_onehot] → target_color_index
    op=0: ADD (A + B mod 8), op=1: SUBTRACT (A - B mod 8)
    """
    ca = torch.randint(0, N_COLORS, (n,))
    cb = torch.randint(0, N_COLORS, (n,))
    op = torch.randint(0, 2, (n,))

    y = torch.where(op == 0, (ca + cb) % N_COLORS, (ca - cb) % N_COLORS)

    # Encode as sequence: [ca_oh, op_oh, cb_oh] — 3-step sequence
    ca_oh = nn.functional.one_hot(ca, N_COLORS).float()  # [n, 8]
    cb_oh = nn.functional.one_hot(cb, N_COLORS).float()  # [n, 8]
    op_oh = nn.functional.one_hot(op, 2).float()          # [n, 2]

    # Pad to same width: input_size = N_COLORS = 8, op uses first 2 dims
    op_pad = torch.zeros(n, N_COLORS)
    op_pad[:, :2] = op_oh

    # Sequence: [ca, op, cb] → shape [n, 3, 8]
    x = torch.stack([ca_oh, op_pad, cb_oh], dim=1)
    return x.to(DEVICE), y.to(DEVICE)

class GRUModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(N_COLORS, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, N_COLORS)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

def train(model, is_helix, epochs=800, lr=0.001953125):
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = []
    X, Y   = generate_algebra_data(2000)
    Xt, Yt = generate_algebra_data(500)
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out = model(X)[0] if is_helix else model(X)
        loss = nn.CrossEntropyLoss()(out, Y)
        loss.backward(); opt.step()
        if (e + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vout = model(Xt)[0] if is_helix else model(Xt)
                acc = (vout.argmax(1) == Yt).float().mean().item()
                hist.append((e+1, acc))
                print(f"  {'Helix' if is_helix else 'GRU  '} | E{e+1:4d} | Acc: {acc:.1%}", flush=True)
            if acc == 1.0 and e > 100:
                print("  LOCKED!", flush=True); break
    return hist

def main():
    print("=" * 55)
    print("HELIX COLOR ALGEBRA BENCHMARK")
    print("Color wheel arithmetic: RED+YELLOW=ORANGE etc.")
    print("=" * 55, flush=True)

    helix = HelixModel(input_size=N_COLORS, hidden_size=128, output_size=N_COLORS,
                       num_layers=1, harmonics=[1, 2, 4, 8],
                       quantization_strength=0.125, persistence=1.0).to(DEVICE)
    gru   = GRUModel(hidden=512).to(DEVICE)

    hp = sum(p.numel() for p in helix.parameters())
    gp = sum(p.numel() for p in gru.parameters())
    print(f"\nHelix: {hp:,} params | GRU: {gp:,} params\n", flush=True)

    print("--- Training Helix ---", flush=True)
    h_hist = train(helix, is_helix=True, epochs=800)
    print("\n--- Training GRU ---", flush=True)
    g_hist = train(gru,   is_helix=False, epochs=800, lr=0.001)

    # Final eval on a held-out test
    Xt, Yt = generate_algebra_data(1000)
    helix.eval(); gru.eval()
    with torch.no_grad():
        h_acc = (helix(Xt)[0].argmax(1) == Yt).float().mean().item()
        g_acc = (gru(Xt).argmax(1) == Yt).float().mean().item()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0a0b10')
    for ax in axes: ax.set_facecolor('#0a0b10')

    # Panel A: Learning curves
    if h_hist:
        axes[0].plot([h[0] for h in h_hist], [h[1]*100 for h in h_hist],
                     color='#00ff88', lw=3, label=f'Helix (H=128, {hp:,}p)')
    if g_hist:
        axes[0].plot([g[0] for g in g_hist], [g[1]*100 for g in g_hist],
                     color='#4488ff', lw=2, alpha=0.8, label=f'GRU (H=512, {gp:,}p)')
    axes[0].set_title("Color Algebra Learning Curves", color='white', fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch", color='white'); axes[0].set_ylabel("Accuracy %", color='white')
    axes[0].set_ylim(0, 105); axes[0].grid(True, alpha=0.08)
    axes[0].legend(facecolor='#1a1b20', edgecolor='#333', labelcolor='white')
    axes[0].tick_params(colors='white')

    # Panel B: Color wheel visualization
    angles = np.linspace(0, 2*np.pi, N_COLORS, endpoint=False)
    wheel_colors = ['#ff2200', '#ff8800', '#ffee00', '#88ff00',
                    '#00cc44', '#00ccaa', '#0055ff', '#8800ff']
    ax = axes[1]
    for i, (ang, col, name) in enumerate(zip(angles, wheel_colors, COLORS)):
        ax.scatter(np.cos(ang), np.sin(ang), color=col, s=400, zorder=5)
        ax.annotate(name, (np.cos(ang)*1.25, np.sin(ang)*1.25),
                    ha='center', va='center', color=col, fontweight='bold', fontsize=9)
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#444', lw=1, ls='--', alpha=0.5)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal')
    ax.set_title(f"Color Wheel | Helix {h_acc:.1%} vs GRU {g_acc:.1%}",
                 color='white', fontsize=13, fontweight='bold')
    ax.text(0, -1.5, "Phase rotation natively encodes circular algebra",
            ha='center', color='#aaa', fontsize=9)
    ax.axis('off')

    plt.suptitle("Helix Color Algebra: Circular Semantic Arithmetic",
                 color='white', fontsize=15, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'color_algebra.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor='#0a0b10')
    print(f"\nFINAL | Helix {h_acc:.1%} | GRU {g_acc:.1%}")
    print(f"Saved: {out}", flush=True)

if __name__ == '__main__':
    main()
