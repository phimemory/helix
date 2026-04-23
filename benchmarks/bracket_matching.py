"""
Bracket Matching Benchmark: Helix vs GRU
Task: Given a sequence of brackets like (()()), determine if it is balanced.
This tests the model's ability to maintain a running COUNT as a stack
across arbitrarily long sequences — a classic long-range dependency task.
Helix naturally tracks running parity/count via phase accumulation.
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
SEQ_LEN = 20  # variable-length sequences up to this length

def generate_bracket_data(n, max_len=20):
    """
    Generate bracket sequences and balanced labels.
    Token: 0=open '(', 1=close ')', 2=pad
    Label: 1 if balanced, 0 if not
    """
    seqs = []; labels = []
    for _ in range(n):
        length = np.random.randint(2, max_len + 1)
        if length % 2 == 1: length -= 1  # must be even
        # Random valid or invalid sequence
        seq = np.random.randint(0, 2, length)
        depth = 0; valid = True
        for t in seq:
            depth += 1 if t == 0 else -1
            if depth < 0: valid = False; break
        if depth != 0: valid = False
        # Pad to max_len
        padded = np.full(max_len, 2, dtype=np.float32)
        padded[:length] = seq
        seqs.append(padded)
        labels.append(float(valid))

    x = torch.tensor(np.array(seqs)).unsqueeze(-1).to(DEVICE)  # [n, max_len, 1]
    y = torch.tensor(labels).unsqueeze(-1).to(DEVICE)           # [n, 1]
    return x, y

class GRUModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

def train(model, is_helix, epochs=400, lr=0.001953125):
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = []
    X, Y   = generate_bracket_data(2000)
    Xt, Yt = generate_bracket_data(500)
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out = model(X)[0] if is_helix else model(X)
        loss = nn.BCEWithLogitsLoss()(out, Y)
        loss.backward(); opt.step()
        if (e + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                vout = model(Xt)[0] if is_helix else model(Xt)
                acc = ((torch.sigmoid(vout) > 0.5) == Yt).float().mean().item()
                hist.append((e+1, acc))
                print(f"  {'Helix' if is_helix else 'GRU  '} | E{e+1:3d} | Acc: {acc:.1%}", flush=True)
            if acc >= 0.99 and e > 50: print("  CONVERGED"); break
    return hist

def main():
    print("=" * 55)
    print("HELIX BRACKET MATCHING BENCHMARK")
    print(f"Balanced brackets up to length {SEQ_LEN}")
    print("=" * 55, flush=True)

    helix = HelixModel(input_size=1, hidden_size=32, output_size=1,
                       num_layers=1, harmonics=[1],
                       quantization_strength=0.125, persistence=1.0).to(DEVICE)
    gru   = GRUModel(hidden=128).to(DEVICE)

    hp = sum(p.numel() for p in helix.parameters())
    gp = sum(p.numel() for p in gru.parameters())
    print(f"\nHelix: {hp:,} params | GRU: {gp:,} params\n", flush=True)

    print("--- Training Helix ---", flush=True)
    h_hist = train(helix, is_helix=True)
    print("\n--- Training GRU ---", flush=True)
    g_hist = train(gru,   is_helix=False, lr=0.001)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0b10'); ax.set_facecolor('#0a0b10')

    if h_hist:
        ax.plot([h[0] for h in h_hist], [h[1]*100 for h in h_hist],
                color='#00ff88', lw=3, marker='s', markersize=4,
                label=f'Helix (H=32, {hp:,} params)')
    if g_hist:
        ax.plot([g[0] for g in g_hist], [g[1]*100 for g in g_hist],
                color='#4488ff', lw=2, marker='o', markersize=3, alpha=0.8,
                label=f'GRU (H=128, {gp:,} params)')

    ax.axhline(50, color='#ffaa00', ls=':', alpha=0.3, lw=1)
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Accuracy %', color='white', fontsize=12)
    ax.set_title(f'Bracket Matching (len≤{SEQ_LEN}): Helix vs GRU',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1a1b20', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white'); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.08, color='white')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#333')
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'bracket_matching.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor='#0a0b10')
    h_f = h_hist[-1][1]*100 if h_hist else 0
    g_f = g_hist[-1][1]*100 if g_hist else 0
    print(f"\nFINAL | Helix {h_f:.1f}% | GRU {g_f:.1f}%")
    print(f"Saved: {out}", flush=True)

if __name__ == '__main__':
    main()
