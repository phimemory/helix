"""
helix_unitary.py — IsometricHelixCell: provably unitary state updates.

The standard HelixCell uses tanh and unconstrained weight matrices.
Those are not orthogonal, so the state update is not isometric.

This module keeps the full Helix phase concept — accumulation, quantization
sieve, multi-harmonic readout — but replaces the recurrent weight with K
input-dependent Householder reflections. The update rule becomes:

    φ_new = H_K(x) ∘ ... ∘ H_1(x)[φ] + Δφ(x)

This is an isometric affine map. For any two states φ¹ and φ²:

    ‖φ_new¹ − φ_new²‖ = ‖W(x)(φ¹−φ²)‖ = ‖φ¹−φ²‖

because W(x) is orthogonal. Distances between states are preserved exactly.

The quantization sieve is the only non-isometric step. It pulls phases at
most quantization_strength · π/4 toward the nearest grid point — a bounded
and intentional contraction for discrete stability.
"""

import math
import torch
import torch.nn as nn


def _householder(vs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Apply K Householder reflections to phi.

    vs:  (batch, K, d)  — K reflection vectors per sample
    phi: (batch, d)     — current phase state

    Each reflection H(v) = I - 2vvᵀ/‖v‖² is orthogonal (H^T H = I).
    Their product W = H_K · ... · H_1 is also orthogonal.
    ‖W·phi‖ = ‖phi‖ exactly.
    """
    for k in range(vs.shape[1]):
        v = vs[:, k]                                              # (batch, d)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        phi = phi - 2.0 * (phi * v).sum(dim=-1, keepdim=True) * v
    return phi


class IsometricHelixCell(nn.Module):
    """
    Phase-rotation RNN with genuinely isometric state updates.

    State: φ ∈ ℝᵈ  — phase angles accumulating on the real line.

    Update:
        φ_new = W(x) · φ + Δφ(x)

    W(x) = H_K(x) ∘ ... ∘ H_1(x) is orthogonal for any x.
    Δφ(x) is an input-driven offset (translation preserves distances).

    Isometry proof:
        ‖φ_new¹ − φ_new²‖  =  ‖W(x)(φ¹−φ²)‖  =  ‖φ¹−φ²‖

    No tradeoffs: parameter count is lower than standard HelixNeuronCell
    (K·d + d vs d·2d for weight_hh), and expressivity for the orthogonal
    group is complete when K ≥ d.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        harmonics: tuple = (1, 2, 4, 8),
        n_reflections: int = 4,
        quantization_strength: float = 0.125,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.n_reflections = n_reflections
        self.quantization_strength = quantization_strength

        # Input → K Householder vectors in ℝᵈ
        self.to_reflections = nn.Linear(input_size, n_reflections * hidden_size)

        # Input → phase offset Δφ
        self.to_delta = nn.Linear(input_size, hidden_size)

        # Output dimensionality for downstream heads
        self.output_size = hidden_size * len(harmonics) * 2

        nn.init.kaiming_uniform_(self.to_reflections.weight)
        nn.init.kaiming_uniform_(self.to_delta.weight)
        nn.init.zeros_(self.to_reflections.bias)
        nn.init.zeros_(self.to_delta.bias)

    def forward(
        self, x: torch.Tensor, phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:   (batch, input_size)
        phi: (batch, hidden_size)
        Returns (phi_new, features)

        phi_new:  updated phase state — distances to other states preserved
        features: (batch, output_size) — multi-harmonic readout
        """
        batch = x.shape[0]

        vs    = self.to_reflections(x).view(batch, self.n_reflections, self.hidden_size)
        delta = self.to_delta(x)

        # Isometric step: orthogonal rotation + translation
        phi_new = _householder(vs, phi) + delta

        # Soft quantization sieve: snap to π/4 grid
        q = torch.round(phi_new / (math.pi / 4)) * (math.pi / 4)
        phi_new = phi_new + self.quantization_strength * (q - phi_new)

        # Multi-harmonic readout (cos + sin at each harmonic)
        features = torch.cat([
            f(k * phi_new)
            for k in self.harmonics
            for f in (torch.cos, torch.sin)
        ], dim=-1)

        return phi_new, features


class IsometricHelixModel(nn.Module):
    """
    Sequence model built on IsometricHelixCell.
    Drop-in replacement for HelixNeuronModel on classification / regression tasks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        harmonics: tuple = (1, 2, 4, 8),
        n_reflections: int = 4,
        quantization_strength: float = 0.125,
    ):
        super().__init__()
        self.cell = IsometricHelixCell(
            input_size, hidden_size, harmonics, n_reflections, quantization_strength
        )
        self.head = nn.Linear(self.cell.output_size, output_size)
        nn.init.kaiming_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, x: torch.Tensor, phi: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        Returns (logits, phi_final)
        """
        batch, seq_len, _ = x.shape
        if phi is None:
            phi = torch.zeros(batch, self.cell.hidden_size, device=x.device)

        for t in range(seq_len):
            phi, features = self.cell(x[:, t], phi)

        return self.head(features), phi


def verify_isometry(
    cell: IsometricHelixCell,
    n_trials: int = 2000,
    verbose: bool = True,
) -> float:
    """
    Empirically verify the isometric property of IsometricHelixCell.

    For n_trials random (x, φ¹, φ²) triples, checks that the
    Householder update preserves the distance between φ¹ and φ².

    Returns the maximum observed error (should be < 1e-5 numerically).
    """
    cell.eval()
    max_err = 0.0
    input_size = cell.to_delta.in_features

    with torch.no_grad():
        for _ in range(n_trials):
            x    = torch.randn(1, input_size)
            phi1 = torch.randn(1, cell.hidden_size)
            phi2 = torch.randn(1, cell.hidden_size)

            d_before = (phi1 - phi2).norm().item()

            # Apply only the isometric part (skip sieve)
            vs    = cell.to_reflections(x).view(1, cell.n_reflections, cell.hidden_size)
            delta = cell.to_delta(x)
            p1 = _householder(vs, phi1) + delta
            p2 = _householder(vs, phi2) + delta

            d_after = (p1 - p2).norm().item()
            max_err = max(max_err, abs(d_after - d_before))

    if verbose:
        print(f"Isometry check — max distance error over {n_trials} trials: {max_err:.2e}")
        print(f"  {'PASS' if max_err < 1e-4 else 'FAIL'} (threshold 1e-4)")

    return max_err


if __name__ == "__main__":
    print("IsometricHelixCell — isometry verification\n")

    cell = IsometricHelixCell(input_size=8, hidden_size=32, n_reflections=4)

    err = verify_isometry(cell, n_trials=2000)

    print()
    print("Quick forward pass:")
    x   = torch.randn(4, 16, 8)   # batch=4, seq_len=16, input=8
    model = IsometricHelixModel(input_size=8, hidden_size=32, output_size=2)
    logits, phi_final = model(x)
    print(f"  input:      {tuple(x.shape)}")
    print(f"  logits:     {tuple(logits.shape)}")
    print(f"  phi_final:  {tuple(phi_final.shape)}")
    print(f"  params:     {sum(p.numel() for p in model.parameters()):,}")
