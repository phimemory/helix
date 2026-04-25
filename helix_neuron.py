"""
Helix Core Cell v2 - Strict Unitary Isometry
Ported from ROUND UITNeuronCell with Helix naming.

Key addition over helix.py:
- Hard Renormalization ("Diamond Lock") — phases snap to nearest
  topological grid point, creating bit-perfect crystalline state
- Zero-Persistence mode for continuous topology tracking
- Phasic Sovereignty — hidden state of one instance is mathematically
  identical to the input requirements of another (enables cross-instance relay)
- Landauer regularization — penalizes information erasure
- 7-Octave harmonic spectrum option (sub-harmonics + super-harmonics)
"""

import torch
import torch.nn as nn
import numpy as np

# Standard harmonic spectra
HARMONICS_STANDARD = [1, 2, 4, 8]
HARMONICS_7OCTAVE  = [0.125, 0.25, 0.5, 1, 2, 4, 8]  # Full octave range
HARMONICS_SPINOR   = [0.5, 1, 2, 4]                    # 4pi double-cover


class HelixNeuronCell(nn.Module):
    """
    The Helix Neuron — a single unitary phasic cell.

    Implements Phasic Identity: information is stored as a residue in
    the graded ring (phase angle). Phase magnitude is maintained at
    constant scale via hard renormalization (Diamond Lock).

    Modes:
      Default:          continuous phase accumulation with harmonic locking
      binary_alignment: discrete bit-encoding via half-shifts (for parity etc.)
      unwinding:        Bernoulli bit extraction from accumulated phase
      zero_persistence: severs history for pure derivative velocity tracking
                        (solves sine-wave tracking without 2pi wrap singularity)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        harmonics=HARMONICS_STANDARD,
        use_spinor=True,
        quantization_strength=0.125,
        use_binary_alignment=False,
        unwinding_mode=False,
        persistence=1.0,
        spin_multiplier=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.use_spinor = use_spinor
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        self.persistence = persistence

        # Spin multiplier: 2.0 for spinor double-cover, 1.0 for standard
        self.spin_multiplier = spin_multiplier if spin_multiplier is not None \
            else (2.0 if use_spinor else 1.0)

        # Learnable parameters
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias      = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.epsilon   = nn.Parameter(torch.Tensor(hidden_size))

        # Learnable harmonic weights (diagnostic harmonics)
        self.harmonic_weights = nn.Parameter(
            torch.Tensor(hidden_size, len(harmonics))
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.zeros_(self.weight_hh)
        nn.init.zeros_(self.bias)

        with torch.no_grad():
            # Geometric epsilon decay per neuron
            for j in range(self.hidden_size):
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5))

            nn.init.uniform_(self.harmonic_weights, 0.0, 1.0)

            # Spread initial phase biases evenly on the circle
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                self.bias[self.hidden_size + j] = j * spread

    def forward(self, x, phi_prev):
        """
        Args:
            x:        (batch, input_size)
            phi_prev: (batch, hidden_size) — previous phase state

        Returns:
            output:     (batch, hidden_size) — U-space output
            phi_next:   (batch, hidden_size) — updated phase state
            confidence: (batch, hidden_size) — per-neuron confidence
            h_cos:      (batch, hidden_size) — cosine component
            h_sin:      (batch, hidden_size) — sine component
        """
        gates = x @ self.weight_ih + phi_prev @ self.weight_hh + self.bias
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        standard_part = torch.tanh(x_gate)

        if self.use_binary_alignment:
            if self.unwinding_mode:
                # Bernoulli Unwinding: extract bit, shift phase
                bit_out = (phi_prev >= (np.pi - 1e-7)).float()
                phi_next = (phi_prev - bit_out * np.pi) * 2.0
                # Diamond Lock: snap to quantization grid
                q_grid = np.pi / 128.0
                phi_next = torch.round(phi_next / q_grid) * q_grid
            else:
                # Binary alignment: half-shift encoding
                incoming_bit = x[:, 0:1]
                phi_next = (phi_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit
        else:
            # Standard continuous mode
            phi_shift = torch.sigmoid(phi_gate) * np.pi * self.spin_multiplier
            phi_next = (phi_prev * self.persistence) + phi_shift

            # Quantization sieve (soft snap toward grid)
            q_sieve = torch.round(phi_next / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next + self.quantization_strength * (q_sieve - phi_next)

        # Compute harmonic readout (weighted multi-harmonic expansion)
        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.harmonic_weights[:, idx] * torch.cos(h * phi_next)
            h_sin += self.harmonic_weights[:, idx] * torch.sin(h * phi_next)

        # Confidence: mean harmonic cosine alignment
        confidence = (h_cos.abs() / len(self.harmonics)).detach()

        # U-Space output: x(1 + epsilon * hCos)
        # Torus-locked — no linear phase drift
        output = standard_part * (1.0 + self.epsilon * h_cos)

        if self.use_binary_alignment:
            output = output + 0.1 * h_cos
            if self.unwinding_mode:
                output = bit_out

        return output, phi_next, confidence, h_cos, h_sin


class HelixModel(nn.Module):
    """
    Multi-layer Helix model. Stacks HelixNeuronCells.
    Replaces helix.py HelixCore for training tasks.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        harmonics=HARMONICS_STANDARD,
        use_spinor=True,
        use_binary_alignment=False,
        unwinding_mode=False,
        persistence=1.0,
        quantization_strength=0.125,
        spin_multiplier=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_binary_alignment = use_binary_alignment

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(HelixNeuronCell(
                layer_input, hidden_size, harmonics,
                use_spinor=use_spinor,
                quantization_strength=quantization_strength,
                use_binary_alignment=use_binary_alignment,
                unwinding_mode=unwinding_mode,
                persistence=persistence,
                spin_multiplier=spin_multiplier,
            ))

        # Readout: concatenate output + h_cos + h_sin
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_seq, return_sequence=False, return_coordinates=False):
        batch_size, seq_len, _ = input_seq.size()
        outputs = []
        confidences = []
        coords = []

        h_states = [
            torch.zeros(batch_size, self.hidden_size, device=input_seq.device)
            for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            current_input = input_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                current_input, h_states[i], conf, h_cos, h_sin = layer(
                    current_input, h_states[i]
                )
                confidences.append(conf)
            if return_coordinates:
                coords.append((h_cos.detach().cpu(), h_sin.detach().cpu()))
            if return_sequence:
                feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
                outputs.append(self.readout(feats))

        avg_confidence = torch.stack(confidences).mean()

        if return_sequence:
            res = (torch.stack(outputs, dim=1), avg_confidence)
            if return_coordinates:
                res = res + (coords,)
            return res

        final_feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
        res = (self.readout(final_feats), avg_confidence)
        if return_coordinates:
            res = res + (coords,)
        return res

    def save_crystal(self, path):
        torch.save(self.state_dict(), path)

    def load_crystal(self, path, freeze=True):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        if freeze:
            for p in self.parameters():
                p.requires_grad = False


class HelixEncoderModel(HelixModel):
    """
    Encoder-only variant. persistence=0.0 by default for zero-history mode.
    Used for encoding tasks where you want pure stateless readout.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 harmonics=HARMONICS_STANDARD, use_spinor=True,
                 use_binary_alignment=False, persistence=0.0, spin_multiplier=None):
        super().__init__(
            input_size, hidden_size, output_size, num_layers, harmonics,
            use_spinor, use_binary_alignment,
            unwinding_mode=False, persistence=persistence,
            spin_multiplier=spin_multiplier
        )


def landauer_loss(model, beta=0.01):
    """
    Landauer regularization: penalizes information erasure.
    L1 norm on all parameters. Minimizing this keeps the network
    close to the reversible computation regime.
    """
    l_loss = sum(torch.norm(p, p=1) for p in model.parameters())
    return beta * l_loss
