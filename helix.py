"""
Helix - Phase-Rotation Sequence Memory Architecture

Core cell: phase angles accumulate over time instead of decaying via
contractive multiplication. A soft quantization sieve snaps phases to a
pi/4 grid for discrete stability. Multi-harmonic readout (cos/sin at
harmonics 1, 2, 4, 8) extracts a rich feature vector from a single angle.

Two cell variants:
  HelixCell       - used by the crystal substrate (has full_state param)
  HelixNeuronCell - cleaner v2 for training tasks
"""
import torch
import torch.nn as nn
import numpy as np

HARMONICS_STANDARD = [1, 2, 4, 8]
HARMONICS_7OCTAVE  = [0.125, 0.25, 0.5, 1, 2, 4, 8]
HARMONICS_SPINOR   = [0.5, 1, 2, 4]

# Multi-clock band speeds inspired by Yarnix.
# Maps to brain oscillation bands: ultra-fast (gamma), fast (beta),
# slow (alpha), ultra-slow (theta).
CLOCK_SPEEDS_DEFAULT = (0.50, 0.80, 0.95, 0.999)


class HelixCell(nn.Module):
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8],
                 use_spinor=True, quantization_strength=0.125,
                 use_binary_alignment=False, unwinding_mode=False,
                 persistence=1.0, spin_multiplier=None, full_state=True,
                 clock_speeds=None):
        super(HelixCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.use_spinor = use_spinor
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        self.full_state = full_state
        self.spin_multiplier = spin_multiplier if spin_multiplier is not None else (2.0 if use_spinor else 1.0)
        self.unwind_threshold = nn.Parameter(torch.tensor(np.pi - 0.1))

        # Multi-clock persistence. When clock_speeds is given each neuron
        # group gets its own persistence value. Without it, falls back to
        # the scalar persistence parameter (original behavior).
        self.clock_speeds = clock_speeds
        if clock_speeds is not None:
            n_bands = len(clock_speeds)
            assert hidden_size % n_bands == 0, (
                f"hidden_size ({hidden_size}) must be divisible by "
                f"n_bands ({n_bands}) when using clock_speeds"
            )
            band_size = hidden_size // n_bands
            vec = []
            for speed in clock_speeds:
                vec.extend([speed] * band_size)
            self.register_buffer(
                'persistence_vec',
                torch.tensor(vec, dtype=torch.float32)
            )
            self.persistence = None  # not used when clock_speeds is set

            # Cross-band mixer: lets fast and slow bands inform each other.
            self.band_mixer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for m in self.band_mixer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.persistence = persistence
            self.persistence_vec = None
            self.band_mixer = None

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.epsilon = nn.Parameter(torch.Tensor(hidden_size))
        self.diagnostic_harmonics = nn.Parameter(torch.Tensor(hidden_size, len(harmonics)))

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.zeros_(p.data)
            elif 'bias' in name:
                nn.init.zeros_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.kaiming_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

        with torch.no_grad():
            for j in range(self.hidden_size):
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5))
            nn.init.uniform_(self.diagnostic_harmonics, 0.0, 1.0)
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                self.bias[self.hidden_size + j] = j * spread

    def forward(self, x, h_prev):
        gates = x @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        standard_part = torch.tanh(x_gate)
        multiplier = self.spin_multiplier

        if self.use_binary_alignment:
            if self.unwinding_mode:
                bit_out = (h_prev >= self.unwind_threshold).float()
                phi_next = (h_prev - bit_out * np.pi) * 2.0
                q_grid = np.pi / 128.0
                q_snap = torch.round(phi_next / q_grid) * q_grid
                phi_next = q_snap
            else:
                incoming_bit = x[:, 0:1]
                phi_next = (h_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit
        else:
            phi_shift = (torch.sigmoid(phi_gate) * np.pi * multiplier)
            # Use per-neuron persistence vector when clock_speeds was set,
            # otherwise use the scalar persistence value (original behavior).
            if self.persistence_vec is not None:
                phi_next = h_prev * self.persistence_vec + phi_shift
            else:
                phi_next = (h_prev * self.persistence) + phi_shift
            q_sieve = torch.round(phi_next / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next + self.quantization_strength * (q_sieve - phi_next)

        if not self.full_state:
            phi_next = torch.remainder(phi_next, 2.0 * np.pi * multiplier)

        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.diagnostic_harmonics[:, idx] * torch.cos(h * phi_next)
            h_sin += self.diagnostic_harmonics[:, idx] * torch.sin(h * phi_next)

        confidence = (h_cos.abs() / len(self.harmonics)).detach()
        output = standard_part * (1.0 + self.epsilon * (h_cos + h_sin) * 0.5)

        # Cross-band mixer: routes information between fast and slow neuron
        # groups so each timescale can inform the others.
        if self.band_mixer is not None:
            output = output + 0.1 * self.band_mixer(output)

        if self.use_binary_alignment:
            output = output + 0.1 * h_cos
            if self.unwinding_mode:
                output = bit_out

        return output, phi_next, confidence, h_cos, h_sin


class HelixModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 harmonics=[1, 2, 4, 8], use_spinor=True,
                 use_binary_alignment=False, unwinding_mode=False,
                 persistence=1.0, quantization_strength=0.125,
                 spin_multiplier=None, full_state=True):
        super(HelixModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_binary_alignment = use_binary_alignment
        self.full_state = full_state

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(HelixCell(
                layer_input, hidden_size, harmonics,
                use_spinor=use_spinor,
                quantization_strength=quantization_strength,
                use_binary_alignment=use_binary_alignment,
                unwinding_mode=unwinding_mode,
                persistence=persistence,
                spin_multiplier=spin_multiplier,
                full_state=full_state
            ))

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
        h_states = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
                     for _ in range(self.num_layers)]
        confidences = []
        coords = []

        for t in range(seq_len):
            current_input = input_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                current_input, h_states[i], conf, h_cos, h_sin = layer(current_input, h_states[i])
                confidences.append(conf)
            if return_coordinates:
                coords.append((h_cos.detach().cpu(), h_sin.detach().cpu()))
            if return_sequence:
                feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
                outputs.append(self.readout(feats))

        avg_confidence = torch.stack(confidences).mean()
        if return_sequence:
            res = (torch.stack(outputs, dim=1), avg_confidence)
            if return_coordinates: res = res + (coords,)
            return res

        final_feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
        res = (self.readout(final_feats), avg_confidence)
        if return_coordinates: res = res + (coords,)
        return res

    def save_model(self, path): torch.save(self.state_dict(), path)
    def load_model(self, path, freeze=True):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        if freeze:
            for p in self.parameters(): p.requires_grad = False

    def save_crystal(self, path): self.save_model(path)
    def load_crystal(self, path, freeze=True): self.load_model(path, freeze)


class HelixEncoderModel(HelixModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 harmonics=[1, 2, 4, 8], use_spinor=True,
                 use_binary_alignment=False, persistence=0.0,
                 spin_multiplier=None, full_state=False):
        super(HelixEncoderModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, harmonics,
            use_spinor, use_binary_alignment, unwinding_mode=False,
            persistence=persistence, spin_multiplier=spin_multiplier,
            full_state=full_state
        )


def landauer_loss(model, beta=0.01):
    l_loss = sum(torch.norm(p, p=1) for p in model.parameters())
    return beta * l_loss


# ---------------------------------------------------------------------------
# HelixNeuronCell — v2 training cell (cleaner names, no full_state wrap)
# ---------------------------------------------------------------------------

class HelixNeuronCell(nn.Module):
    """
    Preferred cell for training tasks. Same phase-rotation core as HelixCell
    but with cleaner parameter names and no full_state wrap — phase accumulates
    on the real line, preserving the winding number across arbitrarily long
    sequences.
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
        self.spin_multiplier = (
            spin_multiplier if spin_multiplier is not None
            else (2.0 if use_spinor else 1.0)
        )

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.epsilon = nn.Parameter(torch.Tensor(hidden_size))
        self.harmonic_weights = nn.Parameter(torch.Tensor(hidden_size, len(harmonics)))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.zeros_(self.weight_hh)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            for j in range(self.hidden_size):
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5))
            nn.init.uniform_(self.harmonic_weights, 0.0, 1.0)
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                self.bias[self.hidden_size + j] = j * spread

    def forward(self, x, phi_prev):
        gates = x @ self.weight_ih + phi_prev @ self.weight_hh + self.bias
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        standard_part = torch.tanh(x_gate)

        if self.use_binary_alignment:
            if self.unwinding_mode:
                bit_out = (phi_prev >= (np.pi - 1e-7)).float()
                phi_next = (phi_prev - bit_out * np.pi) * 2.0
                q_grid = np.pi / 128.0
                phi_next = torch.round(phi_next / q_grid) * q_grid
            else:
                incoming_bit = x[:, 0:1]
                phi_next = (phi_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit
        else:
            phi_shift = torch.sigmoid(phi_gate) * np.pi * self.spin_multiplier
            phi_next = (phi_prev * self.persistence) + phi_shift
            q_sieve = torch.round(phi_next / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next + self.quantization_strength * (q_sieve - phi_next)

        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.harmonic_weights[:, idx] * torch.cos(h * phi_next)
            h_sin += self.harmonic_weights[:, idx] * torch.sin(h * phi_next)

        confidence = (h_cos.abs() / len(self.harmonics)).detach()
        output = standard_part * (1.0 + self.epsilon * h_cos)

        if self.use_binary_alignment:
            output = output + 0.1 * h_cos
            if self.unwinding_mode:
                output = bit_out

        return output, phi_next, confidence, h_cos, h_sin


class HelixNeuronModel(nn.Module):
    """Multi-layer model using HelixNeuronCell. Preferred for training."""

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
            self.layers.append(HelixNeuronCell(
                input_size if i == 0 else hidden_size,
                hidden_size, harmonics,
                use_spinor=use_spinor,
                quantization_strength=quantization_strength,
                use_binary_alignment=use_binary_alignment,
                unwinding_mode=unwinding_mode,
                persistence=persistence,
                spin_multiplier=spin_multiplier,
            ))

        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_seq, return_sequence=False, return_coordinates=False):
        batch_size, seq_len, _ = input_seq.size()
        outputs, confidences, coords = [], [], []
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
