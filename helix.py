# HELIX v1.0
import torch
import torch.nn as nn
import numpy as np

HARMONICS_STANDARD = [1, 2, 4, 8]
HARMONICS_7OCTAVE = [0.125, 0.25, 0.5, 1, 2, 4, 8]


class HelixCell(nn.Module):
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8],
                 use_spinor=True, quantization_strength=0.125,
                 use_binary_alignment=False, unwinding_mode=False,
                 persistence=1.0, spin_multiplier=None, full_state=True):
        super(HelixCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.use_spinor = use_spinor
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        self.persistence = persistence
        self.full_state = full_state
        self.spin_multiplier = spin_multiplier if spin_multiplier is not None else (2.0 if use_spinor else 1.0)
        self.unwind_threshold = nn.Parameter(torch.tensor(np.pi - 0.1))

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
    l_loss = 0
    for p in model.parameters(): l_loss += torch.norm(p, p=1)
    return beta * l_loss
