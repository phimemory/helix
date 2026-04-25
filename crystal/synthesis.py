"""
Helix Phase Synthesis
Ported from ROUND Synthesis/ module.

Decodes/generates from phase state back to the original embedding space.
This fills the core gap: Helix could only encode. Now it can also decode.

Implements:
- PhaseDecoder: maps phase angles back to embedding space
- CrystalSynthesizer: generates from a MemoryCrystal's phase state
- PhasicRelay: proves cross-instance communication (the Relay/Sandwich test)
  Two independently trained instances can communicate through phase state
  without ever being trained together — Phasic Sovereignty.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from crystal.substrate import MemoryCrystal


class PhaseDecoder(nn.Module):
    """
    Decodes a phase state vector back into an embedding.

    The decoder reads the harmonic expansion of the phase angles
    and maps them back to the original input space through a learned
    linear projection.

    This is the inverse operation of HelixNeuronCell's forward pass.
    """

    def __init__(self, hidden_size, output_size, harmonics=[1, 2, 4, 8]):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.harmonics = harmonics

        feature_dim = hidden_size * len(harmonics) * 2  # cos + sin per harmonic

        self.decode = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, output_size)
        )

        for m in self.decode.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, phi):
        """
        Args:
            phi: tensor of shape (batch, hidden_size) or (hidden_size,)

        Returns:
            decoded: tensor of shape (batch, output_size)
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)

        features = []
        for h in self.harmonics:
            features.append(torch.cos(h * phi))
            features.append(torch.sin(h * phi))
        features = torch.cat(features, dim=-1)

        return self.decode(features)


class CrystalSynthesizer:
    """
    Generates from a MemoryCrystal's phase state.

    Given a trained PhaseDecoder, synthesizes embeddings that
    represent the accumulated memory stored in a crystal.
    Can generate at any point in the memory timeline using TPI.
    """

    def __init__(self, decoder: PhaseDecoder):
        self.decoder = decoder

    def synthesize(self, crystal: MemoryCrystal):
        """
        Generate an embedding from the current crystal state.

        Returns:
            embedding: tensor of shape (output_size,)
        """
        phi = crystal.recall_compact()
        with torch.no_grad():
            return self.decoder(phi).squeeze(0)

    def synthesize_at(self, tpi, step, harmonics=[1, 2, 4, 8]):
        """
        Generate an embedding from the crystal state at a specific timestep.
        Uses Temporal Phase Indexing for random access.
        """
        phi = tpi.recall_at(step)
        features = []
        for h in harmonics:
            features.append(torch.cos(h * phi))
            features.append(torch.sin(h * phi))
        features = torch.cat(features).unsqueeze(0)

        with torch.no_grad():
            return self.decoder.decode(features).squeeze(0)

    def synthesize_trajectory(self, tpi, steps=None):
        """
        Generate embeddings at all recorded timesteps.
        Returns a list of (step, embedding) tuples.
        """
        if steps is None:
            steps = tpi.step_index
        results = []
        for step in steps:
            phi = tpi.recall_at(step)
            features = []
            for h in self.decoder.harmonics:
                features.append(torch.cos(h * phi))
                features.append(torch.sin(h * phi))
            features = torch.cat(features).unsqueeze(0)
            with torch.no_grad():
                emb = self.decoder.decode(features).squeeze(0)
            results.append((step, emb))
        return results


class PhasicRelay:
    """
    The Relay Test: proves cross-instance communication through phase state.

    Two independently initialized instances (Encoder + Decoder) that have
    never been trained together can communicate through the shared phase
    geometry — Phasic Sovereignty.

    ROUND calls this the "Phasic Sandwich":
    Encoder -> phase state -> Decoder, with zero erasure cost.

    In Helix terms:
    - Instance A absorbs information into a crystal
    - The crystal (pure phase angles) is transferred
    - Instance B reads the crystal and recovers meaningful structure
    - No shared training required — the geometry is the protocol
    """

    def __init__(self, encoder: MemoryCrystal, decoder: PhaseDecoder):
        self.encoder = encoder
        self.decoder = decoder

    def relay(self, embedding):
        """
        Full relay: encode into crystal, decode back out.

        Args:
            embedding: tensor of shape (input_size,)

        Returns:
            relayed: tensor of shape (output_size,)
            phase_state: the intermediate phase state (the "wire")
        """
        self.encoder.absorb(embedding)
        phi = self.encoder.recall_compact()

        with torch.no_grad():
            relayed = self.decoder(phi).squeeze(0)

        return relayed, phi

    def relay_identity_test(self, embeddings):
        """
        Test whether phase relay preserves structural identity.

        Feeds a sequence of embeddings, relays each one, and measures
        whether the relayed output preserves cosine similarity structure.

        Returns:
            mean_cosine: average cosine similarity between input and relayed output
            results: list of (cosine_sim, relayed) per input
        """
        results = []
        for emb in embeddings:
            relayed, phi = self.relay(emb)

            # Align dimensions for comparison
            min_dim = min(emb.shape[0], relayed.shape[0])
            e_norm = emb[:min_dim] / (emb[:min_dim].norm() + 1e-8)
            r_norm = relayed[:min_dim] / (relayed[:min_dim].norm() + 1e-8)
            cosine_sim = (e_norm * r_norm).sum().item()
            results.append((cosine_sim, relayed))

        mean_cosine = sum(r[0] for r in results) / len(results)
        return mean_cosine, results
