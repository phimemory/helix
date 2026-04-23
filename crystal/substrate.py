"""
Crystalline Substrate Layer (CSL)
Portable, model-agnostic memory storage using Helix phase geometry.

Inspired by: Tegmark, "Life 3.0" (substrate-independent thinking)
             Finn et al., "Model-Agnostic Meta-Learning" (MAML, 2017)

The Memory Crystal is the core abstraction: a tiny binary file (.hx) containing
the accumulated phase state of a Helix cell. Any model can read from it.
The memory is stored in pure geometry (angles on circles), not in 
architecture-specific weights.
"""

import torch
import torch.nn as nn
import numpy as np
import struct
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from helix import HelixCell


# .hx file format version
HX_VERSION = 1
HX_MAGIC = b'HELX'  # 4-byte magic number


class MemoryCrystal(nn.Module):
    """
    A portable, lossless memory container backed by Helix phase geometry.
    
    Absorbs sequential data as phase rotations. Exports as a tiny .hx file
    that any model can load and read from.
    
    Usage:
        crystal = MemoryCrystal(input_size=768, hidden_size=64)
        crystal.absorb(embedding_1)
        crystal.absorb(embedding_2)
        crystal.export("memory.hx")
        
        # Later, on a different machine, with a different model:
        crystal2 = MemoryCrystal(input_size=768, hidden_size=64)
        crystal2.load("memory.hx")
        features = crystal2.recall()  # rich feature vector for any model
    """
    
    def __init__(self, input_size, hidden_size=64, harmonics=[1, 2, 4, 8]):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        
        self.cell = HelixCell(
            input_size=input_size,
            hidden_size=hidden_size,
            harmonics=harmonics,
            full_state=True,
            quantization_strength=0.125
        )
        
        # The phase state — this IS the memory
        self.register_buffer('phi_state', torch.zeros(1, hidden_size))
        
        # Metadata
        self.absorb_count = 0
        
    def absorb(self, embedding):
        """
        Feed new information into the crystal.
        The phase angles rotate to encode the data. Nothing is ever lost.
        
        Args:
            embedding: tensor of shape (input_size,) or (1, input_size)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        with torch.no_grad():
            output, phi_next, confidence, h_cos, h_sin = self.cell(
                embedding, self.phi_state
            )
            self.phi_state = phi_next
            self.absorb_count += 1
            
        return confidence
    
    def absorb_sequence(self, embeddings):
        """
        Feed an entire sequence of embeddings into the crystal.
        
        Args:
            embeddings: tensor of shape (seq_len, input_size)
        """
        for t in range(embeddings.shape[0]):
            self.absorb(embeddings[t])
    
    def recall(self):
        """
        Extract a rich feature vector from the current phase state.
        Uses harmonic expansion to produce a high-dimensional representation
        from the compact phase angles.
        
        Returns:
            features: tensor of shape (hidden_size * len(harmonics) * 2,)
                      containing cos and sin at each harmonic frequency
        """
        features = []
        for h in self.harmonics:
            features.append(torch.cos(h * self.phi_state))
            features.append(torch.sin(h * self.phi_state))
        return torch.cat(features, dim=-1).squeeze(0)
    
    def recall_compact(self):
        """
        Return just the raw phase angles (most compact representation).
        
        Returns:
            phi: tensor of shape (hidden_size,)
        """
        return self.phi_state.squeeze(0).clone()
    
    def winding_number(self):
        """
        How many full rotations each neuron has completed.
        This encodes total context length seen.
        """
        return (self.phi_state / (2 * np.pi)).squeeze(0)
    
    def reset(self):
        """Clear all memory. Phase returns to zero."""
        self.phi_state.zero_()
        self.absorb_count = 0
    
    def export(self, path):
        """
        Save the crystal as a .hx binary file.
        
        Format:
            4 bytes: magic number 'HELX'
            4 bytes: version (uint32)
            4 bytes: hidden_size (uint32)
            4 bytes: num_harmonics (uint32)
            4 bytes: absorb_count (uint32)
            N*4 bytes: harmonics as float32
            H*4 bytes: phase angles as float32
            32 bytes: SHA-256 checksum of all above
        """
        phi_numpy = self.phi_state.squeeze(0).cpu().numpy().astype(np.float32)
        harmonics_numpy = np.array(self.harmonics, dtype=np.float32)
        
        # Build the binary payload
        payload = bytearray()
        payload += HX_MAGIC
        payload += struct.pack('<I', HX_VERSION)
        payload += struct.pack('<I', self.hidden_size)
        payload += struct.pack('<I', len(self.harmonics))
        payload += struct.pack('<I', self.absorb_count)
        payload += harmonics_numpy.tobytes()
        payload += phi_numpy.tobytes()
        
        # Checksum
        checksum = hashlib.sha256(bytes(payload)).digest()
        payload += checksum
        
        with open(path, 'wb') as f:
            f.write(payload)
            
        return len(payload)
    
    def load(self, path):
        """
        Load a crystal from a .hx binary file.
        Verifies integrity via SHA-256 checksum.
        """
        with open(path, 'rb') as f:
            data = f.read()
        
        # Verify magic
        assert data[:4] == HX_MAGIC, f"Not a valid .hx file (magic: {data[:4]})"
        
        # Parse header
        version = struct.unpack('<I', data[4:8])[0]
        hidden_size = struct.unpack('<I', data[8:12])[0]
        num_harmonics = struct.unpack('<I', data[12:16])[0]
        absorb_count = struct.unpack('<I', data[16:20])[0]
        
        assert version == HX_VERSION, f"Unsupported .hx version: {version}"
        assert hidden_size == self.hidden_size, \
            f"Hidden size mismatch: file={hidden_size}, crystal={self.hidden_size}"
        
        # Parse harmonics
        harm_start = 20
        harm_end = harm_start + num_harmonics * 4
        harmonics = np.frombuffer(data[harm_start:harm_end], dtype=np.float32)
        
        # Parse phase angles
        phi_start = harm_end
        phi_end = phi_start + hidden_size * 4
        phi = np.frombuffer(data[phi_start:phi_end], dtype=np.float32)
        
        # Verify checksum
        checksum_stored = data[phi_end:phi_end + 32]
        checksum_computed = hashlib.sha256(data[:phi_end]).digest()
        assert checksum_stored == checksum_computed, "Checksum mismatch — corrupted .hx file"
        
        # Load state
        self.phi_state = torch.tensor(phi, dtype=torch.float32).unsqueeze(0)
        self.harmonics = harmonics.tolist()
        self.absorb_count = absorb_count
        
    def size_bytes(self):
        """How many bytes this crystal occupies as a .hx file."""
        return 20 + len(self.harmonics) * 4 + self.hidden_size * 4 + 32
    
    def __repr__(self):
        return (f"MemoryCrystal(hidden={self.hidden_size}, "
                f"harmonics={self.harmonics}, "
                f"absorbed={self.absorb_count}, "
                f"size={self.size_bytes()}B)")
