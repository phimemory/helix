"""
Multi-Modal Phase Fusion (MMPF)
Unified memory encoding across text, image, and audio modalities.

Inspired by: Ngiam et al., "Multimodal Deep Learning" (ICML 2011)
             Radford et al., "Learning Transferable Visual Models" (CLIP, 2021)

Standard Helix crystals absorb one type of embedding at a time.
MMPF adds a fusion layer that can absorb text embeddings (768d),
image embeddings (512d CLIP), and audio embeddings (384d Whisper)
simultaneously into the same crystal. One crystal that remembers
what was said, shown, and heard — all in a few hundred bytes.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crystal.substrate import MemoryCrystal


class ModalityProjector(nn.Module):
    """
    Projects a modality-specific embedding into the unified
    Helix input space. Each modality gets its own learned projection.
    """
    
    def __init__(self, input_dim, unified_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, unified_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.proj(x)


class MultiModalFusion(nn.Module):
    """
    Fuses multiple modalities into a single Memory Crystal.
    
    Each modality (text, image, audio) is projected into a unified
    embedding space, then absorbed into the same Helix phase state.
    The modality information is preserved through dedicated phase bands.
    
    Usage:
        fusion = MultiModalFusion(hidden_size=64)
        
        # Absorb text
        fusion.absorb_text(text_embedding)    # e.g. from BERT (768d)
        
        # Absorb image
        fusion.absorb_image(image_embedding)  # e.g. from CLIP (512d)
        
        # Absorb audio
        fusion.absorb_audio(audio_embedding)  # e.g. from Whisper (384d)
        
        # Get unified recall
        features = fusion.recall()
        
        # Export as single crystal
        fusion.export("multimodal_memory.hx")
    """
    
    # Default modality dimensions (configurable)
    DEFAULT_DIMS = {
        'text': 768,     # BERT / sentence-transformers
        'image': 512,    # CLIP ViT-B/32
        'audio': 384,    # Whisper tiny
        'generic': 256   # Fallback for unknown modalities
    }
    
    def __init__(self, hidden_size=64, unified_dim=128, 
                 modality_dims=None, harmonics=[1, 2, 4, 8]):
        super().__init__()
        self.hidden_size = hidden_size
        self.unified_dim = unified_dim
        self.harmonics = harmonics
        
        dims = modality_dims or self.DEFAULT_DIMS
        
        # Create a projector for each modality
        self.projectors = nn.ModuleDict()
        for name, dim in dims.items():
            self.projectors[name] = ModalityProjector(dim, unified_dim)
        
        # The shared crystal that all modalities write to
        self.crystal = MemoryCrystal(
            input_size=unified_dim,
            hidden_size=hidden_size,
            harmonics=harmonics
        )
        
        # Track what modalities have been absorbed
        self.modality_counts = {name: 0 for name in dims}
        
        # Modality tags: small learnable vectors added to distinguish
        # which modality contributed which phase shift
        self.modality_tags = nn.ParameterDict()
        for name in dims:
            self.modality_tags[name] = nn.Parameter(
                torch.randn(unified_dim) * 0.01
            )
    
    def _absorb(self, embedding, modality):
        """Internal: project and absorb an embedding from any modality."""
        if modality not in self.projectors:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Available: {list(self.projectors.keys())}"
            )
        
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Project to unified space + add modality tag
        with torch.no_grad():
            projected = self.projectors[modality](embedding)
            tagged = projected + self.modality_tags[modality].unsqueeze(0)
            self.crystal.absorb(tagged.squeeze(0))
        
        self.modality_counts[modality] += 1
    
    def absorb_text(self, text_embedding):
        """Absorb a text embedding (e.g. from BERT, 768d)."""
        self._absorb(text_embedding, 'text')
    
    def absorb_image(self, image_embedding):
        """Absorb an image embedding (e.g. from CLIP, 512d)."""
        self._absorb(image_embedding, 'image')
    
    def absorb_audio(self, audio_embedding):
        """Absorb an audio embedding (e.g. from Whisper, 384d)."""
        self._absorb(audio_embedding, 'audio')
    
    def absorb_generic(self, embedding):
        """Absorb a generic embedding (256d fallback)."""
        self._absorb(embedding, 'generic')
    
    def absorb(self, embedding, modality='generic'):
        """Absorb any embedding with explicit modality label."""
        self._absorb(embedding, modality)
    
    def recall(self):
        """
        Recall the unified multi-modal memory as a feature vector.
        
        Returns:
            features: tensor of shape (hidden_size * len(harmonics) * 2,)
        """
        return self.crystal.recall()
    
    def recall_compact(self):
        """Return raw phase angles."""
        return self.crystal.recall_compact()
    
    def export(self, path):
        """Export the multi-modal crystal as a .hx file."""
        return self.crystal.export(path)
    
    def load(self, path):
        """Load a multi-modal crystal from a .hx file."""
        self.crystal.load(path)
    
    def reset(self):
        """Clear all multi-modal memory."""
        self.crystal.reset()
        for name in self.modality_counts:
            self.modality_counts[name] = 0
    
    def stats(self):
        """Return absorption statistics per modality."""
        total = sum(self.modality_counts.values())
        return {
            'modality_counts': dict(self.modality_counts),
            'total_absorbed': total,
            'crystal_size_bytes': self.crystal.size_bytes(),
            'modalities_active': [
                name for name, count in self.modality_counts.items() 
                if count > 0
            ]
        }
    
    def __repr__(self):
        active = [n for n, c in self.modality_counts.items() if c > 0]
        total = sum(self.modality_counts.values())
        return (f"MultiModalFusion(modalities={active}, "
                f"total_absorbed={total}, "
                f"crystal_size={self.crystal.size_bytes()}B)")
