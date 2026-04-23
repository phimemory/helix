"""
Harmonic Spectrum Caching (HSC)
Incremental spectral computation for real-time Helix inference.

Inspired by: Spectral methods in numerical analysis
             KV-cache optimization in Transformer architectures

Instead of recomputing cos(h*phi) and sin(h*phi) for every harmonic
on every recall, HSC maintains a pre-computed cache that only updates
the frequency bands affected by the most recent phase delta.
"""

import torch
import numpy as np


class SpectrumCache:
    """
    Pre-computes and caches the harmonic feature expansion of a phase state.
    When phase changes by a small delta, incrementally updates only the 
    affected bands instead of recomputing the full spectrum.
    
    Usage:
        cache = SpectrumCache(hidden_size=64, harmonics=[1, 2, 4, 8])
        cache.initialize(phi_state)
        features = cache.get_features()
        
        # After a phase update:
        cache.update(phi_delta)  # Only recomputes what changed
        features = cache.get_features()  # Returns updated features
    """
    
    def __init__(self, hidden_size, harmonics=[1, 2, 4, 8]):
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.num_harmonics = len(harmonics)
        
        # Cached spectrum: shape (num_harmonics * 2, hidden_size)
        # Layout: [cos(h1*phi), sin(h1*phi), cos(h2*phi), sin(h2*phi), ...]
        self.cos_cache = torch.zeros(self.num_harmonics, hidden_size)
        self.sin_cache = torch.zeros(self.num_harmonics, hidden_size)
        
        # Current phase state
        self.phi = torch.zeros(hidden_size)
        
        # Stats
        self.full_computes = 0
        self.incremental_updates = 0
        self.cache_valid = False
        
    def initialize(self, phi_state):
        """
        Full computation of the spectral cache from a phase state.
        Call this once when loading a crystal.
        
        Args:
            phi_state: tensor of shape (hidden_size,) or (1, hidden_size)
        """
        if phi_state.dim() > 1:
            phi_state = phi_state.squeeze(0)
            
        self.phi = phi_state.clone()
        
        for i, h in enumerate(self.harmonics):
            self.cos_cache[i] = torch.cos(h * self.phi)
            self.sin_cache[i] = torch.sin(h * self.phi)
            
        self.cache_valid = True
        self.full_computes += 1
    
    def update(self, phi_delta, threshold=1e-6):
        """
        Incrementally update the cache based on phase changes.
        Only recomputes frequency bands for neurons whose phase actually changed.
        
        Args:
            phi_delta: tensor of shape (hidden_size,) — the change in phase
            threshold: minimum delta to trigger recomputation for a neuron
        """
        if phi_delta.dim() > 1:
            phi_delta = phi_delta.squeeze(0)
            
        # Find which neurons actually changed
        changed_mask = phi_delta.abs() > threshold
        num_changed = changed_mask.sum().item()
        
        if num_changed == 0:
            return  # Nothing changed, cache is still valid
        
        # Update phase
        self.phi += phi_delta
        
        if num_changed > self.hidden_size * 0.5:
            # More than half changed — full recompute is faster
            self.initialize(self.phi)
            return
        
        # Incremental update: only recompute changed neurons
        changed_indices = torch.where(changed_mask)[0]
        for i, h in enumerate(self.harmonics):
            self.cos_cache[i, changed_indices] = torch.cos(h * self.phi[changed_indices])
            self.sin_cache[i, changed_indices] = torch.sin(h * self.phi[changed_indices])
        
        self.incremental_updates += 1
    
    def get_features(self):
        """
        Return the full harmonic feature vector from cache.
        
        Returns:
            features: tensor of shape (hidden_size * num_harmonics * 2,)
        """
        assert self.cache_valid, "Cache not initialized. Call initialize() first."
        
        parts = []
        for i in range(self.num_harmonics):
            parts.append(self.cos_cache[i])
            parts.append(self.sin_cache[i])
        return torch.cat(parts)
    
    def get_cos_features(self):
        """Return only cosine features (for backward compat)."""
        return self.cos_cache.sum(dim=0)
    
    def get_sin_features(self):
        """Return only sine features."""
        return self.sin_cache.sum(dim=0)
    
    def cache_hit_rate(self):
        """Percentage of updates that were incremental vs full recompute."""
        total = self.full_computes + self.incremental_updates
        if total == 0:
            return 0.0
        return self.incremental_updates / total * 100.0
    
    def stats(self):
        return {
            'full_computes': self.full_computes,
            'incremental_updates': self.incremental_updates,
            'cache_hit_rate': f"{self.cache_hit_rate():.1f}%",
            'hidden_size': self.hidden_size,
            'num_harmonics': self.num_harmonics
        }
