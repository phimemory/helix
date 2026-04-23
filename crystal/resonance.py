"""
Anticipatory Phase Resonance (APR)
Predictive pattern detection from accumulated phase state geometry.

Inspired by: Strogatz, "Nonlinear Dynamics and Chaos" (resonance in coupled oscillators)
             Hopfield, "Neural Networks and Physical Systems" (1982)

When a Helix cell has absorbed enough sequential data from the same source,
the phase angles exhibit resonant patterns — predictable oscillation modes.
APR detects these modes and extracts predicted next-input distributions
from the harmonic spectrum.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ResonanceDetector(nn.Module):
    """
    Analyzes the harmonic spectrum of a Helix phase state for stable 
    oscillation modes that indicate predictable patterns.
    
    Usage:
        detector = ResonanceDetector(hidden_size=64, output_size=768)
        
        # After absorbing many interactions from the same user:
        phi_state = crystal.recall_compact()
        
        is_resonant = detector.detect_resonance(phi_state)
        if is_resonant:
            predicted = detector.predict_next(phi_state)
    """
    
    def __init__(self, hidden_size=64, output_size=768, 
                 harmonics=[1, 2, 4, 8], resonance_threshold=0.85):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.harmonics = harmonics
        self.resonance_threshold = resonance_threshold
        
        feature_dim = hidden_size * len(harmonics) * 2
        
        # Resonance detector: analyzes harmonic spectrum for periodicity
        self.resonance_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Prediction head: extracts next-input prediction from resonant state
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, output_size)
        )
        
        # Phase history for resonance analysis
        self.phase_history = []
        self.max_history = 100
        
    def _expand_harmonics(self, phi):
        """Expand phase state into harmonic feature vector."""
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        features = []
        for h in self.harmonics:
            features.append(torch.cos(h * phi))
            features.append(torch.sin(h * phi))
        return torch.cat(features, dim=-1)
    
    def record_state(self, phi_state):
        """
        Record a phase state snapshot for resonance analysis.
        Call this after each interaction with the same source.
        """
        self.phase_history.append(phi_state.detach().clone())
        if len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)
    
    def detect_resonance(self, phi_state):
        """
        Check if the current phase state shows resonant patterns.
        
        Returns:
            resonance_score: float in [0, 1]. Above threshold = resonant.
        """
        features = self._expand_harmonics(phi_state)
        score = self.resonance_head(features).item()
        return score
    
    def is_resonant(self, phi_state):
        """Boolean check for resonance."""
        return self.detect_resonance(phi_state) >= self.resonance_threshold
    
    def predict_next(self, phi_state):
        """
        Extract predicted next input from the resonant phase state.
        
        Returns:
            prediction: tensor of shape (output_size,) — predicted embedding
        """
        features = self._expand_harmonics(phi_state)
        return self.prediction_head(features).squeeze(0)
    
    def compute_phase_velocity(self):
        """
        Compute the angular velocity of phase changes across recorded history.
        Stable velocity = strong resonance. Erratic velocity = no pattern.
        
        Returns:
            velocity: tensor of shape (hidden_size,) — angular velocity per neuron
            stability: float — how stable the velocity is (higher = more resonant)
        """
        if len(self.phase_history) < 3:
            return None, 0.0
            
        deltas = []
        for i in range(1, len(self.phase_history)):
            prev = self.phase_history[i - 1]
            curr = self.phase_history[i]
            if prev.dim() > 1:
                prev = prev.squeeze(0)
            if curr.dim() > 1:
                curr = curr.squeeze(0)
            deltas.append(curr - prev)
            
        deltas = torch.stack(deltas)
        mean_velocity = deltas.mean(dim=0)
        velocity_std = deltas.std(dim=0)
        
        # Stability: inverse of coefficient of variation
        stability = 1.0 / (1.0 + velocity_std.mean().item())
        
        return mean_velocity, stability
    
    def clear_history(self):
        """Clear recorded phase history."""
        self.phase_history = []
