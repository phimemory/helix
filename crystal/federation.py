"""
Federated Phase Alignment (FPA)
Privacy-preserving memory merging across independent Helix instances.

Inspired by: McMahan et al., "Communication-Efficient Learning of Deep Networks 
              from Decentralized Data" (Google, 2017) — Federated Learning
             Fisher, "Statistical Analysis of Circular Data" (1993)

Multiple independent Helix crystals can have their phase states merged
without sharing raw data. FPA computes a weighted circular mean across
phase angles, producing a merged crystal that contains collective 
intelligence from all sources.
"""

import torch
import numpy as np


class PhaseFederation:
    """
    Merges multiple Memory Crystals without exposing raw interaction data.
    Uses circular statistics (Fisher, 1993) for proper angular averaging.
    
    Usage:
        fed = PhaseFederation()
        
        # Merge crystals from different sources
        merged_phi = fed.merge([crystal_A.recall_compact(), 
                                crystal_B.recall_compact()])
        
        # Weighted merge (source A is more trusted)
        merged_phi = fed.merge(
            [phi_A, phi_B, phi_C], 
            weights=[0.6, 0.3, 0.1]
        )
        
        # Check how different two memories are
        div = fed.divergence(phi_A, phi_B)
    """
    
    @staticmethod
    def circular_mean(angles, weights=None):
        """
        Compute the weighted circular mean of a set of angles.
        
        Standard arithmetic mean fails for angles because 
        mean([350°, 10°]) = 180° (wrong — should be 0°).
        Circular mean handles the wrap-around correctly.
        
        Args:
            angles: tensor of shape (N, hidden_size) — N sets of phase angles
            weights: optional tensor of shape (N,) — importance weights
            
        Returns:
            mean_angle: tensor of shape (hidden_size,)
        """
        if weights is None:
            weights = torch.ones(angles.shape[0]) / angles.shape[0]
        else:
            weights = weights / weights.sum()  # Normalize
            
        # Convert to unit vectors on the circle
        cos_sum = torch.zeros(angles.shape[1])
        sin_sum = torch.zeros(angles.shape[1])
        
        for i in range(angles.shape[0]):
            cos_sum += weights[i] * torch.cos(angles[i])
            sin_sum += weights[i] * torch.sin(angles[i])
        
        # Circular mean = atan2(sin_mean, cos_mean)
        return torch.atan2(sin_sum, cos_sum)
    
    @staticmethod
    def circular_variance(angles):
        """
        Compute circular variance (0 = all aligned, 1 = maximally dispersed).
        
        Args:
            angles: tensor of shape (N, hidden_size)
            
        Returns:
            variance: tensor of shape (hidden_size,) — per-neuron variance
        """
        cos_mean = torch.cos(angles).mean(dim=0)
        sin_mean = torch.sin(angles).mean(dim=0)
        R = torch.sqrt(cos_mean ** 2 + sin_mean ** 2)
        return 1.0 - R
    
    def merge(self, phase_states, weights=None):
        """
        Merge multiple phase states into a single unified state.
        
        Args:
            phase_states: list of tensors, each of shape (hidden_size,)
            weights: optional list of floats — importance per source
            
        Returns:
            merged_phi: tensor of shape (hidden_size,)
        """
        # Stack into matrix
        angles = torch.stack(phase_states)
        
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        
        return self.circular_mean(angles, weights)
    
    def divergence(self, phi_a, phi_b):
        """
        Compute the angular divergence between two memories.
        
        0 = identical memories
        pi = maximally different memories
        
        Args:
            phi_a, phi_b: tensors of shape (hidden_size,)
            
        Returns:
            mean_divergence: float — average angular distance
            per_neuron: tensor of shape (hidden_size,) — per-neuron distances
        """
        # Angular distance respecting circular topology
        diff = torch.remainder(phi_a - phi_b + np.pi, 2 * np.pi) - np.pi
        per_neuron = diff.abs()
        mean_divergence = per_neuron.mean().item()
        
        return mean_divergence, per_neuron
    
    def alignment_score(self, phi_a, phi_b):
        """
        How aligned are two memories? 
        1.0 = perfectly aligned, 0.0 = completely orthogonal.
        
        This is the cosine similarity on the unit circle.
        """
        cos_diff = torch.cos(phi_a - phi_b)
        return cos_diff.mean().item()
    
    def consensus(self, phase_states):
        """
        Check if multiple crystals agree on their memory.
        
        Returns:
            consensus_score: float in [0, 1]. 1 = perfect agreement.
            variance_per_neuron: which neurons disagree the most.
        """
        angles = torch.stack(phase_states)
        var = self.circular_variance(angles)
        consensus = 1.0 - var.mean().item()
        return consensus, var
    
    def selective_merge(self, phase_states, weights=None, 
                        agreement_threshold=0.7):
        """
        Smart merge: only merge neurons where sources agree.
        Neurons with high disagreement keep the highest-weighted source's value.
        
        This prevents merging conflicting or contradictory memories.
        """
        angles = torch.stack(phase_states)
        
        if weights is None:
            weights = torch.ones(len(phase_states)) / len(phase_states)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
        
        # Compute per-neuron agreement
        var = self.circular_variance(angles)
        agreement = 1.0 - var
        
        # Circular mean for agreeing neurons
        merged = self.circular_mean(angles, weights)
        
        # For disagreeing neurons, use the highest-weighted source
        best_source_idx = weights.argmax().item()
        
        disagree_mask = agreement < agreement_threshold
        merged[disagree_mask] = phase_states[best_source_idx][disagree_mask]
        
        return merged, agreement
