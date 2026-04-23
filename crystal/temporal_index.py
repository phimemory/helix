"""
Temporal Phase Indexing (TPI)
Random-access memory retrieval from the phase timeline.

Inspired by: Skip lists (Pugh, 1990)
             Hierarchical temporal memory (Hawkins, 2004)

Standard Helix gives you the accumulated state but can't answer
"what happened at step 47?" TPI saves periodic phase snapshots at
configurable intervals, creating a temporal index that enables
random access into the memory timeline. This kills the "no random
access" tradeoff completely.
"""

import torch
import numpy as np
import struct
import hashlib


class TemporalPhaseIndex:
    """
    Maintains a temporal index of phase state snapshots at configurable
    intervals, enabling random-access retrieval into the memory timeline.
    
    Usage:
        tpi = TemporalPhaseIndex(hidden_size=64, snapshot_interval=10)
        
        # During absorption:
        for t, embedding in enumerate(sequence):
            crystal.absorb(embedding)
            tpi.record(t, crystal.recall_compact())
        
        # Random access:
        phi_at_step_47 = tpi.recall_at(47)
        features_at_step_47 = tpi.recall_features_at(47, harmonics=[1,2,4,8])
        
        # Range query:
        segment = tpi.recall_range(20, 50)
        
        # Find when a specific pattern occurred:
        matches = tpi.search(query_phi, top_k=5)
    """
    
    def __init__(self, hidden_size, snapshot_interval=10, max_snapshots=10000):
        self.hidden_size = hidden_size
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        
        # Timeline storage: step -> phase state
        self.timeline = {}
        self.step_index = []  # Sorted list of recorded steps
        self.total_steps = 0
        
    def record(self, step, phi_state):
        """
        Record a phase state snapshot at a given step.
        Automatically records at snapshot_interval boundaries,
        but can be called manually for important moments.
        
        Args:
            step: integer timestep
            phi_state: tensor of shape (hidden_size,) or (1, hidden_size)
        """
        if phi_state.dim() > 1:
            phi_state = phi_state.squeeze(0)
            
        self.total_steps = max(self.total_steps, step + 1)
        
        # Always record at interval boundaries
        should_record = (step % self.snapshot_interval == 0) or (step == 0)
        
        if should_record:
            if len(self.timeline) >= self.max_snapshots:
                # Evict oldest snapshot (keep every Nth for long-term memory)
                self._evict_oldest()
                
            self.timeline[step] = phi_state.detach().clone()
            if step not in self.step_index:
                self.step_index.append(step)
                self.step_index.sort()
    
    def force_record(self, step, phi_state):
        """Force-record at this step regardless of interval."""
        if phi_state.dim() > 1:
            phi_state = phi_state.squeeze(0)
        self.timeline[step] = phi_state.detach().clone()
        if step not in self.step_index:
            self.step_index.append(step)
            self.step_index.sort()
        self.total_steps = max(self.total_steps, step + 1)
    
    def recall_at(self, step):
        """
        Retrieve the phase state at a specific step.
        If exact step isn't recorded, interpolates from nearest snapshots.
        
        Args:
            step: integer timestep to retrieve
            
        Returns:
            phi: tensor of shape (hidden_size,)
        """
        # Exact match
        if step in self.timeline:
            return self.timeline[step].clone()
        
        # Find nearest recorded steps
        before, after = self._find_neighbors(step)
        
        if before is None and after is None:
            raise ValueError(f"No snapshots recorded. Record some first.")
        if before is None:
            return self.timeline[after].clone()
        if after is None:
            return self.timeline[before].clone()
        
        # Linear interpolation on the phase angles
        t = (step - before) / (after - before)
        phi_before = self.timeline[before]
        phi_after = self.timeline[after]
        
        # Use circular interpolation (slerp on angles)
        diff = phi_after - phi_before
        # Wrap to [-pi, pi] for proper interpolation
        diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
        interpolated = phi_before + t * diff
        
        return interpolated
    
    def recall_features_at(self, step, harmonics=[1, 2, 4, 8]):
        """
        Retrieve the full harmonic feature vector at a specific step.
        
        Returns:
            features: tensor of shape (hidden_size * len(harmonics) * 2,)
        """
        phi = self.recall_at(step)
        features = []
        for h in harmonics:
            features.append(torch.cos(h * phi))
            features.append(torch.sin(h * phi))
        return torch.cat(features)
    
    def recall_range(self, start_step, end_step):
        """
        Retrieve all recorded snapshots within a step range.
        
        Returns:
            list of (step, phi_state) tuples
        """
        results = []
        for step in self.step_index:
            if start_step <= step <= end_step:
                results.append((step, self.timeline[step].clone()))
        return results
    
    def search(self, query_phi, top_k=5):
        """
        Find the timesteps where the phase state was most similar
        to a query pattern.
        
        Args:
            query_phi: tensor of shape (hidden_size,) — the pattern to search for
            top_k: number of results to return
            
        Returns:
            list of (step, similarity_score) tuples, sorted by similarity
        """
        if query_phi.dim() > 1:
            query_phi = query_phi.squeeze(0)
            
        similarities = []
        for step in self.step_index:
            phi = self.timeline[step]
            # Cosine similarity on the unit circle
            cos_diff = torch.cos(query_phi - phi).mean().item()
            similarities.append((step, cos_diff))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def phase_velocity_at(self, step):
        """
        Compute how fast the phase was changing at a given step.
        High velocity = lots of new information being absorbed.
        Low velocity = redundant or similar information.
        """
        before, after = self._find_neighbors(step)
        if before is None or after is None:
            return None
            
        delta = self.timeline[after] - self.timeline[before]
        delta = torch.remainder(delta + np.pi, 2 * np.pi) - np.pi
        dt = after - before
        return (delta.abs() / dt).mean().item()
    
    def _find_neighbors(self, step):
        """Find the nearest recorded steps before and after the given step."""
        before = None
        after = None
        for s in self.step_index:
            if s <= step:
                before = s
            if s >= step and after is None:
                after = s
        return before, after
    
    def _evict_oldest(self):
        """Evict the oldest snapshot, but keep every 10th for long-term memory."""
        for step in list(self.step_index):
            if step % (self.snapshot_interval * 10) != 0:
                del self.timeline[step]
                self.step_index.remove(step)
                return
    
    def num_snapshots(self):
        return len(self.timeline)
    
    def memory_bytes(self):
        """Total memory used by the index."""
        return len(self.timeline) * self.hidden_size * 4  # float32
    
    def export_index(self):
        """Export the temporal index as a serializable dict."""
        return {
            'hidden_size': self.hidden_size,
            'snapshot_interval': self.snapshot_interval,
            'total_steps': self.total_steps,
            'snapshots': {
                step: phi.cpu().numpy().tolist() 
                for step, phi in self.timeline.items()
            }
        }
    
    def load_index(self, data):
        """Load a temporal index from a dict."""
        self.hidden_size = data['hidden_size']
        self.snapshot_interval = data['snapshot_interval']
        self.total_steps = data['total_steps']
        self.timeline = {
            int(step): torch.tensor(phi, dtype=torch.float32)
            for step, phi in data['snapshots'].items()
        }
        self.step_index = sorted(self.timeline.keys())
    
    def stats(self):
        return {
            'total_steps_seen': self.total_steps,
            'snapshots_stored': self.num_snapshots(),
            'memory_bytes': self.memory_bytes(),
            'snapshot_interval': self.snapshot_interval,
            'coverage': f"{self.num_snapshots()}/{self.total_steps}" if self.total_steps > 0 else "0/0"
        }
