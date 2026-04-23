"""
Phase Diff Protocol (PDP)
Memory versioning and differential analysis between crystal states.

Inspired by: Git diff algorithm (Myers, 1986)
             Optimal transport / Earth Mover's Distance (Rubner et al., 1998)

Like `git diff` but for memory. Given two .hx crystals, PDP computes
exactly what changed between them. This enables memory versioning:
track how preferences evolved, roll back to a previous state, or
merge two memory branches.
"""

import torch
import numpy as np


class PhaseDiff:
    """
    Computes, visualizes, and applies diffs between Helix phase states.
    
    Usage:
        differ = PhaseDiff()
        
        # Compare two memory states
        diff = differ.diff(phi_old, phi_new)
        print(diff.summary())
        
        # Apply a diff (like git apply)
        phi_patched = differ.apply(phi_old, diff)
        assert torch.equal(phi_patched, phi_new)
        
        # Version history
        tracker = differ.create_tracker()
        tracker.commit(phi_v1, "initial memory")
        tracker.commit(phi_v2, "after conversation with John")
        tracker.commit(phi_v3, "after John's allergy update")
        tracker.rollback(1)  # Go back to v2
    """
    
    def diff(self, phi_old, phi_new):
        """
        Compute the phase diff between two states.
        
        Args:
            phi_old: tensor of shape (hidden_size,) — the "before" state
            phi_new: tensor of shape (hidden_size,) — the "after" state
            
        Returns:
            PhaseChangeSet object containing the diff
        """
        if phi_old.dim() > 1:
            phi_old = phi_old.squeeze(0)
        if phi_new.dim() > 1:
            phi_new = phi_new.squeeze(0)
            
        # Compute circular difference (wrapping at 2*pi)
        raw_delta = phi_new - phi_old
        circular_delta = torch.remainder(raw_delta + np.pi, 2 * np.pi) - np.pi
        
        # Classify changes
        abs_delta = circular_delta.abs()
        threshold_minor = 0.01     # < 0.01 rad = no meaningful change
        threshold_major = np.pi/4  # > 45 degrees = major shift
        
        unchanged_mask = abs_delta < threshold_minor
        minor_mask = (abs_delta >= threshold_minor) & (abs_delta < threshold_major)
        major_mask = abs_delta >= threshold_major
        
        return PhaseChangeSet(
            phi_old=phi_old.clone(),
            phi_new=phi_new.clone(),
            delta=circular_delta,
            unchanged_mask=unchanged_mask,
            minor_mask=minor_mask,
            major_mask=major_mask
        )
    
    def apply(self, phi_state, changeset):
        """
        Apply a phase diff to a state (like git apply / git patch).
        
        Args:
            phi_state: tensor of shape (hidden_size,) — the base state
            changeset: PhaseChangeSet from a previous diff()
            
        Returns:
            patched: tensor of shape (hidden_size,) — patched state
        """
        if phi_state.dim() > 1:
            phi_state = phi_state.squeeze(0)
        return phi_state + changeset.delta
    
    def invert(self, changeset):
        """
        Invert a diff (like git revert).
        Produces a changeset that undoes the original change.
        """
        return PhaseChangeSet(
            phi_old=changeset.phi_new.clone(),
            phi_new=changeset.phi_old.clone(),
            delta=-changeset.delta,
            unchanged_mask=changeset.unchanged_mask,
            minor_mask=changeset.minor_mask,
            major_mask=changeset.major_mask
        )
    
    def create_tracker(self):
        """Create a version tracker for memory state history."""
        return PhaseVersionTracker(self)


class PhaseChangeSet:
    """
    Represents the diff between two phase states.
    Contains the delta, classification of changes, and statistics.
    """
    
    def __init__(self, phi_old, phi_new, delta, 
                 unchanged_mask, minor_mask, major_mask):
        self.phi_old = phi_old
        self.phi_new = phi_new
        self.delta = delta
        self.unchanged_mask = unchanged_mask
        self.minor_mask = minor_mask
        self.major_mask = major_mask
        self.hidden_size = delta.shape[0]
    
    def num_unchanged(self):
        return int(self.unchanged_mask.sum().item())
    
    def num_minor_changes(self):
        return int(self.minor_mask.sum().item())
    
    def num_major_changes(self):
        return int(self.major_mask.sum().item())
    
    def total_rotation(self):
        """Total absolute rotation across all neurons (radians)."""
        return self.delta.abs().sum().item()
    
    def mean_rotation(self):
        """Mean absolute rotation per neuron (radians)."""
        return self.delta.abs().mean().item()
    
    def max_rotation(self):
        """Maximum rotation of any single neuron."""
        return self.delta.abs().max().item()
    
    def most_changed_neurons(self, top_k=5):
        """Return indices of the most changed neurons."""
        values, indices = self.delta.abs().topk(min(top_k, self.hidden_size))
        return list(zip(indices.tolist(), values.tolist()))
    
    def summary(self):
        """Human-readable summary of the diff."""
        lines = [
            f"Phase Diff Summary ({self.hidden_size} neurons):",
            f"  Unchanged:     {self.num_unchanged():3d} neurons",
            f"  Minor changes: {self.num_minor_changes():3d} neurons",
            f"  Major changes: {self.num_major_changes():3d} neurons",
            f"  Total rotation:  {self.total_rotation():.3f} rad",
            f"  Mean rotation:   {self.mean_rotation():.3f} rad/neuron",
            f"  Max rotation:    {self.max_rotation():.3f} rad",
            f"  Most changed: {self.most_changed_neurons(3)}"
        ]
        return "\n".join(lines)
    
    def size_bytes(self):
        """Size of the diff in bytes (just the delta vector)."""
        return self.hidden_size * 4  # float32
    
    def export(self):
        """Export as a compact dict."""
        return {
            'delta': self.delta.cpu().numpy().tolist(),
            'hidden_size': self.hidden_size,
            'stats': {
                'unchanged': self.num_unchanged(),
                'minor': self.num_minor_changes(),
                'major': self.num_major_changes(),
                'total_rotation': self.total_rotation()
            }
        }
    
    @classmethod
    def from_export(cls, data):
        """Load from an exported dict."""
        delta = torch.tensor(data['delta'], dtype=torch.float32)
        abs_delta = delta.abs()
        return cls(
            phi_old=torch.zeros_like(delta),
            phi_new=delta,
            delta=delta,
            unchanged_mask=abs_delta < 0.01,
            minor_mask=(abs_delta >= 0.01) & (abs_delta < np.pi/4),
            major_mask=abs_delta >= np.pi/4
        )


class PhaseVersionTracker:
    """
    Git-like version control for Helix memory states.
    
    Usage:
        tracker = PhaseVersionTracker()
        tracker.commit(phi_v1, "initial state")
        tracker.commit(phi_v2, "after learning John's preferences")
        tracker.commit(phi_v3, "after John's allergy update")
        
        # View history
        tracker.log()
        
        # Rollback to a previous version
        restored = tracker.rollback(1)  # Go back to v2
        
        # Compare versions
        diff = tracker.diff_versions(0, 2)
    """
    
    def __init__(self, differ=None):
        self.differ = differ or PhaseDiff()
        self.versions = []  # List of (phi_state, message, timestamp)
        self.current_version = -1
    
    def commit(self, phi_state, message=""):
        """
        Save a new version of the memory state.
        
        Args:
            phi_state: tensor of shape (hidden_size,)
            message: description of what changed
        """
        if phi_state.dim() > 1:
            phi_state = phi_state.squeeze(0)
            
        import time
        self.versions.append({
            'phi': phi_state.detach().clone(),
            'message': message,
            'timestamp': time.time(),
            'version': len(self.versions)
        })
        self.current_version = len(self.versions) - 1
        return self.current_version
    
    def rollback(self, version_idx):
        """
        Restore a previous version of the memory.
        
        Args:
            version_idx: integer version number to restore
            
        Returns:
            phi: tensor of the restored phase state
        """
        assert 0 <= version_idx < len(self.versions), \
            f"Version {version_idx} doesn't exist. Range: 0-{len(self.versions)-1}"
        self.current_version = version_idx
        return self.versions[version_idx]['phi'].clone()
    
    def current(self):
        """Get the current version's phase state."""
        if self.current_version < 0:
            raise ValueError("No versions committed yet.")
        return self.versions[self.current_version]['phi'].clone()
    
    def diff_versions(self, version_a, version_b):
        """
        Compute the diff between two versions.
        
        Returns:
            PhaseChangeSet
        """
        phi_a = self.versions[version_a]['phi']
        phi_b = self.versions[version_b]['phi']
        return self.differ.diff(phi_a, phi_b)
    
    def log(self):
        """Print the version history."""
        lines = [f"Phase Version History ({len(self.versions)} versions):"]
        for v in self.versions:
            marker = " <-- HEAD" if v['version'] == self.current_version else ""
            lines.append(f"  v{v['version']}: {v['message']}{marker}")
        return "\n".join(lines)
    
    def num_versions(self):
        return len(self.versions)
