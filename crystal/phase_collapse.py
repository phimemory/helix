"""
Phase Collapse Events (PCE)
Irreversible binary state transitions encoded as permanent pi-flips.

Inspired by: Wave function collapse (Copenhagen interpretation)
             Bistable dynamical systems in nonlinear physics

A Phase Collapse is an instantaneous, permanent, irreversible state change.
Once a neuron collapses, it can never be overwritten by any subsequent input.
This guarantees that critical binary facts (out-of-stock, status changes,
permanent flags) are encoded with absolute certainty.
"""

import torch
import numpy as np


class PhaseCollapseRegister:
    """
    A register of dedicated Helix neurons that can undergo irreversible
    pi-flip Phase Collapse Events.
    
    Each neuron represents a binary fact. Uncollapsed = 0 (False).
    Collapsed = pi (True). Once collapsed, permanently frozen.
    
    Usage:
        pcr = PhaseCollapseRegister(num_flags=16)
        pcr.collapse(0)           # Flag 0 is now permanently True
        pcr.query(0)              # True
        pcr.collapse(0)           # No-op, already collapsed
        pcr.attempt_overwrite(0)  # Raises error — irreversible
        
        # Named flags
        pcr.register_flag(3, "salmon_out_of_stock")
        pcr.collapse_named("salmon_out_of_stock")
        pcr.query_named("salmon_out_of_stock")  # True
    """
    
    def __init__(self, num_flags=16):
        self.num_flags = num_flags
        self.phase_register = torch.zeros(num_flags)
        self.frozen_mask = torch.zeros(num_flags, dtype=torch.bool)
        self.flag_names = {}
        self.collapse_timestamps = {}
        
    def register_flag(self, idx, name):
        """Assign a human-readable name to a flag index."""
        assert idx < self.num_flags, f"Index {idx} exceeds register size {self.num_flags}"
        self.flag_names[name] = idx
        
    def collapse(self, idx):
        """
        Execute an irreversible pi-flip on neuron idx.
        Once collapsed, the neuron is permanently frozen.
        
        Returns True if newly collapsed, False if already collapsed.
        """
        assert idx < self.num_flags, f"Index {idx} exceeds register size {self.num_flags}"
        
        if self.frozen_mask[idx]:
            return False  # Already collapsed
            
        self.phase_register[idx] = np.pi
        self.frozen_mask[idx] = True
        self.collapse_timestamps[idx] = torch.tensor(float('inf'))  # Permanent
        return True
    
    def collapse_named(self, name):
        """Collapse a flag by its registered name."""
        assert name in self.flag_names, f"Unknown flag: {name}"
        return self.collapse(self.flag_names[name])
    
    def query(self, idx):
        """Check if a flag is collapsed (True) or not (False)."""
        return bool(self.frozen_mask[idx])
    
    def query_named(self, name):
        """Query a flag by its registered name."""
        assert name in self.flag_names, f"Unknown flag: {name}"
        return self.query(self.flag_names[name])
    
    def attempt_overwrite(self, idx, value=0.0):
        """
        Attempt to overwrite a collapsed neuron.
        Raises AssertionError if the neuron is frozen — proving irreversibility.
        """
        if self.frozen_mask[idx]:
            raise AssertionError(
                f"Phase Collapse is IRREVERSIBLE. Neuron {idx} is permanently "
                f"frozen at pi. Cannot overwrite."
            )
        self.phase_register[idx] = value
    
    def get_state_vector(self):
        """
        Return the full register as a feature vector.
        Collapsed neurons contribute cos(pi) = -1.
        Uncollapsed neurons contribute cos(0) = 1.
        """
        return torch.cos(self.phase_register)
    
    def summary(self):
        """Human-readable summary of all flags."""
        lines = []
        for i in range(self.num_flags):
            status = "COLLAPSED" if self.frozen_mask[i] else "open"
            name = ""
            for n, idx in self.flag_names.items():
                if idx == i:
                    name = f" ({n})"
                    break
            lines.append(f"  [{i}]{name}: {status}")
        return f"PhaseCollapseRegister ({self.num_flags} flags):\n" + "\n".join(lines)
    
    def num_collapsed(self):
        return int(self.frozen_mask.sum().item())
    
    def export_state(self):
        """Export as a dict for serialization."""
        return {
            'phase_register': self.phase_register.clone(),
            'frozen_mask': self.frozen_mask.clone(),
            'flag_names': dict(self.flag_names),
            'num_flags': self.num_flags
        }
    
    def load_state(self, state_dict):
        """Load from a serialized dict."""
        self.phase_register = state_dict['phase_register']
        self.frozen_mask = state_dict['frozen_mask']
        self.flag_names = state_dict['flag_names']
        self.num_flags = state_dict['num_flags']
