"""
TrajectoryEncoder — wraps MemoryCrystal as a pure sequence → φ codec.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch

# repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crystal.substrate import MemoryCrystal


class TrajectoryEncoder:
    """
    Encodes an ordered list of embeddings into a Helix phase state.

    This is the *encoder* half. It does not recover text by itself.
    Pair with PhaseReadout + FactStore for a full memory system.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 32,
        harmonics: Optional[List[float]] = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics or [1, 2, 4, 8]
        self.crystal = MemoryCrystal(
            input_size=input_size,
            hidden_size=hidden_size,
            harmonics=self.harmonics,
        )

    def reset(self) -> None:
        self.crystal.reset()

    def absorb(self, embedding: torch.Tensor) -> torch.Tensor:
        """Absorb one step. Returns confidence (diagnostic)."""
        if embedding.dim() == 1:
            emb = embedding
        else:
            emb = embedding.squeeze(0)
        if emb.numel() != self.input_size:
            raise ValueError(
                f"embedding dim {emb.numel()} != input_size {self.input_size}"
            )
        return self.crystal.absorb(emb)

    def absorb_sequence(self, embeddings: Iterable[torch.Tensor]) -> torch.Tensor:
        """Absorb many steps; return final φ."""
        for emb in embeddings:
            self.absorb(emb)
        return self.phi()

    def encode(self, embeddings: Iterable[torch.Tensor], reset: bool = True) -> torch.Tensor:
        """Fresh encode of a sequence → final phase state."""
        if reset:
            self.reset()
        return self.absorb_sequence(embeddings)

    def phi(self) -> torch.Tensor:
        """Raw phase angles (hidden_size,)."""
        return self.crystal.recall_compact()

    def features(self) -> torch.Tensor:
        """Harmonic expansion (hidden * n_harmonics * 2,)."""
        return self.crystal.recall()

    def absorb_count(self) -> int:
        return self.crystal.absorb_count

    def state_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "harmonics": list(self.harmonics),
            "phi": self.phi().detach().cpu(),
            "absorb_count": self.crystal.absorb_count,
            "cell": self.crystal.cell.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.input_size = state["input_size"]
        self.hidden_size = state["hidden_size"]
        self.harmonics = list(state["harmonics"])
        self.crystal = MemoryCrystal(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            harmonics=self.harmonics,
        )
        self.crystal.cell.load_state_dict(state["cell"])
        phi = state["phi"]
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float32)
        self.crystal.phi_state = phi.view(1, -1).clone()
        self.crystal.absorb_count = int(state.get("absorb_count", 0))

    def __repr__(self) -> str:
        return (
            f"TrajectoryEncoder(in={self.input_size}, "
            f"hidden={self.hidden_size}, steps={self.crystal.absorb_count})"
        )
