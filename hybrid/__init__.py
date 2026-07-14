"""
Helix Hybrid Memory
===================

Closes the encoder-only gap:

  text/events  →  Helix phase φ  (trajectory fingerprint)
  facts        →  FactStore      (SQLite or HelixDB) — the actual payload
  φ            →  PhaseReadout   (trained decoder + classifier)
  query        →  nearest φ      →  stored text returned

Honest claim:
  Helix is the sequence/trajectory codec.
  The fact store holds readable content.
  A trained readout makes φ useful to any downstream model.
  LLMs never "speak phase" — they receive retrieved text + optional summary.
"""

from .embedder import HashEmbedder
from .encoder import TrajectoryEncoder
from .readout import PhaseReadout
from .store import FactStore, HelixDBStore
from .memory import HybridMemory
from .train import train_readout, make_synthetic_sessions

__all__ = [
    "HashEmbedder",
    "TrajectoryEncoder",
    "PhaseReadout",
    "FactStore",
    "HelixDBStore",
    "HybridMemory",
    "train_readout",
    "make_synthetic_sessions",
]
