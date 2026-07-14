from .helix import (
    HelixCell,
    HelixModel,
    HelixEncoderModel,
    HelixNeuronCell,
    HelixNeuronModel,
    landauer_loss,
    HARMONICS_STANDARD,
    HARMONICS_7OCTAVE,
    HARMONICS_SPINOR,
)
from .crystal.memory import HelixMemory

# Hybrid memory product surface (encoder + trained readout + fact store)
try:
    from .hybrid import HybridMemory, train_readout, PhaseReadout
except Exception:  # pragma: no cover - optional during partial installs
    HybridMemory = None  # type: ignore
    train_readout = None  # type: ignore
    PhaseReadout = None  # type: ignore
