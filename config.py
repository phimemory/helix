# HELIX Configuration
import math

def get_lock_strength(epoch, total_epochs, peak_strength=0.125, floor_strength=0.03125):
    """gaussian annealing for quantization strength scheduling."""
    mu = total_epochs / 2.0
    sigma = total_epochs / 6.0
    factor = math.exp(-0.5 * ((epoch - mu) / sigma) ** 2)
    floor = floor_strength if epoch > (total_epochs * 0.5) else 0.0
    return max(peak_strength * factor, floor)

# ==============================================================================
# DEFAULTS
# ==============================================================================
HIDDEN_SIZE = 32
PEAK_LOCKING_STRENGTH = 0.125
HARMONICS = [1, 2, 4, 8]
LR = 0.001953125 # 2^-9
EPOCHS_SHORT = 400
EPOCHS_LONG = 400
SPIN_FACTOR = 0.5
WOBBLE_GRAVITY = 0.1
WOBBLE_HARMONICS = [1]
WOBBLE_COUPLING = -1.0
FULL_STATE = True

# ==============================================================================
# DENSITY DUEL
# ==============================================================================
HIDDEN_SIZE_H = 64   # helix
HIDDEN_SIZE_G = 256   # gru baseline

def get_fair_hidden(helix_size):
    return helix_size * 4

# ==============================================================================
# PER TASK CONFIGS
# ==============================================================================

TOPOLOGY_CONFIG = {
    'HIDDEN_H': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.0625,
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.015625,
    'LR': 0.001953125,
    'WOBBLE': False,
    'SORT_INPUTS': True,
    'DELAYED_LOCKING': 0.5
}

PARITY_CONFIG = {
    'HIDDEN_H': 1,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.01,
    'DELAYED_LOCKING': 0.5
}

BRACKETS_CONFIG = {
    'HIDDEN_H': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.001953125,
    'TERMINAL_ONLY': True
}

COLORS_CONFIG = {
    'HIDDEN_H': 128,
    'HIDDEN_G': 512,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 800,
    'FLOOR': 0.015625,
    'LR': 0.0009765625,
    'WOBBLE': True,
    'WOBBLE_GRAVITY': 0.0,
    'WOBBLE_COUPLING': -1.0,
    'TERMINAL_ONLY': False
}

ASCII_CONFIG = {
    'HIDDEN_H': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.5,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 400,
    'FLOOR': 0.125,
    'LR': 0.001953125,
    'WOBBLE_GRAVITY': 0.1,
    'WOBBLE_COUPLING': -1.0,
    'TERMINAL_ONLY': False
}

ORACLE_CONFIG = {
    'HIDDEN_H': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.0009765625,
    'TERMINAL_ONLY': True
}

PERMS_CONFIG = {
    'HIDDEN_H': 64,
    'HIDDEN_G': 256,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1],
    'EPOCHS': 1500,
    'LR': 0.0009765625,
    'FLOOR': 0.03125,
    'RUNS': 3
}

LONG_TERM_CONFIG = {
    'HIDDEN_H': 64,
    'HIDDEN_G': 256,
    'PEAK_LOCKING_STRENGTH': 0.5,
    'FLOOR': 0.125,
    'EPOCHS': 12500,
    'LR': 0.000244140625
}

MOD17_CONFIG = {
    'MODULUS': 17,
    'HIDDEN_SIZE': 128,
    'LR': 0.001,
    'THRESHOLD': 0.95,
    'CLIP_GRAD': 0.5,
    'EPOCHS': 3000,
    'BATCH_SIZE': 64,
    'MAX_SEQ_LEN': 200
}

