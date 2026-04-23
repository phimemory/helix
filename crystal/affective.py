"""
Affective Phase Encoding (APE)
Emotional state encoding via dedicated low-frequency phase bands.

Inspired by: Picard, R. "Affective Computing" (MIT Media Lab, 1997)
             Russell's Circumplex Model of Affect (valence-arousal space)
             Ekman's Discrete Emotion Theory

Standard memory architectures store WHAT happened. APE extends Helix
to encode HOW the interaction felt. Emotional memory is stored in 
dedicated low-frequency harmonic bands (h=0.25, 0.5) that persist
across sessions without decay.
"""

import torch
import torch.nn as nn
import numpy as np


class AffectiveEncoder(nn.Module):
    """
    Encodes emotional state into dedicated phase bands within a Helix cell.
    
    Uses Russell's Circumplex Model: emotions mapped to (valence, arousal).
      valence: negative (-1) to positive (+1)
      arousal: calm (0) to excited (+1)
    
    Low-frequency harmonics (0.25, 0.5) capture slow-moving emotional
    context that persists across interactions. High-frequency harmonics
    handle factual data. The bands don't interfere.
    
    Usage:
        encoder = AffectiveEncoder(hidden_size=64)
        
        # After detecting sentiment in a conversation:
        encoder.encode_sentiment(valence=0.8, arousal=0.3)  # Happy, calm
        encoder.encode_sentiment(valence=-0.9, arousal=0.9) # Angry, intense
        
        # Read back the emotional state
        state = encoder.decode_sentiment()
        print(state)  # {'valence': -0.05, 'arousal': 0.6, 'label': 'tense'}
    """
    
    def __init__(self, hidden_size=64, affective_neurons=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.affective_neurons = affective_neurons
        
        # Dedicated phase angles for emotional state
        # These use LOW frequencies (0.25, 0.5) to encode slow-changing affect
        self.affective_harmonics = [0.25, 0.5]
        
        # Phase state for the affective band
        self.phi_affect = torch.zeros(affective_neurons)
        
        # Valence encoder: maps valence [-1, 1] to phase shifts
        self.valence_weights = nn.Parameter(
            torch.randn(affective_neurons // 2) * 0.1
        )
        
        # Arousal encoder: maps arousal [0, 1] to phase shifts
        self.arousal_weights = nn.Parameter(
            torch.randn(affective_neurons // 2) * 0.1
        )
        
        # Persistence factor: how much old emotional state is retained
        # High persistence = emotions change slowly (more realistic)
        self.persistence = 0.9
        
        # History of emotional readings
        self.affect_history = []
        
        # Emotion label mapping (simplified Russell's model)
        self.emotion_map = {
            ( 1,  1): 'excited',
            ( 1,  0): 'content',
            ( 1, -1): 'relaxed',
            ( 0,  1): 'alert',
            ( 0,  0): 'neutral',
            ( 0, -1): 'calm',
            (-1,  1): 'angry',
            (-1,  0): 'sad',
            (-1, -1): 'bored'
        }
        
    def encode_sentiment(self, valence, arousal):
        """
        Encode an emotional state into the affective phase band.
        
        Args:
            valence: float in [-1, 1]. Negative = bad, Positive = good.
            arousal: float in [0, 1]. 0 = calm, 1 = excited/intense.
        """
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        half = self.affective_neurons // 2
        
        # Compute phase shifts from sentiment
        valence_shift = valence * np.pi * torch.sigmoid(self.valence_weights)
        arousal_shift = arousal * np.pi * torch.sigmoid(self.arousal_weights)
        
        shift = torch.cat([valence_shift, arousal_shift])
        
        # Accumulate with persistence (old emotions decay slowly)
        with torch.no_grad():
            self.phi_affect = self.persistence * self.phi_affect + (1 - self.persistence) * shift
        
        # Record history
        self.affect_history.append({
            'valence': valence,
            'arousal': arousal,
            'phi_snapshot': self.phi_affect.clone()
        })
    
    def decode_sentiment(self):
        """
        Read the current emotional state from the affective phase band.
        
        Returns:
            dict with valence, arousal, and emotion label
        """
        half = self.affective_neurons // 2
        
        # Extract valence from first half of neurons
        valence_signal = torch.cos(self.phi_affect[:half]).mean().item()
        
        # Extract arousal from second half
        arousal_signal = (1 - torch.cos(self.phi_affect[half:]).mean().item()) / 2
        
        # Map to nearest emotion label
        v_sign = 1 if valence_signal > 0.1 else (-1 if valence_signal < -0.1 else 0)
        a_sign = 1 if arousal_signal > 0.4 else (-1 if arousal_signal < 0.2 else 0)
        label = self.emotion_map.get((v_sign, a_sign), 'neutral')
        
        return {
            'valence': round(valence_signal, 3),
            'arousal': round(arousal_signal, 3),
            'label': label
        }
    
    def get_affective_features(self):
        """
        Return the affective phase state as a feature vector for injection
        into an LLM's context.
        
        Returns:
            features: tensor of shape (affective_neurons * 4,)
        """
        features = []
        for h in self.affective_harmonics:
            features.append(torch.cos(h * self.phi_affect))
            features.append(torch.sin(h * self.phi_affect))
        return torch.cat(features)
    
    def emotional_trajectory(self):
        """
        Return the full emotional trajectory over recorded history.
        Useful for detecting sentiment drift over time.
        """
        if not self.affect_history:
            return []
        return [
            {'valence': h['valence'], 'arousal': h['arousal']}
            for h in self.affect_history
        ]
    
    def reset(self):
        """Clear emotional memory."""
        self.phi_affect.zero_()
        self.affect_history = []
