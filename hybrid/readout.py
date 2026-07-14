"""
PhaseReadout — the missing decoder half.

Turns φ into:
  1. reconstructed embedding  (autoencoder path)
  2. event-class logits         (classification path)
  3. optional short summary vec (for LLM-facing context packing)

After training, φ is no longer "unreadable" — it maps to concrete
signals any downstream system (including an LLM) can use.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def harmonic_features(phi: torch.Tensor, harmonics: List[float]) -> torch.Tensor:
    """
    phi: (batch, hidden) or (hidden,)
    returns: (batch, hidden * len(harmonics) * 2)
    """
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    parts = []
    for h in harmonics:
        parts.append(torch.cos(h * phi))
        parts.append(torch.sin(h * phi))
    return torch.cat(parts, dim=-1)


class PhaseReadout(nn.Module):
    """
    Trained map: phase → embedding reconstruction + class logits.

    This is what makes Claude's critique wrong *after* training:
    phase angles become a latent with a learned inverse.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        embed_dim: int = 128,
        n_classes: int = 8,
        harmonics: Optional[List[float]] = None,
        summary_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.harmonics = harmonics or [1, 2, 4, 8]
        self.summary_dim = summary_dim

        feat_dim = hidden_size * len(self.harmonics) * 2
        self.feat_dim = feat_dim

        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
        )
        mid = feat_dim // 2
        self.recon_head = nn.Linear(mid, embed_dim)
        self.class_head = nn.Linear(mid, n_classes)
        self.summary_head = nn.Linear(mid, summary_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_features(self, phi: torch.Tensor) -> torch.Tensor:
        return harmonic_features(phi, self.harmonics)

    def forward(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          recon:   (B, embed_dim) reconstructed embedding
          logits:  (B, n_classes)
          summary: (B, summary_dim)
        """
        feats = self.encode_features(phi)
        h = self.backbone(feats)
        recon = self.recon_head(h)
        # L2-normalize recon so it lives on same sphere as HashEmbedder
        recon = F.normalize(recon, dim=-1)
        logits = self.class_head(h)
        summary = self.summary_head(h)
        return recon, logits, summary

    def reconstruct(self, phi: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self.forward(phi)
        return recon

    def classify(self, phi: torch.Tensor) -> torch.Tensor:
        _, logits, _ = self.forward(phi)
        return logits

    def predict_class(self, phi: torch.Tensor) -> int:
        logits = self.classify(phi)
        if logits.dim() == 1:
            return int(logits.argmax().item())
        return int(logits[0].argmax().item())

    def summary_vector(self, phi: torch.Tensor) -> torch.Tensor:
        _, _, summary = self.forward(phi)
        return summary.squeeze(0) if summary.dim() > 1 and summary.size(0) == 1 else summary

    def loss(
        self,
        phi: torch.Tensor,
        target_embed: torch.Tensor,
        target_class: torch.Tensor,
        recon_weight: float = 1.0,
        class_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        recon, logits, _ = self.forward(phi)
        # cosine distance as reconstruction loss (embeddings are unit-norm)
        cos = F.cosine_similarity(recon, target_embed, dim=-1)
        recon_loss = (1.0 - cos).mean()
        class_loss = F.cross_entropy(logits, target_class)
        total = recon_weight * recon_loss + class_weight * class_loss
        stats = {
            "loss": float(total.detach()),
            "recon_loss": float(recon_loss.detach()),
            "class_loss": float(class_loss.detach()),
            "mean_cosine": float(cos.detach().mean()),
        }
        return total, stats
