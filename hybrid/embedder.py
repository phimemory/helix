"""
Deterministic text embedder — no heavyweight deps required.

Produces fixed-dim float vectors from text so Helix can absorb
sequences offline. Swap for sentence-transformers / OpenAI later
by implementing the same interface.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List

import numpy as np
import torch


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """
    Feature-hashed bag-of-words + bigrams → L2-normalized vector.

    Stable across runs (pure hash). Good enough for demos and
    structure-sensitive tasks; not a semantic SOTA embedder.
    """

    def __init__(self, dim: int = 128, seed: int = 7):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.seed = seed

    def _hash_index(self, token: str) -> int:
        raw = f"{self.seed}:{token}".encode("utf-8")
        h = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def _hash_sign(self, token: str) -> float:
        raw = f"sign:{self.seed}:{token}".encode("utf-8")
        h = hashlib.blake2b(raw, digest_size=1).digest()[0]
        return 1.0 if (h & 1) else -1.0

    def embed(self, text: str) -> torch.Tensor:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        if not toks:
            toks = ["_empty_"]

        for t in toks:
            i = self._hash_index(t)
            vec[i] += self._hash_sign(t)

        # bigrams — order sensitivity for short phrases
        for a, b in zip(toks, toks[1:]):
            bg = f"{a}_{b}"
            i = self._hash_index(bg)
            vec[i] += 0.5 * self._hash_sign(bg)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return torch.from_numpy(vec)

    def embed_batch(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.stack([self.embed(t) for t in texts], dim=0)

    def __repr__(self) -> str:
        return f"HashEmbedder(dim={self.dim})"
