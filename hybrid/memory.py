"""
HybridMemory — the product surface.

Architecture (honest):

  ┌─────────────┐     absorb      ┌──────────────────┐
  │  text event │ ──────────────► │ TrajectoryEncoder│──► φ (fingerprint)
  └─────────────┘                 │   (Helix cell)   │
         │                        └──────────────────┘
         │ store text                      │
         ▼                                 ▼ trained
  ┌─────────────┐                 ┌──────────────────┐
  │  FactStore  │◄── retrieve ───│   PhaseReadout    │──► class / recon / summary
  │ SQLite/HDB  │   by nearest φ  └──────────────────┘
  └─────────────┘
         │
         ▼
   LLM-readable context (actual text, not raw angles)

Claude was right that raw φ is not a language. This module never
asks an LLM to read φ. It retrieves text and optionally attaches
a compact summary vector / predicted label from the trained readout.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .embedder import HashEmbedder
from .encoder import TrajectoryEncoder
from .readout import PhaseReadout
from .store import BaseStore, Fact, FactStore, HelixDBStore, make_store, new_fact


@dataclass
class RememberResult:
    fact_id: str
    session_id: str
    step: int
    phi: List[float]
    label_hint: Optional[str] = None


@dataclass
class RecallResult:
    """What you hand an LLM / agent."""

    texts: List[str]
    facts: List[Fact]
    scores: List[float]
    predicted_label: Optional[str]
    summary: Optional[List[float]]
    phi: List[float]
    mode: str  # 'phi' | 'embed' | 'session'


class HybridMemory:
    """
    End-to-end hybrid memory.

    Usage:
        mem = HybridMemory()
        mem.remember("user_1", "ordered pizza with extra cheese", label="order")
        mem.remember("user_1", "asked about delivery time", label="support")
        ctx = mem.recall("user_1", query="pizza order")
        print(ctx.texts)  # real text for the LLM
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_size: int = 32,
        n_classes: int = 16,
        harmonics: Optional[List[float]] = None,
        store: Optional[BaseStore] = None,
        store_path: str | Path = ":memory:",
        backend: str = "sqlite",
        helixdb_url: Optional[str] = None,
        label_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.harmonics = harmonics or [1.0, 2.0, 4.0, 8.0]
        self.device = torch.device(device or "cpu")

        self.embedder = HashEmbedder(dim=embed_dim)
        self.encoder = TrajectoryEncoder(
            input_size=embed_dim,
            hidden_size=hidden_size,
            harmonics=self.harmonics,
        )
        self.readout = PhaseReadout(
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            n_classes=n_classes,
            harmonics=self.harmonics,
        ).to(self.device)

        self.store = store or make_store(backend, store_path, helixdb_url)
        self.label_names = label_names or [f"class_{i}" for i in range(n_classes)]
        self._label_to_id = {n: i for i, n in enumerate(self.label_names)}
        self._session_steps: Dict[str, int] = {}
        self._session_phi: Dict[str, torch.Tensor] = {}
        self.readout_trained = False

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------

    def register_labels(self, names: Sequence[str]) -> None:
        self.label_names = list(names)
        self.n_classes = len(self.label_names)
        self._label_to_id = {n: i for i, n in enumerate(self.label_names)}
        # rebuild class head if size changed
        old = self.readout
        self.readout = PhaseReadout(
            hidden_size=self.hidden_size,
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            harmonics=self.harmonics,
        ).to(self.device)
        # keep backbone if same feat dim
        try:
            self.readout.backbone.load_state_dict(old.backbone.state_dict())
            self.readout.recon_head.load_state_dict(old.recon_head.state_dict())
            self.readout.summary_head.load_state_dict(old.summary_head.state_dict())
        except Exception:
            pass

    def _label_id(self, label: str) -> int:
        if label not in self._label_to_id:
            # dynamic expand for open labels (maps to last bucket if full)
            if len(self.label_names) < self.n_classes:
                self.label_names.append(label)
                self._label_to_id[label] = len(self.label_names) - 1
            else:
                return self.n_classes - 1
        return self._label_to_id[label]

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    def remember(
        self,
        session_id: str,
        text: str,
        label: str = "event",
        meta: Optional[dict] = None,
    ) -> RememberResult:
        """
        Absorb one event into the session trajectory and store the fact.
        """
        emb = self.embedder.embed(text)

        # per-session crystal: track step count; re-encode full session
        # for correct cumulative φ (simple + correct for demo scale)
        step = self._session_steps.get(session_id, 0)
        facts_before = self.store.list_session(session_id)

        # rebuild trajectory from history + new event
        self.encoder.reset()
        for f in facts_before:
            self.encoder.absorb(torch.tensor(f.embedding, dtype=torch.float32))
        self.encoder.absorb(emb)
        phi = self.encoder.phi().detach()
        self._session_phi[session_id] = phi.clone()
        self._session_steps[session_id] = step + 1

        fact = new_fact(
            session_id=session_id,
            text=text,
            label=label,
            step=step,
            embedding=emb,
            phi=phi,
            meta=meta,
        )
        self.store.add(fact)

        pred = None
        if self.readout_trained:
            with torch.no_grad():
                cid = self.readout.predict_class(phi.to(self.device))
                if cid < len(self.label_names):
                    pred = self.label_names[cid]

        return RememberResult(
            fact_id=fact.id,
            session_id=session_id,
            step=step,
            phi=phi.tolist(),
            label_hint=pred,
        )

    # ------------------------------------------------------------------
    # read path (LLM-facing)
    # ------------------------------------------------------------------

    def recall(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 5,
        mode: str = "auto",
    ) -> RecallResult:
        """
        Retrieve readable text for an agent/LLM.

        mode:
          'auto'    — use phi search if session trajectory exists, else embed
          'phi'     — nearest trajectory snapshots
          'embed'   — nearest fact embeddings
          'session' — full session timeline in order
        """
        if mode == "auto":
            if query is None and session_id is not None:
                mode = "session"
            elif session_id is not None and session_id in self._session_phi:
                mode = "phi"
            else:
                mode = "embed"

        facts: List[Fact] = []
        scores: List[float] = []
        phi_list: List[float] = []

        if mode == "session":
            if not session_id:
                raise ValueError("session_id required for mode=session")
            facts = self.store.list_session(session_id)
            scores = [1.0] * len(facts)
            if session_id in self._session_phi:
                phi_list = self._session_phi[session_id].tolist()
            elif facts:
                phi_list = facts[-1].phi

        elif mode == "phi":
            if session_id and session_id in self._session_phi:
                q_phi = self._session_phi[session_id]
            elif query:
                # encode query as single-step φ for search key
                emb = self.embedder.embed(query)
                q_phi = self.encoder.encode([emb])
            else:
                raise ValueError("need session_id trajectory or query for phi recall")
            phi_list = q_phi.tolist()
            hits = self.store.search_by_phi(phi_list, top_k=top_k, session_id=session_id)
            facts = [h[0] for h in hits]
            scores = [h[1] for h in hits]

        else:  # embed
            if not query:
                raise ValueError("query text required for embed recall")
            emb = self.embedder.embed(query)
            hits = self.store.search_by_embedding(
                emb.tolist(), top_k=top_k, session_id=session_id
            )
            facts = [h[0] for h in hits]
            scores = [h[1] for h in hits]
            if session_id and session_id in self._session_phi:
                phi_list = self._session_phi[session_id].tolist()

        predicted = None
        summary = None
        if phi_list and self.readout_trained:
            with torch.no_grad():
                phi_t = torch.tensor(phi_list, dtype=torch.float32, device=self.device)
                recon, logits, summ = self.readout(phi_t)
                cid = int(logits.view(-1).argmax().item())
                if cid < len(self.label_names):
                    predicted = self.label_names[cid]
                summary = summ.view(-1).cpu().tolist()

        return RecallResult(
            texts=[f.text for f in facts],
            facts=facts,
            scores=scores,
            predicted_label=predicted,
            summary=summary,
            phi=phi_list,
            mode=mode,
        )

    def context_block(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        Ready-to-paste string for an LLM system/user message.
        Never dumps raw phase angles as 'memory'.
        """
        result = self.recall(session_id=session_id, query=query, top_k=top_k)
        lines = ["## Retrieved memory (Helix hybrid)"]
        if result.predicted_label:
            lines.append(f"Trajectory label: {result.predicted_label}")
        lines.append(f"Retrieval mode: {result.mode}")
        if not result.texts:
            lines.append("(no facts yet)")
        else:
            for i, (text, score) in enumerate(zip(result.texts, result.scores), 1):
                lines.append(f"{i}. ({score:.3f}) {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # training hooks
    # ------------------------------------------------------------------

    def attach_trained_readout(self, readout: PhaseReadout, label_names: Sequence[str]) -> None:
        self.readout = readout.to(self.device)
        self.label_names = list(label_names)
        self.n_classes = len(self.label_names)
        self._label_to_id = {n: i for i, n in enumerate(self.label_names)}
        self.readout_trained = True

    def collect_training_pairs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From stored facts: (phi, embedding, class_id) for each fact.
        Uses cumulative φ stored at each step.
        """
        facts = self.store.all_facts()
        if not facts:
            raise RuntimeError("No facts stored — call remember() first")
        phis, embs, labels = [], [], []
        for f in facts:
            phis.append(torch.tensor(f.phi, dtype=torch.float32))
            embs.append(torch.tensor(f.embedding, dtype=torch.float32))
            labels.append(self._label_id(f.label))
        return (
            torch.stack(phis),
            torch.stack(embs),
            torch.tensor(labels, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "readout": self.readout.state_dict(),
                "encoder_cell": self.encoder.crystal.cell.state_dict(),
                "label_names": self.label_names,
                "embed_dim": self.embed_dim,
                "hidden_size": self.hidden_size,
                "n_classes": self.n_classes,
                "harmonics": self.harmonics,
                "readout_trained": self.readout_trained,
                "session_steps": self._session_steps,
                "session_phi": {k: v.cpu() for k, v in self._session_phi.items()},
            },
            directory / "hybrid.pt",
        )
        # dump facts if sqlite file not already durable
        facts_path = directory / "facts.json"
        with open(facts_path, "w", encoding="utf-8") as f:
            json.dump([x.to_dict() for x in self.store.all_facts()], f, indent=2)
        return directory

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        blob = torch.load(directory / "hybrid.pt", map_location="cpu", weights_only=False)
        self.embed_dim = blob["embed_dim"]
        self.hidden_size = blob["hidden_size"]
        self.n_classes = blob["n_classes"]
        self.harmonics = blob["harmonics"]
        self.label_names = blob["label_names"]
        self._label_to_id = {n: i for i, n in enumerate(self.label_names)}
        self.embedder = HashEmbedder(dim=self.embed_dim)
        self.encoder = TrajectoryEncoder(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            harmonics=self.harmonics,
        )
        self.encoder.crystal.cell.load_state_dict(blob["encoder_cell"])
        self.readout = PhaseReadout(
            hidden_size=self.hidden_size,
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            harmonics=self.harmonics,
        )
        self.readout.load_state_dict(blob["readout"])
        self.readout.to(self.device)
        self.readout_trained = bool(blob.get("readout_trained", False))
        self._session_steps = dict(blob.get("session_steps", {}))
        self._session_phi = {
            k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
            for k, v in blob.get("session_phi", {}).items()
        }

        facts_path = directory / "facts.json"
        if facts_path.exists():
            with open(facts_path, encoding="utf-8") as f:
                raw = json.load(f)
            self.store.clear()
            for item in raw:
                self.store.add(Fact(**item))

    def stats(self) -> dict:
        st = {
            "facts": len(self.store.all_facts()),
            "sessions": len(self._session_steps),
            "readout_trained": self.readout_trained,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "labels": self.label_names,
        }
        if isinstance(self.store, HelixDBStore):
            st["helixdb"] = self.store.status()
        return st

    def __repr__(self) -> str:
        return f"HybridMemory({self.stats()})"
