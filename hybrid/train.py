"""
Train PhaseReadout so φ becomes readable.

Synthetic multi-session event streams → Helix encode → train decoder.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.optim as optim

from .embedder import HashEmbedder
from .encoder import TrajectoryEncoder
from .readout import PhaseReadout


# Curated event templates per intent class — order matters in sessions
DEFAULT_TEMPLATES: Dict[str, List[str]] = {
    "order_food": [
        "I want to order a large pizza",
        "add extra cheese and olives",
        "deliver to 42 market street",
        "pay with card ending 4242",
        "confirm my pizza order please",
    ],
    "support_billing": [
        "there is a wrong charge on my bill",
        "I was billed twice this month",
        "please refund the duplicate payment",
        "send me a corrected invoice",
        "confirm the refund is processing",
    ],
    "account_login": [
        "I cannot log into my account",
        "reset my password email",
        "enable two factor authentication",
        "lock suspicious sessions",
        "confirm my account is secure",
    ],
    "shipping": [
        "where is my package",
        "tracking number is late",
        "reroute delivery to office",
        "package shows delivered but missing",
        "open a shipping claim",
    ],
    "product_question": [
        "does this laptop support 64gb ram",
        "what is the battery life",
        "compare the air and pro models",
        "is student discount available",
        "add the pro model to wishlist",
    ],
    "cancel": [
        "I need to cancel my subscription",
        "stop the monthly renewal",
        "confirm cancellation effective date",
        "export my data before leaving",
        "sorry to see me go survey",
    ],
    "booking": [
        "book a table for four at seven",
        "window seat if possible",
        "add a birthday cake note",
        "confirm reservation under Patel",
        "send calendar invite for dinner",
    ],
    "complaint": [
        "the driver was extremely rude",
        "food arrived cold and late",
        "this is unacceptable service",
        "I want a manager callback",
        "escalate this complaint now",
    ],
}


def make_synthetic_sessions(
    templates: Dict[str, List[str]] | None = None,
    sessions_per_class: int = 40,
    min_len: int = 3,
    max_len: int = 5,
    seed: int = 42,
) -> List[Tuple[str, List[str]]]:
    """
    Returns list of (label, list_of_event_texts).
    """
    rng = random.Random(seed)
    templates = templates or DEFAULT_TEMPLATES
    sessions: List[Tuple[str, List[str]]] = []
    for label, lines in templates.items():
        for _ in range(sessions_per_class):
            n = rng.randint(min_len, min(max_len, len(lines)))
            # keep causal order but maybe drop some middle steps
            idxs = sorted(rng.sample(range(len(lines)), n))
            events = [lines[i] for i in idxs]
            # light paraphrase noise
            if rng.random() < 0.3:
                events = [e + rng.choice(["", " please", " thanks", " asap"]) for e in events]
            sessions.append((label, events))
    rng.shuffle(sessions)
    return sessions


def encode_sessions(
    sessions: Sequence[Tuple[str, List[str]]],
    embedder: HashEmbedder,
    encoder: TrajectoryEncoder,
    label_to_id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each session, absorb events step-by-step and emit a training
    pair at every step: (phi_t, emb_t, class_id).
    """
    phis, embs, labels = [], [], []
    for label, events in sessions:
        cid = label_to_id[label]
        encoder.reset()
        for text in events:
            emb = embedder.embed(text)
            encoder.absorb(emb)
            phis.append(encoder.phi().detach().clone())
            embs.append(emb.detach().clone())
            labels.append(cid)
    return (
        torch.stack(phis),
        torch.stack(embs),
        torch.tensor(labels, dtype=torch.long),
    )


def train_readout(
    sessions: Sequence[Tuple[str, List[str]]] | None = None,
    embed_dim: int = 128,
    hidden_size: int = 32,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[PhaseReadout, TrajectoryEncoder, HashEmbedder, List[str], dict]:
    """
    Train PhaseReadout on synthetic (or provided) sessions.

    Returns:
      readout, encoder, embedder, label_names, metrics
    """
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    sessions = list(sessions) if sessions is not None else make_synthetic_sessions()
    label_names = sorted({lab for lab, _ in sessions})
    label_to_id = {n: i for i, n in enumerate(label_names)}

    embedder = HashEmbedder(dim=embed_dim)
    encoder = TrajectoryEncoder(input_size=embed_dim, hidden_size=hidden_size)
    readout = PhaseReadout(
        hidden_size=hidden_size,
        embed_dim=embed_dim,
        n_classes=len(label_names),
        harmonics=[1, 2, 4, 8],
    ).to(device)

    if verbose:
        print(f"Encoding {len(sessions)} sessions…")
    X_phi, Y_emb, Y_cls = encode_sessions(sessions, embedder, encoder, label_to_id)

    # train/val split
    n = X_phi.size(0)
    perm = torch.randperm(n)
    n_val = max(1, n // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    opt = optim.Adam(readout.parameters(), lr=lr)
    history = []

    def run_epoch(indices, train: bool) -> dict:
        readout.train(train)
        total = 0.0
        cos_sum = 0.0
        correct = 0
        count = 0
        # mini-batches
        idx = indices[torch.randperm(len(indices))] if train else indices
        for start in range(0, len(idx), batch_size):
            batch = idx[start : start + batch_size]
            phi = X_phi[batch].to(device)
            emb = Y_emb[batch].to(device)
            cls = Y_cls[batch].to(device)
            if train:
                opt.zero_grad()
            loss, stats = readout.loss(phi, emb, cls)
            if train:
                loss.backward()
                opt.step()
            total += stats["loss"] * len(batch)
            cos_sum += stats["mean_cosine"] * len(batch)
            with torch.no_grad():
                pred = readout.classify(phi).argmax(dim=-1)
                correct += int((pred == cls).sum().item())
            count += len(batch)
        return {
            "loss": total / max(count, 1),
            "cosine": cos_sum / max(count, 1),
            "acc": correct / max(count, 1),
        }

    best_val_acc = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        tr = run_epoch(train_idx, train=True)
        va = run_epoch(val_idx, train=False)
        history.append({"epoch": ep, "train": tr, "val": va})
        if va["acc"] >= best_val_acc:
            best_val_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == 1 or ep == epochs):
            print(
                f"  epoch {ep:3d}  "
                f"train loss={tr['loss']:.4f} cos={tr['cosine']:.3f} acc={tr['acc']:.1%}  "
                f"val loss={va['loss']:.4f} cos={va['cosine']:.3f} acc={va['acc']:.1%}"
            )

    if best_state is not None:
        readout.load_state_dict(best_state)

    # final eval
    readout.eval()
    with torch.no_grad():
        phi = X_phi[val_idx].to(device)
        emb = Y_emb[val_idx].to(device)
        cls = Y_cls[val_idx].to(device)
        recon, logits, _ = readout(phi)
        cos = torch.nn.functional.cosine_similarity(recon, emb, dim=-1).mean().item()
        acc = (logits.argmax(-1) == cls).float().mean().item()

    metrics = {
        "val_accuracy": acc,
        "val_cosine": cos,
        "best_val_acc": best_val_acc,
        "n_samples": n,
        "n_classes": len(label_names),
        "label_names": label_names,
        "history": history,
    }
    if verbose:
        print(
            f"\nDone. val class accuracy={acc:.1%}  "
            f"val recon cosine={cos:.3f}  classes={label_names}"
        )
    return readout, encoder, embedder, label_names, metrics


if __name__ == "__main__":
    train_readout(epochs=60, verbose=True)
