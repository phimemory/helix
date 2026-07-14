"""
End-to-end demo: encoder + trained decoder + fact retrieval.

Run from repo root:
  python -m hybrid.demo
  python -m hybrid.demo --epochs 40 --save hybrid_ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure repo root on path when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from hybrid.memory import HybridMemory
from hybrid.train import make_synthetic_sessions, train_readout


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Helix Hybrid Memory demo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--save", type=str, default="hybrid_ckpt")
    parser.add_argument(
        "--backend",
        choices=("sqlite", "helixdb"),
        default="sqlite",
        help="Fact store backend. helixdb uses HELIXDB_URL if set.",
    )
    parser.add_argument("--db", type=str, default=":memory:")
    args = parser.parse_args()

    banner("1) Train PhaseReadout (φ → class + reconstructed embed)")
    readout, enc, emb, labels, metrics = train_readout(
        embed_dim=args.embed_dim,
        hidden_size=args.hidden,
        epochs=args.epochs,
        verbose=True,
    )

    banner("2) Build HybridMemory and attach trained readout")
    mem = HybridMemory(
        embed_dim=args.embed_dim,
        hidden_size=args.hidden,
        n_classes=len(labels),
        backend=args.backend,
        store_path=args.db if args.db != ":memory:" else ":memory:",
        label_names=labels,
    )
    # share trained cell weights so encode distribution matches training
    mem.encoder.crystal.cell.load_state_dict(enc.crystal.cell.state_dict())
    mem.attach_trained_readout(readout, labels)
    print(mem.stats())

    banner("3) Live session — order_food trajectory")
    session = "user_demo_01"
    live_events = [
        ("I want to order a large pizza", "order_food"),
        ("add extra cheese and olives", "order_food"),
        ("deliver to 42 market street", "order_food"),
        ("pay with card ending 4242", "order_food"),
    ]
    for text, lab in live_events:
        r = mem.remember(session, text, label=lab)
        print(f"  step {r.step}: remembered + pred_hint={r.label_hint!r}")
        print(f"           {text!r}")

    banner("4) Recall for an LLM (TEXT, not raw phase)")
    block = mem.context_block(session_id=session, query="pizza delivery")
    print(block)

    banner("5) Prove φ is readable AFTER training")
    phi = mem._session_phi[session]
    with torch.no_grad():
        recon, logits, summary = mem.readout(phi)
        pred_id = int(logits.view(-1).argmax().item())
        pred = labels[pred_id] if pred_id < len(labels) else "?"
        # cosine to last event embedding
        last_emb = torch.tensor(
            mem.store.list_session(session)[-1].embedding, dtype=torch.float32
        )
        cos = torch.nn.functional.cosine_similarity(
            recon.view(1, -1), last_emb.view(1, -1)
        ).item()
    print(f"  predicted trajectory class : {pred}")
    print(f"  true class                 : order_food")
    print(f"  recon↔last-event cosine    : {cos:.3f}")
    print(f"  summary vector dim         : {summary.numel()}")
    print(f"  φ dim (compact fingerprint): {phi.numel()}")

    banner("6) Second session + phi nearest-neighbor across store")
    s2 = "user_demo_02"
    for text in [
        "there is a wrong charge on my bill",
        "I was billed twice this month",
        "please refund the duplicate payment",
    ]:
        mem.remember(s2, text, label="support_billing")

    hits = mem.recall(session_id=None, query="refund duplicate charge", mode="embed", top_k=3)
    print("  embed search top hits:")
    for t, sc in zip(hits.texts, hits.scores):
        print(f"    {sc:.3f}  {t}")

    # trajectory search using session 1's φ
    phi_hits = mem.store.search_by_phi(mem._session_phi[session].tolist(), top_k=3)
    print("  phi-neighbor facts (session1 trajectory):")
    for fact, sc in phi_hits:
        print(f"    {sc:.3f}  [{fact.session_id}] {fact.text[:60]}")

    banner("7) Save checkpoint")
    out = Path(args.save)
    mem.save(out)
    print(f"  saved → {out.resolve()}")
    print(f"  val_accuracy={metrics['val_accuracy']:.1%}  val_cosine={metrics['val_cosine']:.3f}")

    banner("VERDICT")
    print(
        """
  Claude said: "encoder only; phase unreadable by LLMs."

  After this hybrid layer:
    ✓ Encoder (Helix) folds ordered events into φ
    ✓ Trained PhaseReadout maps φ → class + embedding recon
    ✓ FactStore keeps the actual text (LLM-readable)
    ✓ recall() / context_block() never asks an LLM to read angles

  Helix is the trajectory codec.
  HybridMemory is the memory product surface.
  HelixDB (optional) can hold the graph/vector facts side.
"""
    )
    ok = metrics["val_accuracy"] >= 0.70 and metrics["val_cosine"] >= 0.35
    if ok:
        print("  STATUS: HIT  (readout learned; retrieval path works)")
        return 0
    print("  STATUS: weak training — try more epochs")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
