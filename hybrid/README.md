# Helix Hybrid Memory

Closes the **encoder-only** gap. Makes Helix a real memory *component*.

## The honest architecture

```
events (text)
   │
   ├─► HashEmbedder → vectors
   │         │
   │         ▼
   │   TrajectoryEncoder (Helix phase cell) ──► φ fingerprint
   │         │
   │         ▼ trained
   │   PhaseReadout ──► class / recon embed / summary
   │
   └─► FactStore (SQLite or HelixDB) ──► actual text payload
                    │
                    ▼
              context_block() → paste into any LLM
```

| Layer | Role | Readable by LLM? |
|-------|------|------------------|
| Helix φ | Order-sensitive trajectory fingerprint | No (and shouldn't be) |
| PhaseReadout | Trained map φ → signals | Indirectly (labels/summary) |
| FactStore | Exact text / facts | **Yes** |

## Quick start

```bash
cd helix
pip install torch numpy

# train readout + live demo
python -m hybrid.demo --epochs 50 --save hybrid_ckpt
```

### In code

```python
from hybrid import HybridMemory, train_readout

readout, enc, emb, labels, metrics = train_readout(epochs=50)
mem = HybridMemory(label_names=labels)
mem.encoder.crystal.cell.load_state_dict(enc.crystal.cell.state_dict())
mem.attach_trained_readout(readout, labels)

mem.remember("u1", "I want to order a large pizza", label="order_food")
mem.remember("u1", "deliver to 42 market street", label="order_food")

print(mem.context_block(session_id="u1", query="pizza"))
# → real text for the LLM, not raw phase angles
```

## Optional HelixDB backend

[HelixDB](https://helix-db.com) is a **graph-vector database for AI memory** (different product from this neural Helix). Use it as the fact/graph store:

```bash
# start your HelixDB instance, then:
set HELIXDB_URL=http://localhost:6969
python -m hybrid.demo --backend helixdb --db ./facts_cache.sqlite
```

If HelixDB is offline, the adapter still works via the local SQLite mirror.

## What this proves (vs the audit)

| Claim | Status |
|-------|--------|
| Helix alone is a full LLM context layer | False |
| Helix is a cracked sequence fingerprint | True |
| φ unreadable without training | True |
| φ usable after PhaseReadout training | True |
| Hybrid retrieves real text for agents | True |

## Files

| file | purpose |
|------|---------|
| `embedder.py` | Deterministic hash embedder (no heavy deps) |
| `encoder.py` | Helix `MemoryCrystal` trajectory codec |
| `readout.py` | Trained decoder / classifier / summary |
| `store.py` | SQLite + optional HelixDB fact store |
| `memory.py` | `HybridMemory` product API |
| `train.py` | Synthetic sessions + training loop |
| `demo.py` | End-to-end proof script |
