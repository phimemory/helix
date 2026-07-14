# Helix - Project Context

Everything we have figured out about what this is, what it is not, and where it goes.

---

## What Helix actually is

Phase-rotation RNN cell. Each neuron maintains a phase angle that accumulates instead of decaying. Soft-quantized to a pi/4 grid. Multi-harmonic readout (cos/sin at harmonics 1,2,4,8). Multi-clock bands added in v1.1 (four persistence speeds: 0.50, 0.80, 0.95, 0.999).

It sits in the same family as uRNN (Arjovsky 2016) and RUM (Dangovski 2019). Real engineering, novel-ish design. Not a singular invention but a legitimate piece of work.

---

## What the benchmarks actually prove

**Parity result is real.** 1 Helix neuron beats 128 GRU neurons on 16-bit parity. This works because Helix exploits float32 mantissa as stable storage - the phase angle accumulates without decay. GRU's contractive update destroys early bits before the network sees later ones. This is a structural property of the update rule, not a capacity problem.

**Honest limit:** it would fail at 24-bit parity (float32 only has ~7 decimal digits of mantissa). The claim is narrow: lossless discrete recall up to the precision of float32.

**Crystalline loop (100% bit accuracy), majority vote (123x fewer params)** - both real.

**Sine wave tracking** - GRU wins. Helix is not the right tool for continuous approximation. The README says this honestly.

---

## What the README overclaims

The "Unitary Isometry" framing does NOT match the code. The actual cell uses tanh, sigmoid, learned weights, soft pull - none of these are unitary. Anyone with ML background will catch this. This needs to be fixed or removed before pitching to researchers.

---

## Repos

- `C:\Users\Pawan\Desktop\helix\` - main repo, pushed to github.com/phimemory/helix (also redirects from Cintu07/helix)
- `C:\Users\Pawan\Desktop\ROUND-research-notes\` - same architecture renamed (HelixCell -> UITNeuronCell). Just benchmarks. Not a separate project, just an alias.
- Three names exist: Helix, ROUND, UIT. This fragments the work. Consolidate under Helix.

---

## Crystal suite (crystal/)

Built on top of the core cell. Includes: MemoryCrystal, TemporalPhaseIndex, AffectiveEncoder, ResonanceDetector, MultiModalFusion, PhaseDecoder, PhiCrypt, PhaseCollapseRegister, SpectrumCache, ContextDistiller, PhaseDiff, HelixMemory.

**Honest status:** mostly theoretical and untested. HelixMemory is a legitimate orchestrator but none of these modules have been validated in production. Do not ship as production-ready.

---

## What Helix is NOT

- Not a Mem0 / Supermemory replacement. Different problem class entirely. Mem0 = fact extraction + RAG over stored text. Helix = sequence compression into phase state. These are complementary, not competing.
- Not a context window killer. That framing was hype from a previous conversation and it crashed when reality-checked. Never use it again.
- Not a singular invention. It is a legitimate research contribution in the rotation-based RNN space, not a breakthrough that invalidates transformers.

---

## Agreed direction (as of 2026-05-07)

**Do not add more features to Helix core.** The architecture is done. Adding features without a concrete use case is wasted effort.

**Use Helix as a component, not a product.** Two concrete uses already exist:

1. **quifer** - already built and working. Helix phase cell encodes wallet transaction sequences into 64-dim fingerprints for Sybil detection. This is the first real-world validation of Helix on non-synthetic data.

2. **Hyli** - voice AI telephony platform. Planned: use HelixMemory for cross-session behavioral memory (how this user's call patterns change over time, emotional state tracking across sessions). Not built yet.

**The positioning that is honest and defensible:** Helix is a sequence compression primitive. Where you need to remember the ORDER of things without losing information, and where the data is discrete or near-discrete, Helix outperforms GRU with far fewer parameters. That is the real claim. Build products on top of this claim, not around abstract memory theory.

---

## Hybrid layer (as of 2026-07-13) — closes the encoder-only audit

Claude (and any serious ML audit) correctly said: core Helix is an **encoder**; raw phase is not LLM-readable text.

**Response shipped in `hybrid/`:**

- `TrajectoryEncoder` — Helix φ fingerprint (unchanged science)
- `PhaseReadout` — **trained** decoder: φ → class logits + embedding recon + summary
- `FactStore` / optional `HelixDBStore` — actual text lives here (SQLite always; [HelixDB](https://helix-db.com) if `HELIXDB_URL` set)
- `HybridMemory.context_block()` — LLM-facing string of **retrieved text**, never raw angles

Run: `python -m hybrid.demo --epochs 50`

This does **not** make Helix an infinite context window. It makes Helix a legitimate **trajectory codec inside a real memory stack**.

---

## What needs to happen before pitching Helix to anyone technical

1. Fix or remove the "Unitary Isometry" framing in README - it is wrong
2. Consolidate ROUND/UIT/Helix naming under one name
3. Add a real-world benchmark beyond synthetic tasks (quifer provides one now)
4. Test the crystal suite modules against actual use cases before claiming they work

---

## Why this project matters personally

Previous company name was Lexideck (named after Pawan's late brother). That name was removed. The project carries personal weight. Do not push on this.

---

## quifer connection

quifer is the first external validation that Helix works on real data. The phase encoding correctly clusters wallet behavioral sequences - it found not just Sybil farm wallets but the operator's funding wallet behind them. This is worth mentioning when explaining what Helix can do.
