"""
Helix Crystal Suite v2.0 — FULL Integration Test (All 11 Modules)
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from crystal.substrate import MemoryCrystal
from crystal.phase_collapse import PhaseCollapseRegister
from crystal.spectrum_cache import SpectrumCache
from crystal.distillation import ContextDistiller
from crystal.resonance import ResonanceDetector
from crystal.affective import AffectiveEncoder
from crystal.federation import PhaseFederation
from crystal.temporal_index import TemporalPhaseIndex
from crystal.phicrypt import PhiCrypt
from crystal.multimodal import MultiModalFusion
from crystal.phase_diff import PhaseDiff, PhaseVersionTracker

PASS_COUNT = 0
FAIL_COUNT = 0

def check(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        print(f"    FAIL: {msg}")

def test_csl():
    print("=" * 60)
    print("TEST 1: Crystalline Substrate Layer (CSL)")
    print("=" * 60)
    
    crystal = MemoryCrystal(input_size=32, hidden_size=16, harmonics=[1, 2, 4, 8])
    for i in range(100):
        crystal.absorb(torch.randn(32))
    
    hx_path = os.path.join(os.path.dirname(__file__), "test_memory.hx")
    size = crystal.export(hx_path)
    print(f"  Exported: {size} bytes ({crystal.absorb_count} absorbed)")
    
    crystal2 = MemoryCrystal(input_size=32, hidden_size=16, harmonics=[1, 2, 4, 8])
    crystal2.load(hx_path)
    
    diff = (crystal.phi_state - crystal2.phi_state).abs().max().item()
    check(diff < 1e-6, f"Phase mismatch: {diff}")
    
    f1, f2 = crystal.recall(), crystal2.recall()
    feat_diff = (f1 - f2).abs().max().item()
    check(feat_diff < 1e-5, f"Feature mismatch: {feat_diff}")
    
    os.remove(hx_path)
    print("  PASSED\n")

def test_pce():
    print("=" * 60)
    print("TEST 2: Phase Collapse Events (PCE)")
    print("=" * 60)
    
    pcr = PhaseCollapseRegister(num_flags=8)
    pcr.register_flag(0, "salmon_out")
    
    check(pcr.collapse_named("salmon_out") == True, "First collapse should return True")
    check(pcr.query_named("salmon_out") == True, "Should be collapsed")
    check(pcr.collapse_named("salmon_out") == False, "Re-collapse should return False")
    
    try:
        pcr.attempt_overwrite(0)
        check(False, "Should have raised error")
    except AssertionError:
        check(True, "")
    
    sv = pcr.get_state_vector()
    check(sv[0].item() == -1.0, f"Collapsed neuron should be -1, got {sv[0]}")
    check(sv[1].item() == 1.0, f"Open neuron should be 1, got {sv[1]}")
    print("  PASSED\n")

def test_hsc():
    print("=" * 60)
    print("TEST 3: Harmonic Spectrum Caching (HSC)")
    print("=" * 60)
    
    cache = SpectrumCache(hidden_size=64, harmonics=[1, 2, 4, 8])
    phi = torch.randn(64)
    cache.initialize(phi)
    
    delta = torch.zeros(64)
    delta[10:15] = torch.randn(5) * 0.1
    cache.update(delta)
    features = cache.get_features()
    
    phi_updated = phi + delta
    naive = []
    for h in [1, 2, 4, 8]:
        naive.append(torch.cos(h * phi_updated))
        naive.append(torch.sin(h * phi_updated))
    naive = torch.cat(naive)
    
    diff = (features - naive).abs().max().item()
    check(diff < 1e-5, f"Cache mismatch: {diff}")
    print(f"  Cache vs naive diff: {diff}")
    print("  PASSED\n")

def test_cdpc():
    print("=" * 60)
    print("TEST 4: Context Distillation (CDPC)")
    print("=" * 60)
    
    distiller = ContextDistiller(input_size=768, hidden_size=64)
    for i in range(1000):
        distiller.feed(torch.randn(768))
    
    summary = distiller.summary()
    ratio = distiller.compression_ratio()
    
    check(summary.shape[0] == 512, f"Summary wrong shape: {summary.shape}")
    check(ratio > 1000, f"Compression too low: {ratio}")
    print(f"  Compression: {distiller.compression_ratio_str()}")
    print("  PASSED\n")

def test_apr():
    print("=" * 60)
    print("TEST 5: Anticipatory Phase Resonance (APR)")
    print("=" * 60)
    
    detector = ResonanceDetector(hidden_size=64, output_size=32)
    for i in range(20):
        phi = torch.randn(64) * 0.5 + torch.sin(torch.arange(64).float() * 0.1 * i)
        detector.record_state(phi)
    
    velocity, stability = detector.compute_phase_velocity()
    check(velocity is not None, "Velocity should be computed")
    check(stability > 0, f"Stability should be positive: {stability}")
    
    score = detector.detect_resonance(torch.randn(64))
    check(0 <= score <= 1, f"Score out of range: {score}")
    
    predicted = detector.predict_next(torch.randn(64))
    check(predicted.shape[0] == 32, f"Wrong prediction shape: {predicted.shape}")
    print(f"  Stability: {stability:.3f}, Score: {score:.3f}")
    print("  PASSED\n")

def test_ape():
    print("=" * 60)
    print("TEST 6: Affective Phase Encoding (APE)")
    print("=" * 60)
    
    encoder = AffectiveEncoder(hidden_size=64, affective_neurons=8)
    encoder.encode_sentiment(0.8, 0.3)
    encoder.encode_sentiment(-0.7, 0.9)
    
    state = encoder.decode_sentiment()
    check('valence' in state, "Missing valence")
    check('arousal' in state, "Missing arousal")
    check('label' in state, "Missing label")
    
    features = encoder.get_affective_features()
    check(features.shape[0] == 32, f"Wrong feature shape: {features.shape}")
    
    trajectory = encoder.emotional_trajectory()
    check(len(trajectory) == 2, f"Wrong trajectory length: {len(trajectory)}")
    print(f"  Final state: {state}")
    print("  PASSED\n")

def test_fpa():
    print("=" * 60)
    print("TEST 7: Federated Phase Alignment (FPA)")
    print("=" * 60)
    
    fed = PhaseFederation()
    phi_a = torch.randn(64) * 0.5
    phi_b = phi_a + torch.randn(64) * 0.1
    phi_c = torch.randn(64) * 0.5
    
    merged = fed.merge([phi_a, phi_b, phi_c])
    check(merged.shape[0] == 64, "Wrong merge shape")
    
    div_ab, _ = fed.divergence(phi_a, phi_b)
    div_ac, _ = fed.divergence(phi_a, phi_c)
    check(div_ab < div_ac, f"A-B should be closer than A-C: {div_ab} vs {div_ac}")
    
    align_ab = fed.alignment_score(phi_a, phi_b)
    check(align_ab > 0.9, f"A-B alignment too low: {align_ab}")
    
    consensus, _ = fed.consensus([phi_a, phi_b, phi_c])
    check(0 <= consensus <= 1, f"Consensus out of range: {consensus}")
    
    selective, agreement = fed.selective_merge([phi_a, phi_b, phi_c])
    check(selective.shape[0] == 64, "Wrong selective merge shape")
    print(f"  Alignment A-B: {align_ab:.3f}, Div A-B: {div_ab:.3f}, Div A-C: {div_ac:.3f}")
    print("  PASSED\n")

def test_tpi():
    print("=" * 60)
    print("TEST 8: Temporal Phase Indexing (TPI)")
    print("=" * 60)
    
    tpi = TemporalPhaseIndex(hidden_size=64, snapshot_interval=10)
    
    # Record 100 steps
    for t in range(100):
        phi = torch.randn(64) * 0.1 * t
        tpi.record(t, phi)
    
    check(tpi.num_snapshots() == 10, f"Expected 10 snapshots, got {tpi.num_snapshots()}")
    
    # Exact recall at recorded step
    phi_0 = tpi.recall_at(0)
    check(phi_0.shape[0] == 64, "Wrong recall shape")
    
    # Interpolated recall at non-recorded step
    phi_5 = tpi.recall_at(5)
    check(phi_5.shape[0] == 64, "Interpolation failed")
    
    # Feature recall
    feat = tpi.recall_features_at(0, harmonics=[1, 2, 4, 8])
    check(feat.shape[0] == 512, f"Wrong feature shape: {feat.shape}")
    
    # Range query
    segment = tpi.recall_range(0, 30)
    check(len(segment) >= 3, f"Range query should return >= 3 snapshots: {len(segment)}")
    
    # Search
    query = tpi.recall_at(0)
    matches = tpi.search(query, top_k=3)
    check(len(matches) == 3, f"Search should return 3 results: {len(matches)}")
    check(matches[0][0] == 0, f"Best match should be step 0: {matches[0]}")
    
    # Phase velocity
    vel = tpi.phase_velocity_at(5)
    check(vel is not None, "Velocity should be computed")
    
    print(f"  Snapshots: {tpi.num_snapshots()}, Memory: {tpi.memory_bytes()} bytes")
    print(f"  Search top match: step {matches[0][0]} (similarity {matches[0][1]:.3f})")
    print("  PASSED\n")

def test_phicrypt():
    print("=" * 60)
    print("TEST 9: Phase Encryption (PhiCrypt)")
    print("=" * 60)
    
    crypt = PhiCrypt()
    
    # Basic encrypt/decrypt
    phi_original = torch.randn(64) * 2
    passphrase = "my_secret_key_2024"
    
    encrypted, salt = crypt.encrypt(phi_original, passphrase)
    decrypted = crypt.decrypt(encrypted, passphrase, salt)
    
    diff = (phi_original - decrypted).abs().max().item()
    check(diff < 1e-5, f"Decrypt mismatch: {diff}")
    print(f"  Encrypt/decrypt diff: {diff}")
    
    # Verify encryption scrambles the data
    verify = crypt.verify_encryption(phi_original, encrypted)
    check(verify['is_secure'], f"Encryption not secure: correlation={verify['correlation']:.3f}")
    print(f"  Encryption correlation: {verify['correlation']:.3f} (should be ~0)")
    
    # Wrong passphrase should produce wrong result
    wrong_decrypt = crypt.decrypt(encrypted, "wrong_password", salt)
    wrong_diff = (phi_original - wrong_decrypt).abs().max().item()
    check(wrong_diff > 0.1, f"Wrong key should produce different result: {wrong_diff}")
    print(f"  Wrong key diff: {wrong_diff:.3f} (should be large)")
    
    # File-level encryption
    crystal = MemoryCrystal(input_size=32, hidden_size=16, harmonics=[1, 2, 4, 8])
    for i in range(50):
        crystal.absorb(torch.randn(32))
    
    hx_path = os.path.join(os.path.dirname(__file__), "test_plain.hx")
    hxe_path = os.path.join(os.path.dirname(__file__), "test_encrypted.hxe")
    hx_dec_path = os.path.join(os.path.dirname(__file__), "test_decrypted.hx")
    
    crystal.export(hx_path)
    enc_size = crypt.encrypt_file(hx_path, hxe_path, "file_password")
    dec_size = crypt.decrypt_file(hxe_path, hx_dec_path, "file_password")
    
    # Load decrypted and compare
    crystal2 = MemoryCrystal(input_size=32, hidden_size=16, harmonics=[1, 2, 4, 8])
    crystal2.load(hx_dec_path)
    
    file_diff = (crystal.phi_state - crystal2.phi_state).abs().max().item()
    check(file_diff < 1e-5, f"File encrypt/decrypt mismatch: {file_diff}")
    print(f"  File encrypt/decrypt diff: {file_diff}")
    
    # Cleanup
    for p in [hx_path, hxe_path, hx_dec_path]:
        if os.path.exists(p):
            os.remove(p)
    
    print("  PASSED\n")

def test_multimodal():
    print("=" * 60)
    print("TEST 10: Multi-Modal Phase Fusion (MMPF)")
    print("=" * 60)
    
    fusion = MultiModalFusion(hidden_size=64, unified_dim=128)
    
    # Absorb text
    fusion.absorb_text(torch.randn(768))
    fusion.absorb_text(torch.randn(768))
    check(fusion.modality_counts['text'] == 2, "Text count wrong")
    
    # Absorb image
    fusion.absorb_image(torch.randn(512))
    check(fusion.modality_counts['image'] == 1, "Image count wrong")
    
    # Absorb audio
    fusion.absorb_audio(torch.randn(384))
    check(fusion.modality_counts['audio'] == 1, "Audio count wrong")
    
    # Absorb generic
    fusion.absorb_generic(torch.randn(256))
    check(fusion.modality_counts['generic'] == 1, "Generic count wrong")
    
    # Recall
    features = fusion.recall()
    check(features.shape[0] == 512, f"Wrong feature shape: {features.shape}")
    
    # Export/load
    hx_path = os.path.join(os.path.dirname(__file__), "test_multimodal.hx")
    fusion.export(hx_path)
    
    fusion2 = MultiModalFusion(hidden_size=64, unified_dim=128)
    fusion2.load(hx_path)
    
    f1 = fusion.recall()
    f2 = fusion2.recall()
    diff = (f1 - f2).abs().max().item()
    check(diff < 1e-5, f"Multimodal export/load mismatch: {diff}")
    
    os.remove(hx_path)
    
    stats = fusion.stats()
    print(f"  Stats: {stats}")
    print(f"  {fusion}")
    print("  PASSED\n")

def test_phase_diff():
    print("=" * 60)
    print("TEST 11: Phase Diff Protocol (PDP)")
    print("=" * 60)
    
    differ = PhaseDiff()
    
    # Create two states
    phi_old = torch.randn(64) * 0.5
    phi_new = phi_old.clone()
    phi_new[0:5] += 1.0    # Major change on first 5 neurons
    phi_new[10:15] += 0.05  # Minor change on next 5
    # Rest unchanged
    
    # Compute diff
    changeset = differ.diff(phi_old, phi_new)
    
    check(changeset.num_major_changes() >= 4, 
          f"Expected >= 4 major changes: {changeset.num_major_changes()}")
    check(changeset.num_unchanged() >= 40, 
          f"Expected >= 40 unchanged: {changeset.num_unchanged()}")
    
    print(changeset.summary())
    
    # Apply diff
    patched = differ.apply(phi_old, changeset)
    patch_diff = (patched - phi_new).abs().max().item()
    check(patch_diff < 1e-6, f"Apply diff mismatch: {patch_diff}")
    print(f"  Apply diff error: {patch_diff}")
    
    # Invert diff
    inverse = differ.invert(changeset)
    reverted = differ.apply(phi_new, inverse)
    revert_diff = (reverted - phi_old).abs().max().item()
    check(revert_diff < 1e-6, f"Revert diff mismatch: {revert_diff}")
    print(f"  Revert diff error: {revert_diff}")
    
    # Version tracker
    tracker = differ.create_tracker()
    tracker.commit(phi_old, "initial state")
    tracker.commit(phi_new, "after changes")
    tracker.commit(phi_new + torch.randn(64) * 0.01, "minor tweaks")
    
    check(tracker.num_versions() == 3, f"Expected 3 versions: {tracker.num_versions()}")
    
    # Rollback
    restored = tracker.rollback(0)
    rollback_diff = (restored - phi_old).abs().max().item()
    check(rollback_diff < 1e-6, f"Rollback mismatch: {rollback_diff}")
    
    # Diff versions
    v_diff = tracker.diff_versions(0, 1)
    check(v_diff.num_major_changes() >= 4, "Version diff should show major changes")
    
    print(tracker.log())
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HELIX CRYSTAL SUITE v2.0 — ALL 11 MODULES TEST")
    print("=" * 60 + "\n")
    
    test_csl()
    test_pce()
    test_hsc()
    test_cdpc()
    test_apr()
    test_ape()
    test_fpa()
    test_tpi()
    test_phicrypt()
    test_multimodal()
    test_phase_diff()
    
    print("=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"  RESULTS: {PASS_COUNT}/{total} checks passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("  ALL 11 MODULES PASSED")
    else:
        print(f"  WARNING: {FAIL_COUNT} FAILURES")
    print("=" * 60)
