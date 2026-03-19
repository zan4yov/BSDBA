# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 10 — Phase 5: Integration & End-to-End Testing
# SRS: All acceptance criteria | Phase: 5 | Mode: Composer
# V.E.R.I.F.Y. Level: 3 (ALL modules) | Q7 MUST close here
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file src/tests/test_audio.py
@file src/tests/test_cv.py
@file src/tests/test_gradcam.py
@file src/tests/test_nlp.py
@file src/tests/test_e2e.py

[C] CHALLENGE: Run the complete test suite and validate all SRS acceptance criteria.
    Resolve Q7 (EER scoring protocol). Produce a benchmark report.

TASK 1: Resolve Q7 — EER Scoring Protocol
  Decision: Use scikit-learn for EER calculation (academic prototype context).
  If submitting to ASVspoof challenge → use official scoring script.
  Document in docs/adr/phase5-eer-protocol.md.
  Update session-cheatsheet.md: Q7 → ✅ RESOLVED.

TASK 2: Full Test Suite Run
  Run: pytest src/tests/ -v --timeout=30 --tb=short
  All tests must pass. Record results in docs/adr/phase5-test-results.md.

TASK 3: Accuracy Validation on FoR Test Set
  Using the trained checkpoint + FoR for-2sec held-out test split:
  - Compute EER → must be ≤ 10% [FR-CV-008]
  - Compute AUC-ROC → must be ≥ 0.90 [FR-CV-008]
  - Compute F1-macro → must be ≥ 0.88 [NFR-Accuracy]
  - Compute overall accuracy → must be ≥ 90% [NFR-Accuracy]
  Record all results in docs/adr/phase5-accuracy-results.md.
  If any target NOT met → flag as [GATE: BLOCKED] and return to Sprint B for retraining.

TASK 4: Latency Benchmark on CPU
  On a CPU-only machine (or HF Spaces CPU emulation):
  - Audio DSP (single 2.0 s clip): record actual ms → target ≤ 500 ms
  - ONNX inference: record actual ms → target ≤ 1,500 ms
  - Grad-CAM: record actual ms → target ≤ 3,000 ms
  - NLP rule-based fallback: record actual ms (no API)
  - End-to-end: record wall clock ms → target ≤ 15,000 ms
  Record all in docs/adr/phase5-latency-benchmark.md with PASS/FAIL per target.
  If any latency FAILS → flag and resolve before Phase 6.

TASK 5: Resilience Test
  Verify Qwen 2.5 API failure → rule-based fallback displayed with warning badge [FR-NLP-003]
  Test: mock API to return HTTP 503 → app.py shows fallback text + warning badge.

TASK 6: Security Audit
  Scan all Python files for: API keys, tokens, stack traces in UI error messages.
  Verify no .pth or .onnx files in Git history.
  Verify requirements.txt has all pinned versions.
  Record in docs/adr/phase5-security-audit.md.