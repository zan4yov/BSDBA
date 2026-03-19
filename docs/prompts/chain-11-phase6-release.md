# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 11 — Phase 6: Build, Release & HuggingFace Spaces Deployment
# SRS: FR-DEP-001–010 final | Phase: 6 | Mode: Agent
# Pre-condition: Phase 5 Gate fully passed (all accuracy + latency targets met)
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase5-accuracy-results.md
@file docs/adr/phase5-latency-benchmark.md
@file app.py
@file requirements.txt

Use huggingface-mcp to:
  (a) Push app.py and all src/ to the HF Space repository
  (b) Set QWEN_API_KEY and HF_TOKEN as Space secrets
  (c) Verify Space builds without errors
  (d) Confirm ONNX model loads correctly at Space startup

TASK 1: Pre-deployment Checklist
  [ ] requirements.txt: all versions pinned, no ranges
  [ ] .gitignore: .env, *.pth, *.onnx excluded from repo
  [ ] ONNX model: numerical equivalence verified (test_cv.py test 5 passing)
  [ ] ONNX session: created once at app.py startup, not per-request
  [ ] API keys: confirmed in HF Spaces secrets panel, not in any file
  [ ] Demo samples: data/samples/bonafide/ (2 files) + data/samples/spoof/ (2 files) exist

TASK 2: HF Spaces Deployment via huggingface-mcp
  Deploy app.py to HF Spaces. Verify:
  - Space builds without import errors
  - ONNX session loads in < 30 s (cold start check) [NFR-Reliability]
  - Upload a test bonafide WAV → verify verdict panel appears before NLP panel
  - Upload a test spoof WAV → verify Grad-CAM overlay renders correctly

TASK 3: Production ONNX Latency Verification
  On the live HF Space CPU (2 vCPU):
  - Time ONNX inference for one 2-second clip
  - Must be ≤ 1,500 ms [FR-DEP-010, NFR-Scalability]
  - If > 1,500 ms → apply ONNX graph optimisations (ort.SessionOptions) and retest

TASK 4: Final Code Freeze
  - Tag Git commit: v2.1.0-release
  - Update README.md with live HF Spaces URL
  - Freeze config.yaml with a comment: "# FROZEN — Phase 6 Release v2.1.0"
  - Document release in docs/adr/phase6-release-notes.md