# DSDBA — Master Implementation Plan (MIP)
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 0 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** BASELINE — update checkbox state at end of each phase

---

## How to Use This Document

- Check off each item when the associated Gate Check passes (see `docs/prompts/gate-check-universal.md`).
- A phase is COMPLETE only when ALL its items are checked AND the Gate Check returns 🟢 PASS.
- Blocking Open Questions are noted per item — never start a downstream item while its blocking Q is 🔴 OPEN.
- SRS FR IDs are the definitive acceptance criteria for each item.

---

## Phase 0 — Project Inception & Architecture Design

**Chain:** 01 | **Mode:** Agent | **Gate:** Architecture Review

- [x] Identify top 3 architectural risks with SRS coupling points → `docs/adr/phase0-risk-register.md` (RISK-001/002/003)
- [x] Produce Mermaid pipeline architecture diagram → `docs/adr/phase0-pipeline-diagram.md`
- [x] Create ADR-0001 for MCP tool selection → `docs/adr/phase0-mcp-selection.md`
- [x] Initial assessment of Q3 (VRAM) and Q4 (Grad-CAM layer) → session-cheatsheet.md updated
- [x] Draft Master Implementation Plan (this document) → `docs/adr/phase0-mip.md`

**SRS FRs Addressed:** FR-AUD-001–011 (scoped), FR-CV-001–016 (scoped), FR-NLP-001–009 (scoped), FR-DEP-001–010 (scoped)  
**Blocking Q:** None  
**Status:** ✅ COMPLETE

---

## Phase 1 — Project Backlog & Environment Setup

**Chain:** 02 | **Mode:** Agent | **Gate:** Environment Verification

- [ ] Create and populate `docs/backlog/` with sprint-level user stories
- [ ] Verify `requirements.txt` — all packages pinned, no version ranges (NFR-Security)
- [ ] Confirm `config.yaml` is the single source of truth — no magic numbers in any Python file (NFR-Maintainability)
- [ ] Create `src/` directory scaffold with empty `__init__.py` files and module docstring stubs
- [ ] Create `tests/` directory scaffold with `conftest.py` and placeholder test files
- [ ] Set up structured logger at `src/utils/logger.py` (JSON output, no `print()`)
- [ ] Verify MCP servers in `mcp.json`: context7, huggingface-skills, stitch — all reachable
- [ ] Load `config.yaml` utility at `src/utils/config.py` — typed, cached, pathlib-based

**SRS FRs Addressed:** NFR-Maintainability, NFR-Security, NFR-Reliability  
**Blocking Q:** None  
**Status:** 🔴 NOT STARTED

---

## Phase 2 — Sprint A: Audio DSP Module

**Chain:** 05 | **Mode:** Agent | **Gate:** Audio DSP Gate Check

- [ ] Implement `src/audio/dsp.py` — format validation (FR-AUD-001)
- [ ] Implement resampling to 16 kHz via librosa (FR-AUD-002)
- [ ] Implement amplitude normalisation (FR-AUD-003)
- [ ] Implement pad/trim to exactly 2.0 s / 32,000 samples (FR-AUD-004)
- [ ] Implement minimum duration rejection — raise `ValueError: AUD-001` if < 0.5 s (FR-AUD-005)
- [ ] Implement Mel spectrogram: 128 bins, n_fft=2048, hop=512, Hann window (FR-AUD-006)
- [ ] Implement log-power dB scaling (FR-AUD-007)
- [ ] Implement resize to 224×224 + channel replication → `[3, 224, 224]` float32 (FR-AUD-008)
- [ ] Implement optional format support: MP3, OGG via ffmpeg/pydub (FR-AUD-011)
- [ ] Audio in-memory only — zero disk writes (NFR-Security)
- [ ] Write `tests/test_audio.py` — shape/dtype assertion + latency assertion ≤ 500 ms
- [ ] Gate Check: all tests pass, latency target met

**SRS FRs Addressed:** FR-AUD-001–011  
**Blocking Q:** Q5 (Mel bin→Hz mapping) — note for Sprint C; not blocking Sprint A  
**Status:** 🔴 NOT STARTED

---

## Phase 3 — Sprint B: CV Model Definition & Training

**Chain:** 06–07 | **Mode:** Agent | **Gate:** CV Training Gate Check

- [ ] Implement `src/cv/model.py` — EfficientNet-B4 definition + 2-class linear head (FR-CV-001, FR-CV-002)
- [ ] Implement `src/cv/train.py` — two-phase training (frozen backbone → full fine-tune) (FR-CV-003)
- [ ] Verify VRAM feasibility: batch_size=16 on Colab T4 (resolve Q3 before starting)
- [ ] Implement SpecAugment augmentation: time_mask ≤ 10%, freq_mask ≤ 10%, time_shift ±0.1 s, SNR ≥ 20 dB (FR-CV-006)
- [ ] Implement BCE with class weights loss function (FR-CV-005)
- [ ] Implement early stopping (patience=5) and checkpoint-per-epoch saving (FR-CV-008, NFR-Reliability)
- [ ] Achieve AUC-ROC ≥ 0.90 and EER ≤ 10% on FoR test set (FR-CV-008, NFR-Accuracy)
- [ ] Push checkpoint to HF Hub — repo ID from `config.yaml:training.hf_model_repo` (FR-CV-007)
- [ ] Write `tests/test_train.py` — forward pass smoke test, checkpoint save/load
- [ ] Gate Check: accuracy targets met, checkpoint verified on HF Hub

**SRS FRs Addressed:** FR-CV-001–009  
**Blocking Q:** Q3 (VRAM on Colab T4) — 🔴 MUST resolve before Sprint B begins  
**Status:** 🔴 NOT STARTED

---

## Phase 3 — Sprint C: ONNX Export & CV Inference

**Chain:** 08 | **Mode:** Composer | **Gate:** ONNX Equivalence Gate Check

- [ ] Implement `src/cv/infer.py` — ONNX Runtime inference wrapper, shape+dtype guard (FR-DEP-010, FR-AUD-008)
- [ ] ONNX session created ONCE at app startup — not per-request (`.cursorrules` rule)
- [ ] Export `.pth` → `.onnx` with dynamic batch axis (FR-DEP-010)
- [ ] Numerical equivalence test: |ONNX output − PyTorch output| < 1e-5 (FR-DEP-010)
- [ ] Verify ONNX inference latency ≤ 1,500 ms on CPU (FR-DEP-010, NFR-Performance)
- [ ] Implement `src/cv/gradcam.py` — GradCAM via pytorch-grad-cam (FR-CV-010, FR-CV-011)
- [ ] Confirm Grad-CAM target layer path (resolve Q4 via `model.named_modules()` introspection)
- [ ] Implement heatmap overlay: jet colormap, α=0.5, 224×224 PNG output (FR-CV-012)
- [ ] Implement Mel bin→Hz band attribution: 4 bands, softmax normalisation (FR-CV-013, FR-CV-014)
- [ ] Validate Q5: Mel bin→Hz mapping for n_mels=128 at 16 kHz (FR-CV-013)
- [ ] Grad-CAM latency ≤ 3,000 ms on CPU (FR-CV-015)
- [ ] Expose raw saliency JSON endpoint for dev access (FR-CV-016)
- [ ] Write `tests/test_cv.py` — malformed tensor guard test, ONNX equivalence test, latency assertions

**SRS FRs Addressed:** FR-CV-010–016, FR-DEP-010  
**Blocking Q:** Q4 (Grad-CAM target layer) — 🔴 MUST resolve before Sprint C begins; Q5 (Mel bin→Hz) — 🔴 MUST resolve before Sprint C begins  
**Status:** 🔴 NOT STARTED

---

## Phase 4 — Sprint D: NLP Explanation Module

**Chain:** 09 | **Mode:** Agent | **Gate:** NLP Explanation Gate Check

- [ ] Implement `src/nlp/explain.py` — async Qwen 2.5 call via stitch-mcp (FR-NLP-001, FR-NLP-002)
- [ ] Implement three-tier fallback: Qwen 2.5 → Gemma 3 → rule-based template (FR-NLP-003, FR-NLP-007)
- [ ] Prompt generates 3–5 sentences in English (FR-NLP-001, FR-NLP-004)
- [ ] API key read from `QWEN_API_KEY` env var — never in code (FR-NLP-005, S.H.I.E.L.D.)
- [ ] NLP result does NOT block Stage 2/3 UI display (FR-NLP-006)
- [ ] Implement LLM response caching keyed on `(label, confidence_bucket, top_band)` (FR-NLP-008)
- [ ] Display warning badge if fallback triggered (FR-NLP-003)
- [ ] Write `tests/test_nlp.py` — mock Qwen timeout, assert fallback triggers, assert Stage 2 visible within 1,500 ms, assert no API key in logs

**SRS FRs Addressed:** FR-NLP-001–009  
**Blocking Q:** stitch-mcp Google API key must be provisioned  
**Status:** 🔴 NOT STARTED

---

## Phase 5 — Sprint E: Gradio UI & Integration

**Chain:** 10 | **Mode:** Agent | **Gate:** UI Integration Gate Check

- [ ] Implement `app.py` — Gradio 4.x UI wiring only, no pipeline logic directly (FR-DEP-001)
- [ ] Audio upload widget with 20 MB limit (FR-DEP-002)
- [ ] Progressive rendering: Stage 2 panel renders before Stage 4 completes (FR-NLP-006)
- [ ] Heatmap panel with band bar chart (FR-DEP-004, FR-DEP-005, FR-DEP-006)
- [ ] NLP explanation panel with warning badge support (FR-DEP-003)
- [ ] Four demo samples: 2 bonafide + 2 spoof (FR-DEP-008)
- [ ] About section with dataset citation (FR-DEP-009)
- [ ] End-to-end latency test ≤ 15,000 ms wall time (FR-DEP-007)
- [ ] Lock Q6: Gradio confirmed as UI framework (from config.yaml `deployment.framework: gradio`)
- [ ] Write `tests/test_e2e.py` — full pipeline integration test with real audio samples

**SRS FRs Addressed:** FR-DEP-001–010  
**Blocking Q:** Q6 (Gradio vs Streamlit) — 🔴 MUST confirm lock before Sprint E begins  
**Status:** 🔴 NOT STARTED

---

## Phase 6 — HF Spaces Deployment & Release

**Chain:** 11 | **Mode:** Composer | **Gate:** Deployment Gate Check

- [ ] Create HF Space repo — fill `config.yaml:deployment.hf_space_repo`
- [ ] Configure Spaces secrets: `QWEN_API_KEY`, `HF_TOKEN` (FR-NLP-005, NFR-Security)
- [ ] Deploy `app.py` + ONNX model to HF Spaces (FR-DEP-001)
- [ ] Verify cold-start time ≤ 30 s (NFR-Reliability)
- [ ] Verify ONNX inference ≤ 1,500 ms on Spaces 2 vCPU (FR-DEP-010)
- [ ] Smoke test with 4 demo samples (FR-DEP-008)
- [ ] Publish Space with About section and dataset citation (FR-DEP-009)

**SRS FRs Addressed:** FR-DEP-001–010  
**Blocking Q:** None (all prior Qs must be resolved)  
**Status:** 🔴 NOT STARTED

---

## Phase 7 — Monitoring, Documentation & Wrap-Up

**Chain:** 12 | **Mode:** Agent | **Gate:** Final Acceptance Gate

- [ ] Resolve Q7: finalise EER scoring protocol (scikit-learn vs ASVspoof convention)
- [ ] Run final accuracy validation: EER ≤ 10%, AUC-ROC ≥ 0.90, F1-macro ≥ 0.88 (FR-CV-008, NFR-Accuracy)
- [ ] Verify ECE ≤ 0.05 (NFR-Accuracy, Assumption)
- [ ] Verify Grad-CAM faithfulness (deletion AUC ≥ 0.65) (NFR-Accuracy, Assumption)
- [ ] Resolve all [TECH DEBT] items or document deferral rationale
- [ ] Update session-cheatsheet.md with final project status
- [ ] Produce final SRS traceability matrix (all FRs → implementation file mapping)
- [ ] Ensure `requirements.txt` all pinned, no ranges (NFR-Security, final check)

**SRS FRs Addressed:** All FR-CV-008, NFR-Accuracy, NFR-Security, NFR-Maintainability  
**Blocking Q:** Q7 (EER protocol) — 🔴 MUST resolve before accuracy validation  
**Status:** 🔴 NOT STARTED

---

## Open Question Resolution Checklist

| Q# | Question | Gate | Must Resolve By |
|----|----------|------|-----------------|
| Q3 | EfficientNet-B4 VRAM on Colab Free T4 | 🔴 OPEN | Before Sprint B |
| Q4 | Grad-CAM target layer confirmed path | 🔴 OPEN | Before Sprint C |
| Q5 | Mel bin→Hz mapping validation (n_mels=128, 16 kHz) | 🔴 OPEN | Before Sprint C |
| Q6 | Gradio framework lock confirmed | 🔴 OPEN | Before Sprint E |
| Q7 | EER protocol (scikit-learn vs ASVspoof) | 🔴 OPEN | Before Phase 7 accuracy validation |

---

*[DRAFT — Phase 0 — Pending V.E.R.I.F.Y.]*
