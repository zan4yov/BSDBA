# DSDBA — Phase 1: Requirement Traceability Matrix (RTM)
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 1 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Standard:** IEEE Std 830-1998 Requirements Traceability  
**Status:** BASELINE — update Status column at sprint close gate

---

## How to Use This Document

- **Status** column values: `⬜ NOT STARTED` → `🔵 IN PROGRESS` → `🟢 IMPLEMENTED` → `✅ VERIFIED`
- A requirement is **VERIFIED** only when the corresponding test passes AND the Gate Check returns 🟢 PASS.
- **Blocking Q** column — if a Q is 🔴 OPEN, the sprint containing that FR must NOT start.
- Update this RTM at the close of every sprint session.

---

## Stage 1 — Audio DSP: FR-AUD-001 to FR-AUD-011

**Module File:** `src/audio/dsp.py`  
**Sprint:** A  
**Test File:** `tests/test_audio.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-AUD-001 | Accept WAV and FLAC; reject unsupported with error AUD-002 | SHALL | A | `src/audio/dsp.py` | None | `audio.supported_formats`, `audio.error_code_bad_format` | ⬜ |
| FR-AUD-002 | Resample to 16,000 Hz using `kaiser_best` method via librosa | SHALL | A | `src/audio/dsp.py` | None | `audio.sample_rate`, `audio.resampling_method` | ⬜ |
| FR-AUD-003 | Normalise amplitude to [-1.0, 1.0] peak range | SHALL | A | `src/audio/dsp.py` | None | — | ⬜ |
| FR-AUD-004 | Pad with zeros or trim to exactly 2.0 s / 32,000 samples | SHALL | A | `src/audio/dsp.py` | None | `audio.duration_sec`, `audio.n_samples` | ⬜ |
| FR-AUD-005 | Reject clips < 0.5 s; raise `ValueError` with error code AUD-001 | SHALL | A | `src/audio/dsp.py` | None | `audio.min_duration_sec`, `audio.error_code_too_short` | ⬜ |
| FR-AUD-006 | Mel spectrogram: 128 bins, n_fft=2048, hop=512, Hann window | SHALL | A | `src/audio/dsp.py` | None | `audio.n_mels`, `audio.n_fft`, `audio.hop_length`, `audio.window` | ⬜ |
| FR-AUD-007 | Log-power dB conversion applied to Mel spectrogram | SHALL | A | `src/audio/dsp.py` | None | — | ⬜ |
| FR-AUD-008 | Resize to 224×224, replicate to 3 channels → `[3, 224, 224]` float32 | SHALL | A | `src/audio/dsp.py` | None | `audio.output_tensor_shape`, `audio.output_dtype` | ⬜ |
| FR-AUD-009 | Batch processing API: accept list of clips, return list of tensors | SHOULD | A | `src/audio/dsp.py` | None | — | ⬜ |
| FR-AUD-010 | Emit structured JSON log entry per processed clip; no `print()` | SHOULD | A | `src/audio/dsp.py` | None | — | ⬜ |
| FR-AUD-011 | Accept MP3 and OGG via ffmpeg/pydub as optional formats | MAY | A | `src/audio/dsp.py` | None | `audio.optional_formats` | ⬜ |

**Sprint A Entry Criteria:** No blocking Q — may start at Phase 2 kickoff.  
**Sprint A Exit Criteria:** All SHALL FRs 🟢 IMPLEMENTED + test_audio.py 🟢 PASS + latency ≤ 500 ms.

---

## Stage 2a — CV Model Definition & Training: FR-CV-001 to FR-CV-009

**Module Files:** `src/cv/model.py`, `src/cv/train.py`  
**Sprint:** B  
**Test File:** `tests/test_train.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-CV-001 | EfficientNet-B4 backbone with ImageNet-1k pretrained weights | SHALL | B | `src/cv/model.py` | Q3 | `model.backbone`, `model.pretrained_weights` | ⬜ |
| FR-CV-002 | Linear 2-class classification head (bonafide=0, spoof=1) | SHALL | B | `src/cv/model.py` | Q3 | `model.num_classes`, `model.head_type` | ⬜ |
| FR-CV-003 | Two-phase training: 5 epochs frozen → full fine-tune at lr=1e-4 | SHALL | B | `src/cv/train.py` | Q3 | `model.frozen_epochs`, `model.finetune_lr` | ⬜ |
| FR-CV-004 | Sigmoid activation on output logit; decision threshold = 0.5 | SHALL | B | `src/cv/model.py` | Q3 | `model.output_activation`, `model.decision_threshold` | ⬜ |
| FR-CV-005 | Binary Cross-Entropy loss with per-class weights | SHALL | B | `src/cv/train.py` | Q3 | `model.loss_function` | ⬜ |
| FR-CV-006 | SpecAugment: time_mask ≤ 10%, freq_mask ≤ 10%, shift ±0.1 s, SNR ≥ 20 dB | SHOULD | B | `src/cv/train.py` | Q3 | `training.augmentation.*` | ⬜ |
| FR-CV-007 | Upload trained checkpoint to HF Hub after training | SHALL | B | `src/cv/train.py` | Q3 | `training.hf_model_repo` | ⬜ |
| FR-CV-008 | Achieve AUC-ROC ≥ 0.90 and EER ≤ 10% on FoR test set | SHALL | B | `src/cv/train.py` | Q3 | `acceptance_criteria.target_auc_roc`, `acceptance_criteria.target_eer` | ⬜ |
| FR-CV-009 | EfficientNet-B0 baseline training for ablation comparison | MAY | B | `src/cv/train.py` | Q3 | — | ⬜ |

**Sprint B Entry Criteria:** Q3 (VRAM on Colab T4) MUST be resolved (🟢 VERIFIED) before Sprint B starts.  
**Sprint B Exit Criteria:** All SHALL FRs 🟢 IMPLEMENTED + FR-CV-008 accuracy targets met + checkpoint verified on HF Hub.

---

## Stage 2b — ONNX Export & CV Inference: FR-DEP-010 + FR-CV-004

**Module File:** `src/cv/infer.py`  
**Sprint:** B (ONNX export runs immediately after training)  
**Test File:** `tests/test_cv.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-DEP-010 | Export `.pth` → `.onnx`; \|ONNX − PyTorch\| < 1e-5; ONNX CPU latency ≤ 1,500 ms; ONNX session created once at startup | SHALL | B | `src/cv/infer.py` | Q3 | `deployment.onnx_enabled`, `deployment.onnx_equivalence_tolerance`, `deployment.onnx_latency_target_ms`, `deployment.onnx_execution_providers` | ⬜ |

**Note:** FR-DEP-010 is listed under Sprint B because ONNX export requires a trained `.pth` checkpoint. The `infer.py` module (ONNX Runtime wrapper) is implemented immediately after training completes.

---

## Stage 3 — XAI / Grad-CAM: FR-CV-010 to FR-CV-016

**Module File:** `src/cv/gradcam.py`  
**Sprint:** C  
**Test File:** `tests/test_cv.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-CV-010 | Grad-CAM targets the final MBConv convolutional block | SHALL | C | `src/cv/gradcam.py` | Q4 | `gradcam.target_layer` | ⬜ |
| FR-CV-011 | Grad-CAM implementation via `jacobgil/pytorch-grad-cam` | SHALL | C | `src/cv/gradcam.py` | Q4 | `gradcam.library` | ⬜ |
| FR-CV-012 | Heatmap overlay: jet colormap, α=0.5 blending, 224×224 PNG output | SHALL | C | `src/cv/gradcam.py` | Q4 | `gradcam.colormap`, `gradcam.overlay_alpha`, `gradcam.output_format` | ⬜ |
| FR-CV-013 | Compute band attribution scores: 4 Hz bands (0–500, 500–2k, 2k–4k, 4k–8k) | SHALL | C | `src/cv/gradcam.py` | Q4, Q5 | `gradcam.band_hz.*` | ⬜ |
| FR-CV-014 | Softmax normalise band attribution so all 4 bands sum to 100% | SHALL | C | `src/cv/gradcam.py` | Q5 | `gradcam.band_normalisation` | ⬜ |
| FR-CV-015 | Grad-CAM end-to-end latency ≤ 3,000 ms on CPU | SHALL | C | `src/cv/gradcam.py` | Q4 | `gradcam.latency_target_ms` | ⬜ |
| FR-CV-016 | Expose raw saliency tensor as JSON endpoint for developer access | SHOULD | C | `src/cv/gradcam.py` | Q4 | `gradcam.expose_raw_saliency` | ⬜ |

**Sprint C Entry Criteria:** Q4 (Grad-CAM target layer) AND Q5 (Mel bin→Hz mapping) MUST BOTH be resolved before Sprint C starts.  
**Sprint C Exit Criteria:** All SHALL FRs 🟢 IMPLEMENTED + saliency non-zero + band pct sums to 100% + latency ≤ 3,000 ms.

---

## Stage 4 — NLP Explanation: FR-NLP-001 to FR-NLP-009

**Module File:** `src/nlp/explain.py`  
**Sprint:** D  
**Test File:** `tests/test_nlp.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-NLP-001 | Generate explanation of 3–5 sentences; prompt enforces length | SHALL | D | `src/nlp/explain.py` | stitch-mcp provisioned | `nlp.explanation_min_sentences`, `nlp.explanation_max_sentences` | ⬜ |
| FR-NLP-002 | Primary LLM: Qwen 2.5 via async API call; timeout = 30 s | SHALL | D | `src/nlp/explain.py` | stitch-mcp provisioned | `nlp.primary_provider`, `nlp.timeout_sec` | ⬜ |
| FR-NLP-003 | Rule-based fallback always available; display warning badge on activation | SHALL | D | `src/nlp/explain.py` | None | `nlp.final_fallback`, `nlp.warning_badge_text` | ⬜ |
| FR-NLP-004 | All LLM explanations output in English | SHALL | D | `src/nlp/explain.py` | None | `nlp.output_language` | ⬜ |
| FR-NLP-005 | API key sourced from `QWEN_API_KEY` env var only; never in code/logs/Git | SHALL | D | `src/nlp/explain.py` | None | `nlp.api_key_env_var` | ⬜ |
| FR-NLP-006 | NLP call is non-blocking; Stage 2 result renders before NLP text arrives | SHALL | D | `src/nlp/explain.py` | None | — | ⬜ |
| FR-NLP-007 | Gemma-3 secondary fallback if Qwen 2.5 times out or errors | SHOULD | D | `src/nlp/explain.py` | None | `nlp.fallback_provider` | ⬜ |
| FR-NLP-008 | Cache LLM responses keyed on `(label, confidence_bucket, top_band)` | SHOULD | D | `src/nlp/explain.py` | None | `nlp.caching.*` | ⬜ |
| FR-NLP-009 | Language toggle in UI (English / Indonesian) | MAY | D | `src/nlp/explain.py` | None | — | ⬜ |

**Sprint D Entry Criteria:** stitch-mcp Google API key provisioned; Qwen 2.5 endpoint reachable from dev environment.  
**Sprint D Exit Criteria:** All SHALL FRs 🟢 IMPLEMENTED + mock timeout test 🟢 PASS + Stage 2 visible < 1,500 ms + zero API key in logs.

---

## Stage 5 — Deployment / UI: FR-DEP-001 to FR-DEP-009

**Module File:** `app.py`  
**Sprint:** E  
**Test File:** `tests/test_e2e.py`

| FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Config Key | Status |
|-------|-------------|----------|--------|-------------|-----------|-----------|--------|
| FR-DEP-001 | Gradio 4.x UI; public access; no login required | SHALL | E | `app.py` | Q6 | `deployment.framework`, `deployment.gradio_version`, `deployment.auth_required` | ⬜ |
| FR-DEP-002 | Audio upload widget; max file size 20 MB | SHALL | E | `app.py` | Q6 | `deployment.max_upload_mb` | ⬜ |
| FR-DEP-003 | NLP explanation panel with AI-unavailable warning badge on fallback | SHALL | E | `app.py` | Q6 | `nlp.warning_badge_text` | ⬜ |
| FR-DEP-004 | Heatmap panel displaying Grad-CAM overlay PNG | SHALL | E | `app.py` | Q6 | — | ⬜ |
| FR-DEP-005 | Frequency band bar chart (4 bands, softmax normalised %) | SHALL | E | `app.py` | Q6 | — | ⬜ |
| FR-DEP-006 | Component streaming: Stage 2 panel renders before Stage 3 NLP completes | SHALL | E | `app.py` | Q6 | — | ⬜ |
| FR-DEP-007 | End-to-end wall time: upload → full result display ≤ 15,000 ms | SHALL | E | `app.py` | Q6 | `deployment.e2e_latency_target_ms` | ⬜ |
| FR-DEP-008 | Four bundled demo samples: 2 bonafide + 2 spoof accessible in UI | SHOULD | E | `app.py` | Q6 | `deployment.demo_samples.*` | ⬜ |
| FR-DEP-009 | About section with Abdel-Dayem (2023) FoR dataset citation | SHOULD | E | `app.py` | Q6 | `deployment.about_section`, `deployment.dataset_citation` | ⬜ |

**Sprint E Entry Criteria:** Q6 (Gradio vs Streamlit) MUST be resolved (locked to Gradio) before Sprint E starts.  
**Sprint E Exit Criteria:** All SHALL FRs 🟢 IMPLEMENTED + e2e test ≤ 15,000 ms + all panels render correctly.

---

## RTM Summary — Coverage by Stage

| Stage | Sprint | SHALL FRs | SHOULD FRs | MAY FRs | Total FRs | Blocking Q |
|-------|--------|-----------|------------|---------|-----------|-----------|
| Audio DSP | A | 8 | 2 | 1 | 11 | None |
| CV Training | B | 7 | 1 | 1 | 9 | Q3 |
| ONNX Export | B | 1 | 0 | 0 | 1 | Q3 |
| XAI Grad-CAM | C | 6 | 1 | 0 | 7 | Q4, Q5 |
| NLP Explanation | D | 6 | 2 | 1 | 9 | stitch-mcp key |
| Deployment UI | E | 7 | 2 | 0 | 9 | Q6 |
| **Total** | — | **35** | **8** | **3** | **46** | — |

> **Note:** The SRS document references 37 FR IDs in the chain-02 prompt scope. The baseline RTM above traces all 46 FR-IDs derived from the four SRS sections (FR-AUD-001–011, FR-CV-001–016, FR-NLP-001–009, FR-DEP-001–010) as defined in `config.yaml` and `.cursorrules`. The count of 46 supersedes the 37 figure; all 46 must be implemented for full SRS compliance.

---

## Open Questions Impact on RTM

| Q# | Question | Status | FRs Blocked |
|----|----------|--------|------------|
| Q3 | EfficientNet-B4 VRAM on Colab Free T4 | 🟡 LIKELY FEASIBLE | FR-CV-001 to FR-CV-009, FR-DEP-010 |
| Q4 | Grad-CAM target layer confirmed path | 🟡 CANDIDATE CONFIRMED — needs introspection | FR-CV-010, FR-CV-011, FR-CV-012, FR-CV-015, FR-CV-016 |
| Q5 | Mel bin→Hz mapping validation (n_mels=128, 16 kHz) | 🔴 OPEN | FR-CV-013, FR-CV-014 |
| Q6 | Gradio framework lock confirmed | 🔴 OPEN (Gradio recommended — see phase1-backlog.md) | FR-DEP-001 to FR-DEP-009 |
| Q7 | EER scoring protocol (scikit-learn vs ASVspoof) | 🔴 OPEN | FR-CV-008 (accuracy validation in Phase 7) |

---

## Non-Functional Requirements Cross-Reference

| NFR | Description | Enforcement Point | FRs Coupled |
|-----|-------------|------------------|-------------|
| NFR-Performance | Audio DSP ≤ 500 ms, ONNX ≤ 1,500 ms, Grad-CAM ≤ 3,000 ms, E2E ≤ 15,000 ms | Latency assertions in `tests/` | FR-AUD-*, FR-CV-015, FR-DEP-007, FR-DEP-010 |
| NFR-Accuracy | AUC-ROC ≥ 0.90, EER ≤ 10%, F1-macro ≥ 0.88, accuracy ≥ 90% | `tests/test_train.py` acceptance gate | FR-CV-008 |
| NFR-Security | API key in env var only; audio in-memory only; all deps pinned | S.H.I.E.L.D. rules in `.cursorrules` | FR-NLP-005, FR-AUD-*, NFR-wide |
| NFR-Maintainability | No magic numbers; all constants from `config.yaml` | R.E.F.A.C.T. rules in `.cursorrules` | All FR groups |
| NFR-Reliability | Checkpoint every epoch; cold-start ≤ 30 s; rule-based fallback always available | `training.save_every_epoch`, FR-NLP-003 | FR-CV-008, FR-NLP-003 |
| NFR-Scalability | ONNX Runtime on 2 vCPU HF Spaces; no GPU dependency at inference | `deployment.onnx_execution_providers` | FR-DEP-010 |

---

*[DRAFT — Phase 1 — Pending V.E.R.I.F.Y.]*
