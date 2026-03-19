# DSDBA — Phase 1: Prioritised Sprint Backlog
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 1 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** BASELINE — update Sprint status column at sprint close

---

## Priority Classification

| Symbol | Label | Meaning |
|--------|-------|---------|
| 🔴 SHALL | Sprint Critical | System MUST implement this — failure = SRS non-compliance |
| 🟡 SHOULD | Sprint Target | Strongly desired; defer only with written justification |
| 🟢 MAY | If Time Permits | Optional enhancement; deferral does NOT violate SRS |

---

## Sprint A — Audio DSP Module

**Target File:** `src/audio/dsp.py`  
**SRS Scope:** FR-AUD-001–011  
**Blocking Q:** None (Sprint A may start immediately)  
**Gate:** Audio DSP Gate Check — all tests pass + latency ≤ 500 ms  

| FR-ID | Priority | Description | Config Key | Status |
|-------|----------|-------------|-----------|--------|
| FR-AUD-001 | 🔴 SHALL | Accept WAV and FLAC formats; reject unsupported with AUD-002 error | `audio.supported_formats` | ⬜ |
| FR-AUD-002 | 🔴 SHALL | Resample all audio to 16,000 Hz using `kaiser_best` method | `audio.sample_rate`, `audio.resampling_method` | ⬜ |
| FR-AUD-003 | 🔴 SHALL | Normalise amplitude to [-1.0, 1.0] peak range | — | ⬜ |
| FR-AUD-004 | 🔴 SHALL | Pad (zero) or trim to exactly 2.0 s / 32,000 samples | `audio.duration_sec`, `audio.n_samples` | ⬜ |
| FR-AUD-005 | 🔴 SHALL | Reject clips shorter than 0.5 s; raise `ValueError: AUD-001` | `audio.min_duration_sec`, `audio.error_code_too_short` | ⬜ |
| FR-AUD-006 | 🔴 SHALL | Compute Mel spectrogram: 128 bins, n_fft=2048, hop=512, Hann window | `audio.n_mels`, `audio.n_fft`, `audio.hop_length`, `audio.window` | ⬜ |
| FR-AUD-007 | 🔴 SHALL | Apply log-power dB scaling to Mel spectrogram | — | ⬜ |
| FR-AUD-008 | 🔴 SHALL | Resize to 224×224 and replicate to 3 channels → `[3, 224, 224]` float32 | `audio.output_tensor_shape`, `audio.output_dtype` | ⬜ |
| FR-AUD-009 | 🟡 SHOULD | Expose batch processing API for multiple clips in a single call | — | ⬜ |
| FR-AUD-010 | 🟡 SHOULD | Emit structured JSON log entry per processed clip (no `print()`) | — | ⬜ |
| FR-AUD-011 | 🟢 MAY | Accept MP3 and OGG via ffmpeg/pydub; surface as optional formats | `audio.optional_formats` | ⬜ |

**Deferral note:** FR-AUD-009 and FR-AUD-010 may be deferred to Sprint D clean-up without SRS violation; deferral must be recorded in session-cheatsheet.md. FR-AUD-011 may be deferred to Phase 7 wrap-up.

---

## Sprint B — CV Model Training + ONNX Export

**Target Files:** `src/cv/model.py`, `src/cv/train.py`, `src/cv/infer.py`  
**SRS Scope:** FR-CV-001–009, FR-DEP-010  
**Blocking Q:** Q3 (EfficientNet-B4 VRAM on Colab T4) — 🔴 MUST resolve before Sprint B starts  
**Gate:** CV Training Gate Check — accuracy targets met + checkpoint on HF Hub  

| FR-ID | Priority | Description | Config Key | Status |
|-------|----------|-------------|-----------|--------|
| FR-CV-001 | 🔴 SHALL | EfficientNet-B4 with ImageNet-1k pretrained weights | `model.backbone`, `model.pretrained_weights` | ⬜ |
| FR-CV-002 | 🔴 SHALL | 2-class linear classification head (bonafide=0, spoof=1) | `model.num_classes`, `model.head_type` | ⬜ |
| FR-CV-003 | 🔴 SHALL | Two-phase training: 5 epochs frozen backbone → full fine-tune at lr=1e-4 | `model.frozen_epochs`, `model.finetune_lr` | ⬜ |
| FR-CV-004 | 🔴 SHALL | Sigmoid output activation; decision threshold = 0.5 | `model.output_activation`, `model.decision_threshold` | ⬜ |
| FR-CV-005 | 🔴 SHALL | BCE loss with per-class weights to handle dataset imbalance | `model.loss_function` | ⬜ |
| FR-CV-006 | 🟡 SHOULD | SpecAugment: time_mask ≤ 10%, freq_mask ≤ 10%, time_shift ±0.1 s, SNR ≥ 20 dB | `training.augmentation.*` | ⬜ |
| FR-CV-007 | 🔴 SHALL | Upload checkpoint to HF Hub after training completes | `training.hf_model_repo` | ⬜ |
| FR-CV-008 | 🔴 SHALL | Achieve AUC-ROC ≥ 0.90 and EER ≤ 10% on FoR test set | `acceptance_criteria.target_auc_roc`, `acceptance_criteria.target_eer` | ⬜ |
| FR-CV-009 | 🟢 MAY | Train EfficientNet-B0 baseline for ablation comparison | — | ⬜ |
| FR-DEP-010 | 🔴 SHALL | Export `.pth` → `.onnx`; numerical equivalence \|diff\| < 1e-5; ONNX latency ≤ 1,500 ms CPU | `deployment.onnx_equivalence_tolerance`, `deployment.onnx_latency_target_ms` | ⬜ |

**Deferral note:** FR-CV-006 (SpecAugment) deferred = reduced generalisation; acceptable risk if timeline is tight. FR-CV-009 is fully deferrable.

---

## Sprint C — XAI / Grad-CAM Module

**Target File:** `src/cv/gradcam.py`  
**SRS Scope:** FR-CV-010–016  
**Blocking Q:** Q4 (Grad-CAM target layer) AND Q5 (Mel bin→Hz mapping) — 🔴 BOTH must resolve before Sprint C starts  
**Gate:** Grad-CAM Gate Check — non-zero saliency, band pct sums to 100%, latency ≤ 3,000 ms  

| FR-ID | Priority | Description | Config Key | Status |
|-------|----------|-------------|-----------|--------|
| FR-CV-010 | 🔴 SHALL | Target Grad-CAM at final MBConv convolutional block | `gradcam.target_layer` | ⬜ |
| FR-CV-011 | 🔴 SHALL | Use `jacobgil/pytorch-grad-cam` library exclusively | `gradcam.library` | ⬜ |
| FR-CV-012 | 🔴 SHALL | Generate heatmap overlay: jet colormap, α=0.5, 224×224 PNG output | `gradcam.colormap`, `gradcam.overlay_alpha`, `gradcam.output_format` | ⬜ |
| FR-CV-013 | 🔴 SHALL | Compute Mel-bin attribution scores across 4 frequency bands (0–500, 500–2k, 2k–4k, 4k–8k Hz) | `gradcam.band_hz.*` | ⬜ |
| FR-CV-014 | 🔴 SHALL | Normalise band attribution via Softmax so scores sum to 100% | `gradcam.band_normalisation` | ⬜ |
| FR-CV-015 | 🔴 SHALL | Grad-CAM latency ≤ 3,000 ms on CPU | `gradcam.latency_target_ms` | ⬜ |
| FR-CV-016 | 🟡 SHOULD | Expose raw saliency tensor as JSON endpoint for developer access | `gradcam.expose_raw_saliency` | ⬜ |

**Deferral note:** FR-CV-016 is SHOULD; deferring it does not affect user-facing functionality. Defer only if Sprint C is behind schedule.

---

## Sprint D — NLP Explanation Module

**Target File:** `src/nlp/explain.py`  
**SRS Scope:** FR-NLP-001–009  
**Blocking Q:** stitch-mcp Google API key must be provisioned; Qwen 2.5 endpoint reachable  
**Gate:** NLP Explanation Gate Check — fallback triggers correctly; no API key in logs; Stage 2 visible < 1,500 ms  

| FR-ID | Priority | Description | Config Key | Status |
|-------|----------|-------------|-----------|--------|
| FR-NLP-001 | 🔴 SHALL | Generate explanation of 3–5 English sentences | `nlp.explanation_min_sentences`, `nlp.explanation_max_sentences` | ⬜ |
| FR-NLP-002 | 🔴 SHALL | Primary LLM: Qwen 2.5 via async API call (timeout = 30 s) | `nlp.primary_provider`, `nlp.timeout_sec` | ⬜ |
| FR-NLP-003 | 🔴 SHALL | Rule-based fallback template — zero external dependencies | `nlp.final_fallback`, `nlp.warning_badge_text` | ⬜ |
| FR-NLP-004 | 🔴 SHALL | All explanations output in English | `nlp.output_language` | ⬜ |
| FR-NLP-005 | 🔴 SHALL | API key read from `QWEN_API_KEY` env var only — never in code, logs, or Git | `nlp.api_key_env_var` | ⬜ |
| FR-NLP-006 | 🔴 SHALL | NLP call is non-blocking; Stage 2 result (label + heatmap) renders before NLP completes | — | ⬜ |
| FR-NLP-007 | 🟡 SHOULD | Gemma-3 secondary fallback if Qwen 2.5 times out | `nlp.fallback_provider` | ⬜ |
| FR-NLP-008 | 🟡 SHOULD | Cache LLM responses keyed on `(label, confidence_bucket, top_band)` | `nlp.caching.*` | ⬜ |
| FR-NLP-009 | 🟢 MAY | Language toggle in UI (e.g., English / Indonesian) | — | ⬜ |

**Deferral note:** FR-NLP-007 deferral = single fallback only (rule-based); acceptable risk for MVP. FR-NLP-008 deferral = increased API costs on repeated queries; acceptable for demo. FR-NLP-009 is fully deferrable.

---

## Sprint E — Gradio UI & Integration

**Target File:** `app.py`  
**SRS Scope:** FR-DEP-001–010 (FR-DEP-010 implemented in Sprint B)  
**Blocking Q:** Q6 (Gradio vs Streamlit framework lock) — 🔴 MUST confirm before Sprint E starts  
**Gate:** UI Integration Gate Check — end-to-end latency ≤ 15,000 ms; all panels render  

| FR-ID | Priority | Description | Config Key | Status |
|-------|----------|-------------|-----------|--------|
| FR-DEP-001 | 🔴 SHALL | Gradio 4.x UI framework; public access, no login required | `deployment.framework`, `deployment.gradio_version`, `deployment.auth_required` | ⬜ |
| FR-DEP-002 | 🔴 SHALL | Audio upload widget with 20 MB file size limit | `deployment.max_upload_mb` | ⬜ |
| FR-DEP-003 | 🔴 SHALL | NLP explanation panel with AI warning badge on fallback | `nlp.warning_badge_text` | ⬜ |
| FR-DEP-004 | 🔴 SHALL | Heatmap panel displaying Grad-CAM overlay PNG | — | ⬜ |
| FR-DEP-005 | 🔴 SHALL | Frequency band bar chart (4 bands, softmax normalised %) | — | ⬜ |
| FR-DEP-006 | 🔴 SHALL | Component streaming: Stage 2 panel renders before Stage 3 NLP text completes | — | ⬜ |
| FR-DEP-007 | 🔴 SHALL | End-to-end wall time: upload → full result ≤ 15,000 ms | `deployment.e2e_latency_target_ms` | ⬜ |
| FR-DEP-008 | 🟡 SHOULD | Four bundled demo samples: 2 bonafide + 2 spoof | `deployment.demo_samples.*` | ⬜ |
| FR-DEP-009 | 🟡 SHOULD | About section with dataset citation (Abdel-Dayem, M. 2023) | `deployment.about_section`, `deployment.dataset_citation` | ⬜ |

**Deferral note:** FR-DEP-008 and FR-DEP-009 are SHOULD; they improve UX but their absence does not break core functionality. Defer only if Sprint E is behind schedule.

---

## Deferred Requirements Summary

The following requirements are classified MAY or SHOULD and may be deferred without SRS violation. Each must be recorded with a deferral rationale if postponed.

| FR-ID | Priority | Sprint | Deferral Impact |
|-------|----------|--------|-----------------|
| FR-AUD-009 | SHOULD | A | No batch API; single-clip pipeline only (acceptable for demo) |
| FR-AUD-010 | SHOULD | A | Structured log may be replaced by basic logger without JSON format |
| FR-AUD-011 | MAY | A | MP3/OGG not supported; WAV/FLAC only (SRS-compliant) |
| FR-CV-006 | SHOULD | B | Reduced augmentation; may slightly reduce generalisation |
| FR-CV-009 | MAY | B | No ablation baseline; academic comparison unavailable |
| FR-CV-016 | SHOULD | C | No raw saliency endpoint; developer tooling only |
| FR-NLP-007 | SHOULD | D | Single rule-based fallback; no Gemma-3 secondary tier |
| FR-NLP-008 | SHOULD | D | No caching; repeated inputs incur full API cost |
| FR-NLP-009 | MAY | D | English-only output |
| FR-DEP-008 | SHOULD | E | No demo samples; users must upload own files |
| FR-DEP-009 | SHOULD | E | No About section; dataset citation absent from UI |

---

## Q6 Framework Decision — Recommendation

### Decision Scope
Open Question Q6 asks whether Gradio 4.x or Streamlit should be used as the deployment UI framework (FR-DEP-001). This section provides the technical rationale to support the final lock at Phase 2 gate.

### Evaluation

| Criterion | Gradio 4.x | Streamlit |
|-----------|-----------|-----------|
| Native audio widget | ✅ `gr.Audio` — upload + playback built-in | ❌ Requires `audio_recorder_streamlit` or custom component |
| HF Spaces deployment | ✅ First-party support; Spaces defaults to Gradio | ⚠️ Supported but secondary; more configuration required |
| Async/streaming API | ✅ `gr.Interface` supports `yield`-based generator streaming (FR-NLP-006, FR-DEP-006) | ⚠️ Streaming via `st.empty()` — less composable with async pipeline |
| Component-level updates | ✅ Independent panel updates support Stage 2→Stage 3 progressive render | ⚠️ Full re-run model complicates partial panel rendering |
| PNG image display | ✅ `gr.Image` renders PIL Image / numpy array directly | ✅ `st.image` equivalent |
| Bar chart (band scores) | ✅ `gr.BarPlot` — native, no matplotlib required in UI layer | ⚠️ Requires `st.bar_chart` or plotly import |
| Framework lock in config.yaml | ✅ `deployment.framework: gradio` already set | Requires config change + dependency swap |
| Community adoption for ML demos | ✅ Standard for HF model demos | ✅ Common in data science contexts |

### Recommendation

**Select Gradio 4.x.**

Rationale:
1. `gr.Audio` provides the exact upload+playback widget required by FR-DEP-002 without additional dependencies.
2. Generator-based streaming (`yield` in the inference function) directly satisfies FR-NLP-006 (Stage 2 before Stage 3) and FR-DEP-006 (component streaming) without architectural workarounds.
3. HF Spaces first-party support reduces deployment friction for FR-DEP-001 (HF Spaces target).
4. `config.yaml:deployment.framework` is already set to `gradio`; changing it requires a breaking config update and dependency swap with no technical benefit.
5. `gr.BarPlot` eliminates the need for matplotlib in the UI layer (aligns with module boundary rule: `app.py` → UI wiring only).

**Q6 Status after this recommendation:** 🟡 RECOMMENDED — not officially closed until Phase 2 gate review confirms no new constraints.

---

*[DRAFT — Phase 1 — Pending V.E.R.I.F.Y.]*
