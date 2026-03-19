# DSDBA — Phase 0 Architectural Risk Register
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 0 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** APPROVED — Phase 0 baseline

---

## Purpose

This register identifies the three highest-priority architectural risks in the DSDBA pipeline before any implementation begins. Each risk entry includes its SRS coupling point, severity classification, and the mitigation strategy that must be in place before the relevant sprint starts.

---

## Risk 1 — Tensor Shape Contract Violation

| Field | Detail |
|-------|--------|
| **Risk ID** | RISK-001 |
| **Title** | Silent downstream corruption via [3, 224, 224] shape mismatch |
| **Severity** | CRITICAL |
| **Probability** | MEDIUM |
| **Coupling Point** | FR-AUD-008 → FR-CV-001 interface |
| **Blocks** | All CV Inference, Grad-CAM, and NLP stages |

### Description

The pipeline interface contract between the Audio DSP stage and the CV Inference stage requires the output tensor to be exactly `[3, 224, 224]` with dtype `float32`. If `src/audio/dsp.py` produces any other shape (e.g., `[1, 128, 128]` from a single-channel Mel spectrogram without channel replication), the EfficientNet-B4 forward pass will either:

- Raise a runtime `RuntimeError` with no meaningful diagnostic (best case), or  
- Silently accept the malformed tensor if upstream dims accidentally match internal layer expectations (worst case — undetectable corruption).

This risk is elevated because the spectrogram-to-RGB channel replication step (duplicating the Mel spectrogram across 3 channels to satisfy ImageNet-pretrained weight expectations) is a non-obvious transformation that is easy to omit or mis-implement.

### SRS Coupling

- **FR-AUD-008 SHALL:** DSP module output MUST be `[3, 224, 224]` float32.
- **FR-CV-001 SHALL:** EfficientNet-B4 input MUST be `[3, 224, 224]` float32.
- **config.yaml:** `audio.output_tensor_shape: [3, 224, 224]` and `audio.output_dtype: float32` are the single source of truth.

### Mitigation

1. **Guard assertion at CV module entry** (`src/cv/infer.py` and `src/cv/gradcam.py`): validate `tensor.shape == (3, 224, 224)` and `tensor.dtype == torch.float32` before any forward pass.
2. **Unit test** `tests/test_cv.py`: assertion test using a malformed tensor — must raise `ValueError` with error code `AUD-001` per FR-AUD-005.
3. **DSP output test** `tests/test_audio.py`: assert shape and dtype on the output of `dsp.process()` for every supported format.
4. **No magic numbers**: shape values read from `config.yaml:audio.output_tensor_shape` — never hardcoded.

---

## Risk 2 — Grad-CAM Target Layer Misidentification

| Field | Detail |
|-------|--------|
| **Risk ID** | RISK-002 |
| **Title** | Wrong saliency map due to incorrect Grad-CAM target layer path (Q4) |
| **Severity** | HIGH |
| **Probability** | MEDIUM |
| **Coupling Point** | FR-CV-010 → `model.features[-1]` must be confirmed via PyTorch introspection |
| **Blocks** | Sprint C (Grad-CAM), Sprint D (NLP band attribution), Phase 5 XAI faithfulness metric |

### Description

The `pytorch-grad-cam` library (jacobgil) requires an explicit reference to the target convolutional layer object inside the model. For torchvision EfficientNet-B4, the final feature block is nominally accessible as `model.features[-1]`. However:

- The exact layer path changes if any model surgery is applied (head replacement, feature freezing, or custom wrapper classes).
- `model.features[-1]` may resolve to a `Sequential` container rather than the last `Conv2d` — passing a `Sequential` to `GradCAM` produces invalid (zero or NaN) saliency.
- If the wrong layer is targeted, the band attribution scores (`band_pct[4]`) fed to the NLP stage become semantically meaningless, yet no error is raised.

### SRS Coupling

- **FR-CV-010 SHALL:** Grad-CAM MUST use the final MBConv convolutional block.
- **FR-CV-011 SHALL:** Library MUST be `jacobgil/pytorch-grad-cam`.
- **config.yaml:** `gradcam.target_layer: "model.features[-1]"` — locked here, subject to empirical confirmation in Sprint C.
- **Open Question Q4:** 🔴 OPEN — MUST be resolved before Sprint C begins.

### Mitigation

1. **PyTorch introspection script** (Sprint C entry task): iterate `model.named_modules()` and print all layer names. Identify the last `Conv2d` before the classifier head. Update `config.yaml:gradcam.target_layer` with the exact confirmed path.
2. **Smoke test**: run a single forward-backward pass and assert that the Grad-CAM heatmap contains non-zero, non-NaN values with spatial variance > 0.
3. **Lock in config.yaml before Sprint C**: any change to `gradcam.target_layer` after Sprint C baseline is a breaking change requiring full Grad-CAM re-validation.
4. **Q4 gate rule** (from `.cursorrules`): Sprint C MUST NOT start until Q4 is resolved.

---

## Risk 3 — Qwen 2.5 API Latency Blocks CV Result Display

| Field | Detail |
|-------|--------|
| **Risk ID** | RISK-003 |
| **Title** | Synchronous NLP call blocks UI from displaying Stage 2 results |
| **Severity** | HIGH |
| **Probability** | HIGH |
| **Coupling Point** | FR-NLP-006 → UI MUST display Stage 2 result before Stage 3 (NLP) completes |
| **Blocks** | Sprint E (UI/UX), FR-DEP-007 (end-to-end latency ≤ 15,000 ms) |

### Description

Qwen 2.5 API calls have a `timeout_sec: 30` ceiling (config.yaml). In a naive synchronous implementation, the Gradio UI would block the full 30 seconds before showing any result to the user if the API is slow or unavailable. This violates:

- **FR-NLP-006**: Stage 2 (label + confidence + heatmap) MUST be rendered before Stage 3 (NLP text) is available.
- **FR-DEP-007**: end-to-end wall time ≤ 15,000 ms — a 30-second API stall alone exceeds this.
- **NFR-Usability**: users must receive actionable feedback within the 15-second budget.

Additionally, network-level flakiness on HF Spaces (shared infrastructure) means Qwen 2.5 availability cannot be guaranteed at inference time.

### SRS Coupling

- **FR-NLP-002 SHALL:** Primary LLM is Qwen 2.5 via async API call.
- **FR-NLP-003 SHALL:** Rule-based fallback MUST always be available — no dependency on external API for core result.
- **FR-NLP-006 SHALL:** Progressive UI rendering — Stage 2 before Stage 3.
- **FR-NLP-007 SHOULD:** Gemma 3 secondary fallback if Qwen times out.
- **config.yaml:** `nlp.timeout_sec: 30`, `nlp.final_fallback: rule_based`.

### Mitigation

1. **`asyncio.Task` isolation** (`src/nlp/explain.py`): NLP call runs as a non-blocking coroutine; Gradio receives Stage 2 result immediately via `yield` or streaming update.
2. **Three-tier fallback chain**: Qwen 2.5 → Gemma 3 (FR-NLP-007) → rule-based template (FR-NLP-003). Rule-based fallback has zero external dependencies.
3. **NLP caching** (FR-NLP-008): cache keyed on `(label, confidence_bucket, top_band)` — eliminates redundant API calls for repeated inputs.
4. **UI warning badge**: if fallback is triggered, display `config.yaml:nlp.warning_badge_text` ("AI explanation unavailable") — never show a blank or error state to user.
5. **Latency test** `tests/test_nlp.py`: mock Qwen timeout and assert Stage 2 result arrives within 1,500 ms even when NLP stage is pending.

---

## Risk Summary Table

| Risk ID | Title | Severity | Sprint Gate | Status |
|---------|-------|----------|-------------|--------|
| RISK-001 | Tensor shape contract violation | CRITICAL | Sprint A/B | 🔴 OPEN — mitigations pending implementation |
| RISK-002 | Grad-CAM layer misidentification | HIGH | Sprint C | 🔴 OPEN — Q4 unresolved |
| RISK-003 | Qwen 2.5 latency blocks UI | HIGH | Sprint E | 🔴 OPEN — async design required |

---

*[DRAFT — Phase 0 — Pending V.E.R.I.F.Y.]*
