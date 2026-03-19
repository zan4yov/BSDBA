# ADR-0007 — Phase 2: Module Interface Contracts
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 2 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** BASELINE — binding contract for Sprints A–E  
**SRS References:** All FR groups (FR-AUD-*, FR-CV-*, FR-NLP-*, FR-DEP-*)  
**Standard:** IEEE Std 830-1998 Interface Specification  

> **Rule:** These signatures are LOCKED after Phase 2 Gate Check. Sprint implementations MUST conform. Any deviation requires a formal Phase 2 amendment and session-cheatsheet.md update.

---

## Pipeline Data Flow (Reference)

```
[Audio File]
     │  Path
     ▼
┌─────────────────────────────────────────────────────┐
│  src/audio/dsp.py :: preprocess_audio()             │
│  FR-AUD-001–011  │  ≤ 500 ms on CPU                │
└───────────────────────────┬─────────────────────────┘
                            │ torch.Tensor [3, 224, 224] float32
                            ▼
        ┌───────────────────┴───────────────────────┐
        │                                           │
┌───────▼──────────────────────┐   ┌───────────────▼──────────────────┐
│ src/cv/infer.py              │   │ src/cv/gradcam.py                │
│ run_inference()              │   │ run_gradcam()                    │
│ FR-CV-004, FR-DEP-010        │   │ FR-CV-010–016                    │
│ ≤ 1,500 ms (ONNX CPU)       │   │ ≤ 3,000 ms on CPU               │
└──────────────┬───────────────┘   └──────────────┬───────────────────┘
               │ (label, confidence)               │ (heatmap_path, band_pct)
               └────────────────┬──────────────────┘
                                │
                    ┌───────────▼──────────────────────┐
                    │ src/nlp/explain.py               │
                    │ generate_explanation()           │
                    │ FR-NLP-001–009                   │
                    │ ≤ 8,000 ms (Qwen 2.5 API)       │
                    └───────────┬──────────────────────┘
                                │ explanation: str
                                ▼
                           [app.py UI]
                           FR-DEP-001–009
```

---

## Contract 1 — Audio DSP: `src/audio/dsp.py`

### Primary Function

```python
def preprocess_audio(file_path: pathlib.Path) -> torch.Tensor:
    """
    Module: dsp
    SRS Reference: FR-AUD-001, FR-AUD-002, FR-AUD-003, FR-AUD-004,
                   FR-AUD-005, FR-AUD-006, FR-AUD-007, FR-AUD-008
    SDLC Phase: Phase 2 — System Design (contract); Phase 3 — Sprint A (implementation)
    Sprint: A
    Pipeline Stage: Audio DSP

    Interface Contract:
      Input:  pathlib.Path — absolute path to audio file (WAV or FLAC per FR-AUD-001)
              File must not be written to disk after processing (NFR-Security: in-memory)
      Output: torch.Tensor shape=[3, 224, 224], dtype=torch.float32 — FR-AUD-008

    Processing Steps (in order, all mandatory):
      1. Validate file format — accept WAV/FLAC; raise ValueError(AUD-002) otherwise [FR-AUD-001]
      2. Load audio via soundfile — returns raw samples and native sample rate [FR-AUD-001]
      3. Resample to 16,000 Hz using res_type='kaiser_best' [FR-AUD-002]
      4. Normalise amplitude to [-1.0, 1.0] peak range [FR-AUD-003]
      5. Reject clips < 0.5 s — raise ValueError(AUD-001) [FR-AUD-005]
      6. Pad (zero-right) or trim to exactly 32,000 samples (2.0 s × 16,000 Hz) [FR-AUD-004]
      7. Compute Mel spectrogram: n_mels=128, n_fft=2048, hop_length=512,
         window='hann', fmin=0.0, fmax=8000.0 [FR-AUD-006]
      8. Apply log-power dB conversion: librosa.power_to_db(ref=np.max) [FR-AUD-007]
      9. Resize to 224×224 using bilinear interpolation [FR-AUD-008]
      10. Replicate single channel to 3 channels → shape [3, 224, 224] [FR-AUD-008]
      11. Cast to torch.float32 [FR-AUD-008]

    Args:
        file_path (pathlib.Path): Path to audio file. Must be WAV or FLAC per
                                  config.yaml:audio.supported_formats [FR-AUD-001].
                                  File must exist and be readable.

    Returns:
        torch.Tensor: shape=[3, 224, 224], dtype=torch.float32.
                      Values normalised for ImageNet-pretrained EfficientNet-B4 input.
                      All three channels are identical (monochrome replicated).

    Raises:
        ValueError: f"AUD-001: clip duration {dur:.2f}s < {min_dur}s minimum"
                    if clip shorter than config.yaml:audio.min_duration_sec [FR-AUD-005]
        ValueError: f"AUD-002: unsupported format '{suffix}'. Accepted: {formats}"
                    if file format not in config.yaml:audio.supported_formats [FR-AUD-001]
        OSError: if file does not exist or cannot be read (re-raised from soundfile)

    Latency Target: ≤ 500 ms on CPU per NFR-Performance
    MCP Tools Used: context7-mcp (librosa)
    AI Generated: true
    Verified (V.E.R.I.F.Y.): false
    Author: Ferel / Safa
    Date: 2026-03-18
    """
```

### Optional Batch Function

```python
def preprocess_batch(
    file_paths: list[pathlib.Path]
) -> list[torch.Tensor]:
    """
    SRS Reference: FR-AUD-009 (SHOULD)
    Applies preprocess_audio() to each path in the list.
    Returns list of [3, 224, 224] float32 tensors.
    Preserves order. Raises on first error (fail-fast).
    Latency: ≤ 500 ms × len(file_paths)
    """
```

---

## Contract 2 — CV Inference: `src/cv/infer.py`

```python
def run_inference(tensor: torch.Tensor) -> tuple[str, float]:
    """
    Module: infer
    SRS Reference: FR-CV-004, FR-DEP-010
    SDLC Phase: Phase 2 — contract; Phase 3 — Sprint B (implementation)
    Sprint: B
    Pipeline Stage: CV Inference

    Interface Contract:
      Input:  torch.Tensor shape=[3, 224, 224], dtype=torch.float32
              Validated at entry — raise ValueError if shape or dtype mismatch
      Output: tuple[str, float]
              - label: "bonafide" or "spoof" (decision_threshold=0.5 from config.yaml)
              - confidence: float in (0.0, 1.0) — sigmoid output of final logit

    Implementation Notes:
      - ONNX Runtime session MUST be created ONCE at module import time,
        not per call. Session singleton pattern required. [FR-DEP-010, .cursorrules]
      - Input tensor is converted to numpy [1, 3, 224, 224] for ONNX Runtime.
      - Output: ONNX returns raw logit → apply sigmoid → compare to threshold.
      - Execution provider: CPUExecutionProvider only (HF Spaces CPU-only). [FR-DEP-010]

    Args:
        tensor (torch.Tensor): Preprocessed audio spectrogram image.
                               shape MUST equal config.yaml:audio.output_tensor_shape
                               = [3, 224, 224]. dtype MUST be torch.float32.

    Returns:
        tuple[str, float]: (label, confidence) where:
            label: "bonafide" if confidence >= 0.5 else "spoof" [FR-CV-004]
            confidence: sigmoid(logit) ∈ (0.0, 1.0)

    Raises:
        ValueError: "CV-001: expected tensor shape [3,224,224] float32, got {shape} {dtype}"
                    if input contract is violated
        RuntimeError: if ONNX Runtime session is not initialised at module load time

    Latency Target: ≤ 1,500 ms on CPU via ONNX Runtime [FR-DEP-010, NFR-Performance]
    ONNX Equivalence: |ONNX output − PyTorch output| < 1e-5 [FR-DEP-010]
    MCP Tools Used: context7-mcp (ONNX Runtime)
    AI Generated: true
    Verified (V.E.R.I.F.Y.): false
    Author: Ferel / Safa
    Date: 2026-03-18
    """
```

---

## Contract 3 — XAI Grad-CAM: `src/cv/gradcam.py`

```python
def run_gradcam(
    tensor: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[pathlib.Path, dict[str, float]]:
    """
    Module: gradcam
    SRS Reference: FR-CV-010, FR-CV-011, FR-CV-012, FR-CV-013, FR-CV-014, FR-CV-015
    SDLC Phase: Phase 2 — contract; Phase 3 — Sprint C (implementation)
    Sprint: C
    Pipeline Stage: XAI

    Interface Contract:
      Input:
        tensor: torch.Tensor shape=[3, 224, 224], dtype=torch.float32
                Identical tensor as passed to run_inference() — no re-load from disk
        model:  torch.nn.Module — trained EfficientNet-B4 PyTorch model (NOT ONNX session)
                Grad-CAM requires a PyTorch model for gradient hooks.
                model.eval() MUST be called before passing to this function.
      Output:
        tuple[pathlib.Path, dict[str, float]]
        - heatmap_path: pathlib.Path — path to generated PNG file (224×224, jet colormap,
                        α=0.5 overlay blending) [FR-CV-012]
        - band_attributions: dict[str, float] with exactly 4 keys:
                             {"low": %, "low_mid": %, "high_mid": %, "high": %}
                             Values are Softmax-normalised percentages summing to 100.0
                             [FR-CV-013, FR-CV-014]

    Grad-CAM Target Layer: model.features[7][-1]  [FR-CV-010, ADR-0004]
    Mel Band Mapping: librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0) [ADR-0005]

    Args:
        tensor (torch.Tensor): Preprocessed spectrogram. shape=[3, 224, 224], float32.
        model (torch.nn.Module): EfficientNet-B4 in eval mode with loaded weights.
                                 Must have model.features[7][-1] as a valid sub-module.

    Returns:
        tuple[pathlib.Path, dict[str, float]]:
            heatmap_path: Path to 224×224 PNG. Stored in BytesIO then written to
                          a temp path. Zero disk persistence of audio data (NFR-Security
                          applies to audio — heatmap PNG output is permitted).
            band_attributions: dict with keys "low", "low_mid", "high_mid", "high".
                               Each value: float, softmax percentage, sums to 100.0.
                               Example: {"low": 12.5, "low_mid": 43.2, "high_mid": 31.1, "high": 13.2}

    Raises:
        ValueError: "CV-002: Grad-CAM saliency map is all-zero — check target layer"
                    if grayscale_cam.max() == 0 (indicates bad target layer config)
        ValueError: "CV-001: expected tensor shape [3,224,224] float32" if input violated

    Latency Target: ≤ 3,000 ms on CPU [FR-CV-015, NFR-Performance]
    MCP Tools Used: context7-mcp (pytorch-grad-cam, librosa)
    AI Generated: true
    Verified (V.E.R.I.F.Y.): false
    Author: Ferel / Safa
    Date: 2026-03-18
    """
```

### Optional Developer Endpoint

```python
def get_raw_saliency(
    tensor: torch.Tensor,
    model: torch.nn.Module,
) -> dict:
    """
    SRS Reference: FR-CV-016 (SHOULD)
    Returns raw saliency map as JSON-serialisable dict.
    Output: {"saliency": [[float, ...], ...], "shape": [224, 224]}
    Used by developer tooling only — not exposed to end users.
    """
```

---

## Contract 4 — NLP Explanation: `src/nlp/explain.py`

```python
async def generate_explanation(
    label: str,
    confidence: float,
    band_pct: dict[str, float],
) -> str:
    """
    Module: explain
    SRS Reference: FR-NLP-001, FR-NLP-002, FR-NLP-003, FR-NLP-004,
                   FR-NLP-005, FR-NLP-006, FR-NLP-007, FR-NLP-008
    SDLC Phase: Phase 2 — contract; Phase 3 — Sprint D (implementation)
    Sprint: D
    Pipeline Stage: NLP

    Interface Contract:
      Input:
        label:      str — "bonafide" or "spoof" (from run_inference output)
        confidence: float ∈ (0.0, 1.0) — sigmoid confidence score
        band_pct:   dict[str, float] — 4-band attribution from run_gradcam
                    Keys: "low", "low_mid", "high_mid", "high"
                    Values sum to 100.0 (Softmax-normalised)
      Output:
        str — English explanation paragraph, 3–5 sentences [FR-NLP-001, FR-NLP-004]
              If API fails: rule-based template string (always non-empty) [FR-NLP-003]

    Fallback Chain (FR-NLP-002, FR-NLP-007, FR-NLP-003):
      1. Qwen 2.5 via async API (timeout = 30 s) [FR-NLP-002]
      2. Gemma-3 secondary fallback if Qwen times out or errors [FR-NLP-007, SHOULD]
      3. Rule-based template — zero external dependencies, always available [FR-NLP-003]

    Security Constraint:
      - API key MUST be read from os.environ[config.yaml:nlp.api_key_env_var] [FR-NLP-005]
      - NEVER log the key value — structured log must show key name only
      - If key is absent → skip API tier 1 and tier 2, use rule-based fallback [FR-NLP-003]

    Caching (FR-NLP-008 — SHOULD):
      - Cache key: (label, confidence_bucket, top_band)
      - confidence_bucket: discretised from config.yaml:nlp.caching.confidence_buckets
      - top_band: argmax of band_pct dict
      - Cache is in-memory dict (reset on app restart) — not persisted to disk

    Args:
        label (str): Prediction label. Must be "bonafide" or "spoof".
        confidence (float): Sigmoid output ∈ (0.0, 1.0). Passed verbatim to prompt.
        band_pct (dict[str, float]): Band attribution scores summing to 100.0.
                                     All four keys must be present.

    Returns:
        str: English explanation, 3–5 sentences. Non-empty guarantee — rule-based
             fallback ensures return value is always a valid string.

    Raises:
        ValueError: if label not in {"bonafide", "spoof"}
        ValueError: if band_pct does not contain all 4 required keys

    Latency Target:
        ≤ 8,000 ms (Qwen 2.5 API time-to-first-byte) [NFR-Performance]
        ≤ 100 ms (rule-based fallback, no API call)
    MCP Tools Used: stitch-mcp (Qwen 2.5 async), context7-mcp (openai client)
    AI Generated: true
    Verified (V.E.R.I.F.Y.): false
    Author: Ferel / Safa
    Date: 2026-03-18
    """
```

### Rule-Based Fallback Template (FR-NLP-003)

```python
_RULE_BASED_TEMPLATES: dict[str, str] = {
    "spoof": (
        "The audio sample has been classified as SPOOF with {confidence:.0%} confidence. "
        "The analysis detected artificial or synthesised speech characteristics. "
        "The model's attention was most concentrated in the {top_band} frequency band "
        "({top_band_pct:.1f}%), which is commonly associated with synthesis artefacts. "
        "This result suggests the audio may have been generated by a text-to-speech "
        "or voice conversion system."
    ),
    "bonafide": (
        "The audio sample has been classified as BONAFIDE with {confidence:.0%} confidence. "
        "The analysis identified natural human speech characteristics. "
        "The model's attention was most concentrated in the {top_band} frequency band "
        "({top_band_pct:.1f}%), consistent with natural vocal tract resonances. "
        "No significant artefacts of voice synthesis or conversion were detected."
    ),
}
```

---

## Contract 5 — Deployment UI: `app.py`

```python
# app.py is a wiring-only module — no pipeline logic permitted directly.
# All pipeline calls go through the four contracted functions above.
# Module boundary rule: app.py → UI wiring only [.cursorrules]

def build_demo() -> gr.Blocks:
    """
    Module: app (wiring only)
    SRS Reference: FR-DEP-001, FR-DEP-002, FR-DEP-003, FR-DEP-004,
                   FR-DEP-005, FR-DEP-006, FR-DEP-007, FR-DEP-008, FR-DEP-009
    SDLC Phase: Phase 2 — contract; Phase 3 — Sprint E (implementation)
    Sprint: E
    Pipeline Stage: Deployment

    Interface Contract:
      Input:  None (called once at startup)
      Output: gr.Blocks — Gradio application object

    UI Components (all required for SHALL FRs):
      - gr.Audio           → file upload widget, max 20 MB [FR-DEP-002]
      - gr.Textbox (×2)    → prediction label + AI explanation [FR-DEP-003]
      - gr.Number          → confidence score
      - gr.Image           → Grad-CAM heatmap PNG [FR-DEP-004]
      - gr.BarPlot         → 4-band frequency attribution chart [FR-DEP-005]
      - gr.HTML            → AI-unavailable warning badge (conditional) [FR-DEP-003]

    Pipeline Wiring:
      inference_fn: generator that yields two times [FR-DEP-006]:
        yield 1 → (label, confidence, heatmap_path, band_pct, "Generating explanation...")
        yield 2 → (label, confidence, heatmap_path, band_pct, explanation_text)

    Returns:
        gr.Blocks: Configured Gradio application.
                   Call .launch(share=False) in __main__ block.
    """
```

---

## Cross-Contract Invariants (Enforced at All Sprint Gates)

| Invariant | Description | Enforced By |
|-----------|-------------|-------------|
| **Tensor shape contract** | `[3, 224, 224]` float32 passed between dsp→infer, dsp→gradcam | Validation at infer.py and gradcam.py entry |
| **Label domain** | `label ∈ {"bonafide", "spoof"}` | Validated at explain.py entry |
| **Band sum invariant** | `sum(band_pct.values()) == 100.0 ± 0.01` | Validated at explain.py entry; asserted in test_cv.py |
| **No disk audio persistence** | Audio bytes stay in BytesIO throughout dsp.py | NFR-Security; verified in test_audio.py |
| **ONNX session singleton** | Created once at infer.py module import | Module-level session var; verified in test_cv.py |
| **API key never logged** | QWEN_API_KEY value never appears in structured log | FR-NLP-005; verified in test_nlp.py |

---

## Interface Change Protocol

Any proposed change to a function signature above requires:
1. Update this document with the new signature and rationale
2. Update the corresponding FR entry in `docs/adr/phase1-rtm.md`
3. Update `config.yaml` if any config key is affected
4. Update `session-cheatsheet.md` with the change record
5. Gate Check approval before Sprint implementation proceeds

---

*[DRAFT — Phase 2 — Pending V.E.R.I.F.Y.]*
