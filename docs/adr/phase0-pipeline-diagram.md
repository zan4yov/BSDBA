# DSDBA — Phase 0 Pipeline Architecture Diagram
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 0 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** APPROVED — Phase 0 baseline

---

## Overview

The DSDBA pipeline is a strictly feed-forward, 5-stage sequential architecture. Each stage boundary is annotated with the exact data type and shape crossing the interface. No stage may call back to an upstream stage. The NLP stage (Stage 4) runs asynchronously and must not block Stage 2/3 results from displaying in the UI.

---

## Full Pipeline Diagram

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        U([User Upload])
    end

    subgraph STAGE1["Stage 1 — Audio DSP\n(src/audio/dsp.py)\nFR-AUD-001–011"]
        A1[Format Validation\nFR-AUD-001 / FR-AUD-011]
        A2[Resample → 16 kHz\nFR-AUD-002]
        A3[Normalise Amplitude\nFR-AUD-003]
        A4[Pad / Trim → 2.0 s\nFR-AUD-004 / FR-AUD-005]
        A5[Remove Silence\nFR-AUD-009]
        A6[Mel Spectrogram\n128 bins, n_fft=2048\nFR-AUD-006]
        A7[Log-Power dB Scale\nFR-AUD-007]
        A8[Resize → 224×224\nReplicate to 3 channels\nFR-AUD-008]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8
    end

    subgraph STAGE2["Stage 2 — CV Inference\n(src/cv/infer.py)\nFR-CV-001–009"]
        B1[Shape + dtype Guard\nFR-CV-001 / FR-AUD-008]
        B2[EfficientNet-B4 ONNX\nFR-CV-001 / FR-DEP-010]
        B3[Sigmoid Activation\nFR-CV-004]
        B4[Threshold → Label\nFR-CV-004\nthreshold=0.5]
        B1 --> B2 --> B3 --> B4
    end

    subgraph STAGE3["Stage 3 — XAI Grad-CAM\n(src/cv/gradcam.py)\nFR-CV-010–016"]
        C1[GradCAM\nmodel.features[-1]\nFR-CV-010 / FR-CV-011]
        C2[Overlay Heatmap PNG\njet colormap, α=0.5\nFR-CV-012]
        C3[Mel Bin → Hz Map\nFR-CV-013 / Q5]
        C4[Band Attribution\n4 bands, softmax norm\nFR-CV-013 / FR-CV-014]
        C1 --> C2
        C1 --> C3 --> C4
    end

    subgraph STAGE4["Stage 4 — NLP Explanation\n(src/nlp/explain.py)\nFR-NLP-001–009\n⚡ ASYNC — non-blocking"]
        D1[Build Prompt\nFR-NLP-001 / FR-NLP-004]
        D2{Qwen 2.5 API\nFR-NLP-002}
        D3{Gemma 3 Fallback\nFR-NLP-007}
        D4[Rule-Based Template\nFR-NLP-003]
        D5[Cache Result\nFR-NLP-008]
        D1 --> D2
        D2 -->|timeout / error| D3
        D3 -->|timeout / error| D4
        D2 -->|success| D5
        D3 -->|success| D5
        D4 --> D5
    end

    subgraph STAGE5["Stage 5 — Gradio UI\n(app.py)\nFR-DEP-001–010"]
        E1[Audio Upload Widget\nFR-DEP-002 / FR-DEP-008]
        E2[Stage 2 Result Panel\nLabel + Confidence\nFR-DEP-004 / FR-NLP-006]
        E3[Stage 3 Heatmap Panel\nFR-DEP-005]
        E4[Band Bar Chart\nFR-DEP-006]
        E5[Stage 4 NLP Panel\nFR-DEP-003]
        E6[About Section\nFR-DEP-009]
    end

    U -->|"WAV / FLAC (≤20 MB)\nFR-AUD-001, FR-DEP-002"| A1
    A8 -->|"torch.Tensor\n[3, 224, 224] float32\nFR-AUD-008"| B1
    B4 -->|"tuple: (label: str,\nconfidence: float)\nFR-CV-004"| E2
    B4 -->|"torch.Tensor\n[3, 224, 224] float32\n+ ONNX model ref"| C1
    C2 -->|"heatmap: PIL.Image\n(PNG, 224×224)\nFR-CV-012"| E3
    C4 -->|"band_pct: list[float, 4]\nsoftmax-normalised\nFR-CV-014"| E4
    C4 -->|"band_pct: list[float, 4]"| D1
    B4 -->|"label: str\nconfidence: float"| D1
    D5 -->|"explanation: str\n3–5 sentences, English\nFR-NLP-001"| E5
```

---

## Interface Contract Summary

| Boundary | From | To | Type | Shape / Format |
|----------|------|----|------|----------------|
| Input → Stage 1 | User | `dsp.py` | Audio file | WAV or FLAC, ≤ 20 MB |
| Stage 1 → Stage 2 | `dsp.py` | `infer.py` | `torch.Tensor` | `[3, 224, 224]` float32 |
| Stage 2 → Stage 3 | `infer.py` | `gradcam.py` | Tensor + model ref | `[3, 224, 224]` float32 |
| Stage 2 → UI | `infer.py` | `app.py` | `tuple` | `(label: str, confidence: float)` |
| Stage 3 → UI | `gradcam.py` | `app.py` | `PIL.Image` | PNG, 224 × 224 |
| Stage 3 → Stage 4 | `gradcam.py` | `explain.py` | `list[float]` | `band_pct[4]`, softmax-normalised |
| Stage 4 → UI | `explain.py` | `app.py` | `str` | English paragraph, 3–5 sentences |

---

## Key Design Constraints

| Constraint | Requirement | Source |
|------------|-------------|--------|
| Feed-forward only | No back-references between stages | SRS Section 0.1 |
| Async NLP | Stage 4 MUST NOT block Stage 2/3 display | FR-NLP-006 |
| ONNX mandatory | CV inference uses ONNX Runtime at deployment | FR-DEP-010 |
| Audio in-memory | No disk writes for audio data | NFR-Security |
| Tensor validated at CV entry | Shape + dtype assertion before forward pass | FR-AUD-008 / FR-CV-001 |
| Config-driven | All hyperparameters from config.yaml | NFR-Maintainability |

---

## Latency Budget

| Stage | Module | Target | NFR Reference |
|-------|--------|--------|---------------|
| Stage 1 | `dsp.py` | ≤ 500 ms | NFR-Performance |
| Stage 2 (ONNX) | `infer.py` | ≤ 1,500 ms | FR-DEP-010 |
| Stage 3 | `gradcam.py` | ≤ 3,000 ms | FR-CV-015 |
| Stage 4 | `explain.py` | ≤ 8,000 ms | NFR-Performance |
| **End-to-end** | **`app.py`** | **≤ 15,000 ms** | **FR-DEP-007** |

---

*[DRAFT — Phase 0 — Pending V.E.R.I.F.Y.]*
