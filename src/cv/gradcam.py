"""
Module: gradcam
SRS Reference: FR-CV-010, FR-CV-011, FR-CV-012, FR-CV-013, FR-CV-014,
               FR-CV-015, FR-CV-016
SDLC Phase: Phase 3 — Sprint C (implementation begins Chain 07)
Sprint: C
Pipeline Stage: XAI
Interface Contract:
  Input:  torch.Tensor shape=[3, 224, 224], dtype=torch.float32
          torch.nn.Module — trained EfficientNet-B4 in eval() mode
  Output: tuple[pathlib.Path, dict[str, float]]
          - heatmap_path: PNG file (224×224, jet colormap, α=0.5 overlay) [FR-CV-012]
          - band_attributions: {"low": %, "low_mid": %, "high_mid": %, "high": %}
                               Softmax-normalised, sums to 100.0 [FR-CV-013, FR-CV-014]
Latency Target: ≤ 3,000 ms on CPU per FR-CV-015
Open Questions Resolved: Q4 — target layer model.features[7][-1] (ADR-0004)
                          Q5 — Mel band mapping via librosa.mel_frequencies (ADR-0005)
Open Questions Blocking: None (smoke-test deferred to Sprint C setup — TECH DEBT)
MCP Tools Used: context7-mcp (pytorch-grad-cam, librosa)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 3 — Sprint C — Pending V.E.R.I.F.Y.]
# Implementation: Chain 07 — Sprint C
# TECH DEBT: Grad-CAM smoke-test (non-zero saliency) deferred to Sprint C setup [FR-CV-010]

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn


def run_gradcam(
    tensor: torch.Tensor,
    model: nn.Module,
) -> tuple[pathlib.Path, dict[str, float]]:
    """Compute Grad-CAM saliency map and 4-band frequency attribution.

    Grad-CAM target: model.features[7][-1] (Stage 7 final MBConv, named features.7.1)
    Mel band mapping: librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0) [ADR-0005]

    Pipeline:
        1. Run GradCAM with target_layers=[model.features[7][-1]] [FR-CV-010, ADR-0004]
        2. Generate jet-colormap overlay PNG (α=0.5) [FR-CV-012]
        3. Map 224 saliency rows → 128 Mel bins → Hz via librosa.mel_frequencies [ADR-0005]
        4. Compute mean saliency per band [FR-CV-013]
        5. Softmax-normalise 4 band scores → percentages summing to 100.0 [FR-CV-014]

    Args:
        tensor (torch.Tensor): Preprocessed spectrogram. shape=[3, 224, 224], float32.
                               Identical tensor as passed to run_inference() — no re-load.
        model (nn.Module): EfficientNet-B4 in eval() mode with loaded weights.
                           model.features[7][-1] MUST be a valid sub-module [ADR-0004].
                           Must be a PyTorch model (NOT ONNX) — Grad-CAM requires hooks.

    Returns:
        tuple[pathlib.Path, dict[str, float]]:
            heatmap_path: Path to 224×224 PNG. Written to temp path (audio stays in-memory).
            band_attributions: Keys "low", "low_mid", "high_mid", "high".
                               Values: softmax %, sum == 100.0 ± 0.01 [FR-CV-014].
                               Example: {"low": 12.5, "low_mid": 43.2,
                                         "high_mid": 31.1, "high": 13.2}

    Raises:
        DSDBAError(CV-002): if grayscale_cam.max() == 0 (bad target layer config)
        DSDBAError(CV-001): if tensor shape != [3, 224, 224] or dtype != float32

    Latency Target: ≤ 3,000 ms on CPU [FR-CV-015, NFR-Performance]
    """
    raise NotImplementedError(
        "run_gradcam() not implemented — stub only. "
        "Implement in Chain 07 (Sprint C)."
    )


def get_raw_saliency(
    tensor: torch.Tensor,
    model: nn.Module,
) -> dict[str, object]:
    """Return raw Grad-CAM saliency map as a JSON-serialisable dict.

    SRS Reference: FR-CV-016 (SHOULD)
    Used by developer tooling only — not exposed to end users.

    Args:
        tensor (torch.Tensor): shape=[3, 224, 224], float32.
        model (nn.Module): EfficientNet-B4 in eval() mode.

    Returns:
        dict: {"saliency": [[float, ...], ...], "shape": [224, 224]}
              Values in [0.0, 1.0].
    """
    raise NotImplementedError(
        "get_raw_saliency() not implemented — stub only. "
        "Implement in Chain 07 (Sprint C)."
    )
