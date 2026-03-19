"""
Module: model
SRS Reference: FR-CV-001, FR-CV-002
SDLC Phase: Phase 3 — Sprint B (implementation begins Chain 06)
Sprint: B
Pipeline Stage: CV Inference
Interface Contract:
  Input:  N/A — module exposes build_model() factory function
  Output: torch.nn.Module — EfficientNet-B4 with custom 2-class linear head
Latency Target: N/A (model is built once; inference is in infer.py)
Open Questions Resolved: Q4 — Grad-CAM target confirmed as model.features[7][-1]
                                [ADR-0004]; Q3 — VRAM feasible at batch_size=16 [ADR-0008]
Open Questions Blocking: None
MCP Tools Used: context7-mcp (torchvision EfficientNet-B4)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 3 — Sprint B — Pending V.E.R.I.F.Y.]
# Implementation: Chain 06 — Sprint B

from __future__ import annotations

import torch
import torch.nn as nn


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B4 with a custom linear classification head.

    Architecture:
        - Backbone: torchvision EfficientNet-B4 (ImageNet-1k pretrained) [FR-CV-001]
        - Head: replaces classifier[-1] with nn.Linear(1792, num_classes) [FR-CV-002]
        - Grad-CAM target: model.features[7][-1] (Stage 7 final MBConv) [ADR-0004]

    All hyperparameters sourced from config.yaml:model — no magic numbers [R.E.F.A.C.T.].

    Args:
        num_classes (int): Output classes. Must equal config.yaml:model.num_classes (2).
                           bonafide=0, spoof=1 [FR-CV-002].
        pretrained (bool): If True, loads ImageNet-1k weights per
                           config.yaml:model.pretrained_weights [FR-CV-001].

    Returns:
        nn.Module: EfficientNet-B4 with modified classification head.
                   model.features[7][-1] is the Grad-CAM target layer [ADR-0004].
                   Call model.eval() before inference; model.train() for training.

    Raises:
        ValueError: if num_classes < 1

    Latency Target: N/A (model construction is a one-time cost)
    """
    raise NotImplementedError(
        "build_model() not implemented — stub only. "
        "Implement in Chain 06 (Sprint B)."
    )
