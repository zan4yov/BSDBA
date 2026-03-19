"""
Module: model
SRS Reference: FR-CV-001, FR-CV-002
SDLC Phase: Phase 4 — Sprint B (full implementation)
Sprint: B
Pipeline Stage: CV Inference
Interface Contract:
  Input:  N/A — module exposes DSDBAModel class + build_model() factory
  Output: torch.nn.Module — EfficientNet-B4 with custom 2-class linear head
Latency Target: N/A (model is built once; inference is in infer.py)
Open Questions Resolved: Q4 — Grad-CAM target confirmed as model.features[7][-1]
                                [ADR-0004]; Q3 — VRAM feasible at batch_size=16 [ADR-0008]
Open Questions Blocking: None
MCP Tools Used: context7-mcp (torchvision EfficientNet-B4 API verified)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 4 — Sprint B — Pending V.E.R.I.F.Y.]

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from src.utils.config import DSDBAConfig, load_config
from src.utils.logger import get_logger, log_info

# ── Module-level singletons ───────────────────────────────────────────────────

_CONFIG_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "config.yaml"
)
_CFG: DSDBAConfig = load_config(_CONFIG_PATH)
_log = get_logger(__name__)

# EfficientNet-B4 classifier input dimension (constant across all B4 variants)
_B4_CLASSIFIER_IN_FEATURES: int = 1792


class DSDBAModel(nn.Module):
    """EfficientNet-B4 with custom 2-class linear head for deepfake detection.

    Architecture [FR-CV-001, FR-CV-002]:
        - Backbone: torchvision EfficientNet-B4 (ImageNet-1k pretrained)
        - Head: nn.Sequential(Dropout, Linear(1792, num_classes))
        - Grad-CAM target: model.features[7][-1] (Stage 7 final MBConv) [ADR-0004]

    Supports gradient checkpointing for Colab T4 VRAM safety [Q3, ADR-0008].

    Args:
        num_classes (int): Output class count. Default 2 (bonafide=0, spoof=1).
        pretrained (bool): Load ImageNet-1k weights if True [FR-CV-001].
        gradient_checkpointing (bool): Enable activation checkpointing [Q3].

    Raises:
        ValueError: if num_classes < 1.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        self.features: nn.Sequential = backbone.features
        self.avgpool: nn.Module = backbone.avgpool

        dropout_rate: float = backbone.classifier[0].p
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(_B4_CLASSIFIER_IN_FEATURES, num_classes),
        )

        self._gradient_checkpointing = gradient_checkpointing
        self._num_classes = num_classes

    @property
    def gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is enabled."""
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        self._gradient_checkpointing = value

    def freeze_backbone(self) -> None:
        """Freeze all backbone feature layers; only classifier head trains.

        Used during Phase 1 training (frozen_epochs) [FR-CV-003].
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_top_n(self, n: int) -> None:
        """Unfreeze the top N stages of the backbone features for fine-tuning.

        EfficientNet-B4 has 9 stages (indices 0–8) in self.features.
        Example: n=3 unfreezes stages 6, 7, 8. [FR-CV-003]

        Args:
            n (int): Number of top stages to unfreeze. Range [0, 9].

        Raises:
            ValueError: if n is out of valid range.
        """
        total_stages: int = len(self.features)
        if n < 0 or n > total_stages:
            raise ValueError(
                f"n must be in [0, {total_stages}], got {n}"
            )
        self.freeze_backbone()
        for stage_idx in range(total_stages - n, total_stages):
            for param in self.features[stage_idx].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits [batch, num_classes].

        Sigmoid/softmax is NOT applied here — it belongs in the loss
        function (CrossEntropyLoss) and inference post-processing.

        Args:
            x (torch.Tensor): Input tensor [batch, 3, 224, 224] float32.

        Returns:
            torch.Tensor: Raw logits [batch, num_classes].
        """
        if self._gradient_checkpointing and self.training:
            x = checkpoint_sequential(
                self.features,
                len(self.features),
                x,
                use_reentrant=False,
            )
        else:
            x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(
    num_classes: int | None = None,
    pretrained: bool = True,
) -> DSDBAModel:
    """Build DSDBAModel from config.yaml parameters.

    Factory function per Phase 2 interface contract (ADR-0007).

    Args:
        num_classes (int | None): Override config.yaml:model.num_classes.
                                  Default None uses config value (2).
        pretrained (bool): Load ImageNet-1k weights [FR-CV-001].

    Returns:
        DSDBAModel: Initialised model ready for training or inference.

    Latency Target: N/A (model construction is a one-time cost).
    """
    cfg_model = _CFG.model
    cfg_train = _CFG.training

    n_classes: int = num_classes if num_classes is not None else cfg_model.num_classes

    model = DSDBAModel(
        num_classes=n_classes,
        pretrained=pretrained,
        gradient_checkpointing=cfg_train.gradient_checkpointing,
    )

    log_info(
        stage="CV Model",
        message="Model built",
        srs_ref="FR-CV-001",
        data={
            "backbone": cfg_model.backbone,
            "num_classes": n_classes,
            "pretrained": pretrained,
            "gradient_checkpointing": cfg_train.gradient_checkpointing,
            "params_total": sum(p.numel() for p in model.parameters()),
        },
    )

    return model
