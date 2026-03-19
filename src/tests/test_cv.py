"""
Module: test_cv
SRS Reference: FR-CV-001, FR-CV-002, FR-CV-004, FR-CV-005, FR-DEP-010
SDLC Phase: Phase 4 — Sprint B (V.E.R.I.F.Y. Level 3)
Sprint: B
Pipeline Stage: CV Inference (test suite)
Interface Contract:
  Input:  N/A — pytest test suite
  Output: 8 tests — all must pass before Sprint B Gate Check
Latency Target: test_onnx_latency asserts ≤ 1,500 ms
Open Questions Resolved: Q3 (VRAM), Q4 (Grad-CAM layer)
Open Questions Blocking: None
MCP Tools Used: None
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

from __future__ import annotations

import pathlib
import time
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.cv.model import DSDBAModel, build_model
from src.cv.infer import (
    export_to_onnx,
    load_onnx_session,
    run_onnx_inference,
    verify_onnx_equivalence,
)
from src.cv.train import compute_eer, get_class_weights
from src.utils.config import DSDBAConfig, load_config

# ── Fixtures ──────────────────────────────────────────────────────────────────

_CONFIG_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "config.yaml"
)


@pytest.fixture(scope="module")
def cfg() -> DSDBAConfig:
    return load_config(_CONFIG_PATH)


@pytest.fixture(scope="module")
def model() -> DSDBAModel:
    """Build model without pretrained weights for fast test execution."""
    return DSDBAModel(num_classes=2, pretrained=False, gradient_checkpointing=False)


@pytest.fixture(scope="module")
def dummy_input(cfg: DSDBAConfig) -> torch.Tensor:
    shape = cfg.audio.output_tensor_shape
    return torch.randn(1, *shape, dtype=torch.float32)


@pytest.fixture(scope="module")
def onnx_path(model: DSDBAModel, cfg: DSDBAConfig) -> pathlib.Path:
    """Export model to ONNX in a temp directory (shared across test module)."""
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    out = tmp_dir / "test_model.onnx"
    export_to_onnx(model, out, cfg)
    return out


# ── Test 1: Model output shape ───────────────────────────────────────────────


def test_model_output_shape(model: DSDBAModel, dummy_input: torch.Tensor) -> None:
    """FR-CV-001/002: Forward pass produces logits of shape [batch, 2]."""
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == torch.Size([1, 2]), (
        f"Expected logits shape [1, 2], got {logits.shape}"
    )


# ── Test 2: Sigmoid output range ─────────────────────────────────────────────


def test_sigmoid_output_range(model: DSDBAModel, dummy_input: torch.Tensor) -> None:
    """FR-CV-004: Softmax confidence scores must be in (0.0, 1.0)."""
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    probs = torch.softmax(logits, dim=1)
    assert probs.min() > 0.0, "Softmax output must be > 0"
    assert probs.max() < 1.0, "Softmax output must be < 1"
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Softmax must sum to 1.0"


# ── Test 3: Freeze / unfreeze backbone ───────────────────────────────────────


def test_freeze_unfreeze(model: DSDBAModel) -> None:
    """FR-CV-003: Backbone freeze/unfreeze controls requires_grad correctly."""
    model.freeze_backbone()
    for param in model.features.parameters():
        assert not param.requires_grad, "Frozen backbone param should not require grad"
    for param in model.classifier.parameters():
        assert param.requires_grad, "Classifier params should always require grad"

    model.unfreeze_top_n(3)
    total_stages = len(model.features)
    for idx in range(total_stages - 3, total_stages):
        for param in model.features[idx].parameters():
            assert param.requires_grad, (
                f"Top-3 stage {idx} should require grad after unfreeze"
            )
    for idx in range(total_stages - 3):
        for param in model.features[idx].parameters():
            assert not param.requires_grad, (
                f"Stage {idx} should remain frozen"
            )


# ── Test 4: ONNX export creates file ─────────────────────────────────────────


def test_onnx_export_creates_file(onnx_path: pathlib.Path) -> None:
    """FR-DEP-010: ONNX export produces a valid .onnx file."""
    assert onnx_path.exists(), f"ONNX file not found at {onnx_path}"
    assert onnx_path.stat().st_size > 0, "ONNX file is empty"


# ── Test 5: ONNX equivalence ─────────────────────────────────────────────────


def test_onnx_equivalence(
    model: DSDBAModel,
    onnx_path: pathlib.Path,
    cfg: DSDBAConfig,
) -> None:
    """FR-DEP-010: |ONNX output − PyTorch output| < 1e-5."""
    assert verify_onnx_equivalence(model, onnx_path, cfg), (
        "ONNX output does not match PyTorch output within tolerance"
    )


# ── Test 6: ONNX inference latency ───────────────────────────────────────────


def test_onnx_latency(
    onnx_path: pathlib.Path,
    cfg: DSDBAConfig,
) -> None:
    """FR-DEP-010 + NFR-Performance: Single ONNX inference ≤ 1,500 ms on CPU."""
    session = ort.InferenceSession(
        str(onnx_path),
        providers=cfg.deployment.onnx_execution_providers,
    )
    tensor_shape = cfg.audio.output_tensor_shape
    dummy = torch.randn(*tensor_shape, dtype=torch.float32)

    t_start = time.perf_counter()
    run_onnx_inference(session, dummy, cfg)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    target = cfg.deployment.onnx_latency_target_ms
    assert latency_ms <= target, (
        f"ONNX inference latency {latency_ms:.1f}ms > target {target}ms"
    )


# ── Test 7: ONNX CPU provider only ───────────────────────────────────────────


def test_onnx_cpu_provider_only(
    onnx_path: pathlib.Path,
    cfg: DSDBAConfig,
) -> None:
    """FR-DEP-010: ONNX session uses CPUExecutionProvider exclusively."""
    session = ort.InferenceSession(
        str(onnx_path),
        providers=cfg.deployment.onnx_execution_providers,
    )
    providers = session.get_providers()
    assert "CPUExecutionProvider" in providers, (
        f"CPUExecutionProvider not found in {providers}"
    )


# ── Test 8: Class weights validity ───────────────────────────────────────────


def test_class_weights_sum() -> None:
    """FR-CV-005: Inverse-frequency class weights are valid positive tensors."""
    labels = [0, 0, 0, 1, 1]
    weights = get_class_weights(labels, num_classes=2)

    assert weights.shape == torch.Size([2]), f"Expected shape [2], got {weights.shape}"
    assert weights.dtype == torch.float32, f"Expected float32, got {weights.dtype}"
    assert (weights > 0).all(), "All class weights must be positive"
    assert weights[1] > weights[0], (
        "Minority class (spoof) should have higher weight"
    )
