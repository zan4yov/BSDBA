"""
Module: infer
SRS Reference: FR-CV-004, FR-DEP-010
SDLC Phase: Phase 4 — Sprint B (full implementation)
Sprint: B
Pipeline Stage: CV Inference
Interface Contract:
  Input:  torch.Tensor shape=[3, 224, 224], dtype=torch.float32
  Output: tuple[str, float] — (label: "bonafide"|"spoof", confidence: float ∈ (0,1))
Latency Target: ≤ 1,500 ms on CPU via ONNX Runtime per FR-DEP-010
Open Questions Resolved: None blocking this module
Open Questions Blocking: None
MCP Tools Used: context7-mcp (ONNX Runtime InferenceSession API verified)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 4 — Sprint B — Pending V.E.R.I.F.Y.]
# CRITICAL: ONNX session singleton — created once, reused per request [.cursorrules]

from __future__ import annotations

import pathlib
import time

import numpy as np
import onnxruntime as ort
import torch

from src.cv.model import DSDBAModel
from src.utils.config import DSDBAConfig, load_config
from src.utils.errors import DSDBAError, ErrorCode
from src.utils.logger import get_logger, log_info, log_warning

# ── Module-level singletons ───────────────────────────────────────────────────

_CONFIG_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "config.yaml"
)
_CFG: DSDBAConfig = load_config(_CONFIG_PATH)
_log = get_logger(__name__)

_onnx_session: ort.InferenceSession | None = None


# ── ONNX Export ───────────────────────────────────────────────────────────────


def export_to_onnx(
    model: DSDBAModel,
    output_path: pathlib.Path,
    cfg: DSDBAConfig | None = None,
) -> pathlib.Path:
    """Export a trained DSDBAModel to ONNX format.

    Uses torch.onnx.export with opset 17 (compatible with onnxruntime 1.16).
    The exported model accepts [1, 3, 224, 224] float32 input and returns
    [1, 2] float32 logits.

    Args:
        model (DSDBAModel): Trained model. Will be set to eval mode.
        output_path (pathlib.Path): Destination .onnx file path.
        cfg (DSDBAConfig | None): Config. Uses module default if None.

    Returns:
        pathlib.Path: The output_path after successful export.

    Raises:
        RuntimeError: if export fails.

    SRS: FR-DEP-010.
    """
    if cfg is None:
        cfg = _CFG

    model.eval()

    tensor_shape = cfg.audio.output_tensor_shape
    dummy_input = torch.randn(1, *tensor_shape, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    log_info(
        stage="CV Inference",
        message=f"ONNX model exported to {output_path.name}",
        srs_ref="FR-DEP-010",
        data={"path": str(output_path), "opset": 17},
    )

    return output_path


def verify_onnx_equivalence(
    model: DSDBAModel,
    onnx_path: pathlib.Path,
    cfg: DSDBAConfig | None = None,
) -> bool:
    """Verify ONNX output matches PyTorch output within tolerance.

    Runs the same random input through both the PyTorch model and the
    ONNX Runtime session, asserting |diff| < config tolerance (1e-5).

    Args:
        model (DSDBAModel): PyTorch model (must be on CPU, eval mode).
        onnx_path (pathlib.Path): Path to exported .onnx file.
        cfg (DSDBAConfig | None): Config for tolerance. Uses module default if None.

    Returns:
        bool: True if outputs match within tolerance.

    SRS: FR-DEP-010 — |ONNX − PyTorch| < 1e-5.
    """
    if cfg is None:
        cfg = _CFG

    model.cpu().eval()
    tensor_shape = cfg.audio.output_tensor_shape
    dummy = torch.randn(1, *tensor_shape, dtype=torch.float32)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=cfg.deployment.onnx_execution_providers,
    )
    ort_out = session.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    tolerance = cfg.deployment.onnx_equivalence_tolerance
    passed = max_diff < tolerance

    log_info(
        stage="CV Inference",
        message=f"ONNX equivalence {'PASSED' if passed else 'FAILED'}",
        srs_ref="FR-DEP-010",
        data={"max_diff": max_diff, "tolerance": tolerance},
    )

    return passed


# ── ONNX Session Management ──────────────────────────────────────────────────


def load_onnx_session(
    onnx_path: pathlib.Path,
    cfg: DSDBAConfig | None = None,
) -> ort.InferenceSession:
    """Create and cache the ONNX Runtime InferenceSession singleton.

    MUST be called once at application startup (e.g., in app.py). Subsequent
    calls to run_inference() use the cached session. [.cursorrules]

    Args:
        onnx_path (pathlib.Path): Path to the exported .onnx model file.
        cfg (DSDBAConfig | None): Config for execution providers.

    Returns:
        ort.InferenceSession: Initialised ONNX session.

    SRS: FR-DEP-010 — CPUExecutionProvider only.
    """
    global _onnx_session  # noqa: PLW0603

    if cfg is None:
        cfg = _CFG

    providers = cfg.deployment.onnx_execution_providers

    _onnx_session = ort.InferenceSession(
        str(onnx_path),
        providers=providers,
    )

    log_info(
        stage="CV Inference",
        message="ONNX session initialised (singleton)",
        srs_ref="FR-DEP-010",
        data={"path": str(onnx_path), "providers": providers},
    )

    return _onnx_session


def run_onnx_inference(
    session: ort.InferenceSession,
    tensor: torch.Tensor,
    cfg: DSDBAConfig | None = None,
) -> tuple[str, float]:
    """Run inference through an explicit ONNX session.

    Converts tensor to numpy, feeds through session, applies softmax,
    and returns (label, confidence).

    Args:
        session (ort.InferenceSession): Active ONNX session.
        tensor (torch.Tensor): Input [3, 224, 224] float32.
        cfg (DSDBAConfig | None): Config for threshold.

    Returns:
        tuple[str, float]: (label, confidence).
            label: "bonafide" if confidence >= threshold else "spoof".
            confidence: P(bonafide) via softmax ∈ (0.0, 1.0).

    Raises:
        DSDBAError(CV-001): if tensor shape or dtype is wrong.

    SRS: FR-CV-004, FR-DEP-010.
    """
    if cfg is None:
        cfg = _CFG

    expected_shape = torch.Size(cfg.audio.output_tensor_shape)
    if tensor.shape != expected_shape or tensor.dtype != torch.float32:
        raise DSDBAError(
            code=ErrorCode.CV_001,
            message=(
                f"Expected tensor shape {list(expected_shape)} float32, "
                f"got {list(tensor.shape)} {tensor.dtype}."
            ),
            stage="CV Inference",
        )

    np_input = tensor.unsqueeze(0).numpy()  # [1, 3, 224, 224]
    logits = session.run(None, {"input": np_input})[0]  # [1, 2]

    # Softmax to get class probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    confidence_bonafide = float(probs[0, 0])
    threshold = cfg.model.decision_threshold

    label = "bonafide" if confidence_bonafide >= threshold else "spoof"

    return label, confidence_bonafide


# ── Public API (ADR-0007 Contract) ───────────────────────────────────────────


def run_inference(tensor: torch.Tensor) -> tuple[str, float]:
    """Run ONNX Runtime inference on a preprocessed spectrogram tensor.

    This is the ADR-0007 contract function. Uses the module-level
    ONNX session singleton initialised by load_onnx_session().

    Args:
        tensor (torch.Tensor): Preprocessed Mel spectrogram image.
                               shape MUST equal [3, 224, 224].
                               dtype MUST be torch.float32.

    Returns:
        tuple[str, float]: (label, confidence) where:
            label: "bonafide" if confidence >= 0.5 else "spoof" [FR-CV-004]
            confidence: P(bonafide) via softmax ∈ (0.0, 1.0)

    Raises:
        DSDBAError(CV-001): if tensor shape/dtype mismatch.
        RuntimeError: if ONNX session not initialised.

    Latency Target: ≤ 1,500 ms on CPU [FR-DEP-010, NFR-Performance].
    """
    if _onnx_session is None:
        raise RuntimeError(
            "ONNX session not initialised. Call load_onnx_session() "
            "at application startup before run_inference()."
        )

    t_start = time.perf_counter()
    label, confidence = run_onnx_inference(_onnx_session, tensor, _CFG)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    log_info(
        stage="CV Inference",
        message="Inference complete",
        srs_ref="FR-CV-004",
        data={
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
        },
    )

    target_ms = _CFG.deployment.onnx_latency_target_ms
    if latency_ms > target_ms:
        log_warning(
            stage="CV Inference",
            message=f"Latency {latency_ms:.1f}ms exceeded target {target_ms}ms",
            srs_ref="NFR-Performance",
        )

    return label, confidence
