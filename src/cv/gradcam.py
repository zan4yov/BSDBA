"""
Module: src.cv.gradcam
SRS Reference: FR-CV-010–016
SDLC Phase: 4 — Sprint C
Pipeline Stage: XAI (Grad-CAM)
Interface Contract:
  Input: torch.Tensor [3, 224, 224] float32 + DSDBAModel + cfg dict
  Output: run_gradcam → (heatmap PNG Path, 4-band dict summing to 100.0); optional raw saliency JSON
Latency Target: ≤ 3,000 ms on CPU per FR-CV-015 / NFR-Performance
Open Questions Resolved: Q4 (layer path), Q5 (mel_frequencies band mapping)
Open Questions Blocking: None
MCP Tools Used: context7-mcp (pytorch-grad-cam GradCAM API)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-29
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn

from src.cv.model import DSDBAModel
from src.utils.logger import log_info


def get_target_layer(model: DSDBAModel, cfg: dict[str, Any]) -> nn.Module:
    """
    Resolve Grad-CAM target layer from config path (FR-CV-010, FR-CV-011).

    Args:
        model: DSDBAModel instance; path is evaluated as Python against the name `model`.
        cfg: Full configuration; uses `cfg["gradcam"]["target_layer"]`.

    Returns:
        The `nn.Module` at the configured path (Q4: backbone `features[8]`).

    Raises:
        ValueError: CV-010 when the path is missing, unsafe, or does not resolve to a module.
    """

    path = str(cfg["gradcam"]["target_layer"]).strip()
    if not path:
        raise ValueError("CV-010: gradcam.target_layer is empty per FR-CV-010")
    try:
        layer = eval(path, {"__builtins__": {}}, {"model": model})  # noqa: S307 — ADR-locked path only
    except Exception as exc:
        raise ValueError("CV-010: gradcam.target_layer eval failed per FR-CV-010") from exc
    if not isinstance(layer, nn.Module):
        raise ValueError("CV-010: gradcam.target_layer must resolve to nn.Module per FR-CV-010")
    return layer


def compute_gradcam(model: DSDBAModel, tensor: Tensor, cfg: dict[str, Any]) -> np.ndarray:
    """
    Compute Grad-CAM saliency (FR-CV-011) and return a normalised [224, 224] map.

    Args:
        model: DSDBAModel in eval mode for forward; gradients enabled internally by GradCAM.
        tensor: Input shaped [3, 224, 224] or [1, 3, 224, 224], float32.
        cfg: Full configuration.

    Returns:
        `np.ndarray` of shape `(224, 224)`, dtype float32, values in [0, 1].

    Raises:
        ValueError: CV-011 on unexpected CAM spatial dimensions after resize.
    """

    target_layer = get_target_layer(model, cfg)
    if tensor.ndim == 3:
        batch = tensor.unsqueeze(0)
    else:
        batch = tensor
    batch = batch.detach().to(dtype=torch.float32, device="cpu")
    target_class = int(cfg["gradcam"]["cam_target_class_index"])
    targets = [ClassifierOutputTarget(target_class)]
    model.eval()
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        cam_tensor = cam(input_tensor=batch, targets=targets)
    saliency = np.asarray(cam_tensor[0], dtype=np.float32)
    if saliency.ndim != 2:
        raise ValueError("CV-011: GradCAM output must be 2D per FR-CV-011")
    h, w = saliency.shape
    if (h, w) != (224, 224):
        t = torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0)
        saliency = (
            F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
            .astype(np.float32)
        )
    return _normalise_saliency_to_unit(saliency)


def _normalise_saliency_to_unit(saliency: np.ndarray) -> np.ndarray:
    """Min–max normalise saliency to [0, 1] (FR-CV-011)."""
    s = saliency.astype(np.float64, copy=False)
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    if s_max - s_min < 1e-12:
        return np.zeros((224, 224), dtype=np.float32)
    out = (s - s_min) / (s_max - s_min)
    return out.astype(np.float32)


def create_heatmap_overlay(tensor: Tensor, saliency: np.ndarray, cfg: dict[str, Any]) -> Path:
    """
    Apply jet colormap and alpha blend (FR-CV-012), save PNG, return path.

    Args:
        tensor: Input image tensor [3, 224, 224] float32.
        saliency: Array [224, 224] in [0, 1].
        cfg: Full configuration (`gradcam.overlay_alpha`, `gradcam.colormap`, `heatmap_output_dir`).

    Returns:
        Path to written PNG file.
    """

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    alpha = float(cfg["gradcam"]["overlay_alpha"])
    cmap_name = str(cfg["gradcam"]["colormap"])
    rel = Path(str(cfg["gradcam"]["heatmap_output_dir"]))
    root = Path(__file__).resolve().parents[2]
    out_dir = rel.resolve() if rel.is_absolute() else (root / rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gradcam_{uuid.uuid4().hex}.png"

    base = _tensor_to_hwc_rgb(tensor)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(np.asarray(saliency, dtype=np.float64))[..., :3].astype(np.float32)
    blended = alpha * heat_rgb + (1.0 - alpha) * base
    blended = np.clip(blended, 0.0, 1.0)
    img = Image.fromarray((blended * 255.0).round().astype(np.uint8))
    img.save(out_path, format="PNG")
    return out_path


def _tensor_to_hwc_rgb(tensor: Tensor) -> np.ndarray:
    """Convert [3,H,W] tensor to HWC float RGB in [0, 1] for overlay."""
    x = tensor.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError("CV-012: tensor must be [3, H, W] per FR-CV-012")
    hwc = x.permute(1, 2, 0).numpy()
    vmin = float(np.min(hwc))
    vmax = float(np.max(hwc))
    if vmax - vmin < 1e-12:
        return np.zeros_like(hwc, dtype=np.float32)
    return ((hwc - vmin) / (vmax - vmin)).astype(np.float32)


def get_mel_band_row_indices(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """
    Map Hz band boundaries to inclusive Mel bin index ranges (FR-CV-013, Q5).

    Uses `librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)` — not linear 128/4 splits.

    Args:
        cfg: Full configuration; uses `audio.n_mels`, `gradcam.band_hz`.

    Returns:
        Keys `low`, `low_mid`, `high_mid`, `high`; values are `(min_bin, max_bin)` inclusive in [0, 127].
    """

    n_mels = int(cfg["audio"]["n_mels"])
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=8000.0)
    band_hz = cfg["gradcam"]["band_hz"]
    order = ("low", "low_mid", "high_mid", "high")
    out: dict[str, tuple[int, int]] = {}
    for name in order:
        lo_hz, hi_hz = (float(band_hz[name][0]), float(band_hz[name][1]))
        if name == "high":
            idx = np.where((mel_freqs >= lo_hz) & (mel_freqs <= hi_hz))[0]
        else:
            idx = np.where((mel_freqs >= lo_hz) & (mel_freqs < hi_hz))[0]
        if idx.size == 0:
            raise ValueError(f"CV-013: no mel bins for band {name} per FR-CV-013")
        out[name] = (int(idx.min()), int(idx.max()))
    return out


def _row_to_mel_bin(row: int, height: int, n_mels: int) -> int:
    """Map a saliency row (resized spectrogram height) to a mel bin index."""
    return int(min(n_mels - 1, (row * n_mels) // height))


def compute_band_attributions(saliency: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    """
    Sum saliency per frequency band and Softmax-normalise to 100% (FR-CV-013, FR-CV-014).

    Args:
        saliency: [224, 224] saliency map.
        cfg: Full configuration.

    Returns:
        Four keys with values summing to 100.0 (within floating tolerance).
    """

    height = int(saliency.shape[0])
    n_mels = int(cfg["audio"]["n_mels"])
    band_ranges = get_mel_band_row_indices(cfg)
    row_sum = np.sum(saliency.astype(np.float64), axis=1)
    order = ("low", "low_mid", "high_mid", "high")
    raw = []
    for name in order:
        lo_b, hi_b = band_ranges[name]
        total = 0.0
        for r in range(height):
            m = _row_to_mel_bin(r, height, n_mels)
            if lo_b <= m <= hi_b:
                total += float(row_sum[r])
        raw.append(total)
    t = torch.tensor(raw, dtype=torch.float32)
    sm = F.softmax(t, dim=0)
    pct = (sm * 100.0).tolist()
    result = {name: float(pct[i]) for i, name in enumerate(order)}
    s = sum(result.values())
    if abs(s - 100.0) > 0.001:
        raise ValueError("CV-014: band softmax must sum to 100.0 per FR-CV-014")
    return result


def run_gradcam(tensor: Tensor, model: DSDBAModel, cfg: dict[str, Any]) -> tuple[Path, dict[str, float]]:
    """
    Full Grad-CAM pipeline: saliency, heatmap PNG, band percentages (FR-CV-010–015).

    Args:
        tensor: [3, 224, 224] float32.
        model: DSDBAModel instance.
        cfg: Full configuration.

    Returns:
        `(heatmap_png_path, band_attributions)` per phase2-interface-contracts.md.
    """

    start = time.perf_counter()
    saliency = compute_gradcam(model, tensor, cfg)
    heatmap_path = create_heatmap_overlay(tensor, saliency, cfg)
    bands = compute_band_attributions(saliency, cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log_info(
        stage="gradcam",
        message="run_gradcam_complete",
        data={"latency_ms": round(elapsed_ms, 3), "heatmap": str(heatmap_path)},
    )
    return heatmap_path, bands


def get_raw_saliency_json(saliency: np.ndarray) -> dict[str, Any]:
    """
    Serialise saliency matrix for JSON (FR-CV-016 SHOULD).

    Args:
        saliency: Numpy array of shape `[224, 224]`.

    Returns:
        JSON-serialisable dict including nested list `saliency`.
    """

    return {
        "shape": list(saliency.shape),
        "dtype": str(saliency.dtype),
        "saliency": np.asarray(saliency, dtype=np.float32).tolist(),
    }
