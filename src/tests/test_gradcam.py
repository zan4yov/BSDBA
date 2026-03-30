"""Grad-CAM / XAI tests (FR-CV-010–016, V.E.R.I.F.Y. L3)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
import yaml

from src.cv.gradcam import (
    compute_band_attributions,
    compute_gradcam,
    create_heatmap_overlay,
    get_mel_band_row_indices,
    get_raw_saliency_json,
    get_target_layer,
    run_gradcam,
)
from src.cv.model import DSDBAModel


def _load_cfg() -> dict:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "config.yaml").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
    return _load_cfg()


@pytest.fixture(scope="module")
def model(cfg: dict) -> DSDBAModel:
    m = DSDBAModel(cfg=cfg, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sample_tensor(cfg: dict) -> torch.Tensor:
    shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
    return torch.randn(*shape, dtype=torch.float32)


def test_target_layer_exists(cfg: dict, model: DSDBAModel) -> None:
    layer = get_target_layer(model, cfg)
    assert isinstance(layer, torch.nn.Module)


def test_saliency_shape(cfg: dict, model: DSDBAModel, sample_tensor: torch.Tensor) -> None:
    sal = compute_gradcam(model, sample_tensor, cfg)
    assert sal.shape == (224, 224)


def test_saliency_range(cfg: dict, model: DSDBAModel, sample_tensor: torch.Tensor) -> None:
    sal = compute_gradcam(model, sample_tensor, cfg)
    assert float(sal.min()) >= 0.0
    assert float(sal.max()) <= 1.0


def test_heatmap_png_created(
    cfg: dict,
    model: DSDBAModel,
    sample_tensor: torch.Tensor,
    tmp_path: Path,
) -> None:
    cfg_local = dict(cfg)
    cfg_local["gradcam"] = dict(cfg["gradcam"])
    cfg_local["gradcam"]["heatmap_output_dir"] = str(tmp_path.resolve())
    sal = compute_gradcam(model, sample_tensor, cfg_local)
    out = create_heatmap_overlay(sample_tensor, sal, cfg_local)
    assert out.exists()
    assert out.suffix.lower() == ".png"


def test_mel_band_mapping_not_linear(cfg: dict) -> None:
    actual = get_mel_band_row_indices(cfg)
    naive = {
        "low": (0, 31),
        "low_mid": (32, 63),
        "high_mid": (64, 95),
        "high": (96, 127),
    }
    assert actual != naive


def test_band_sum_100(cfg: dict, model: DSDBAModel, sample_tensor: torch.Tensor) -> None:
    sal = compute_gradcam(model, sample_tensor, cfg)
    bands = compute_band_attributions(sal, cfg)
    s = sum(bands.values())
    assert abs(s - 100.0) <= 0.001


def test_gradcam_latency(cfg: dict, model: DSDBAModel, sample_tensor: torch.Tensor) -> None:
    for _ in range(2):
        run_gradcam(sample_tensor, model, cfg)
    t0 = time.perf_counter()
    run_gradcam(sample_tensor, model, cfg)
    ms = (time.perf_counter() - t0) * 1000.0
    assert ms <= float(cfg["gradcam"]["latency_target_ms"])


def test_raw_saliency_json_serialisable(cfg: dict, model: DSDBAModel, sample_tensor: torch.Tensor) -> None:
    sal = compute_gradcam(model, sample_tensor, cfg)
    payload = get_raw_saliency_json(sal)
    json.dumps(payload)
