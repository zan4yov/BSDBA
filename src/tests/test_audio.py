"""
Module: test_audio
SRS Reference: FR-AUD-001–011, NFR-Performance
SDLC Phase: Phase 4 — Sprint A (test implementation)
Sprint: A
Pipeline Stage: Audio DSP
Interface Contract:
  Input:  Synthetic WAV files generated via soundfile + numpy
  Output: Assertions on DSDBAError codes, tensor shape/dtype, latency
Latency Target: preprocess_audio ≤ 500 ms per test_latency [NFR-Performance]
Open Questions Resolved: N/A
Open Questions Blocking: None
MCP Tools Used: None
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 4 — Sprint A — Pending V.E.R.I.F.Y.]

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio.dsp import (
    preprocess_audio,
    preprocess_batch,
    fix_duration,
    to_mono,
)
from src.utils.errors import DSDBAError, ErrorCode

# ── Constants (all from config.yaml values — no magic numbers) ────────────────

_SAMPLE_RATE = 16_000          # config.yaml:audio.sample_rate [FR-AUD-002]
_N_SAMPLES = 32_000            # config.yaml:audio.n_samples   [FR-AUD-004]
_MIN_DUR = 0.5                 # config.yaml:audio.min_duration_sec [FR-AUD-005]
_OUTPUT_SHAPE = (3, 224, 224)  # config.yaml:audio.output_tensor_shape [FR-AUD-008]
_LATENCY_BUDGET_MS = 500       # config.yaml:performance_targets.audio_dsp_ms
_LATENCY_BUDGET_S = _LATENCY_BUDGET_MS / 1000.0


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _write_wav(
    path: pathlib.Path,
    duration_sec: float,
    sample_rate: int = _SAMPLE_RATE,
    n_channels: int = 1,
    amplitude: float = 0.5,
) -> pathlib.Path:
    """Write a synthetic sine-wave WAV file to *path* and return *path*.

    Using a sine wave avoids a flat spectrogram (all-zero after normalisation)
    which would break the Mel extraction assertions.

    Args:
        path (pathlib.Path): Destination file path (must end in .wav).
        duration_sec (float): Length of the clip in seconds.
        sample_rate (int): Sample rate in Hz. Defaults to 16,000.
        n_channels (int): 1 = mono, 2 = stereo. Defaults to 1.
        amplitude (float): Sine wave amplitude in (0, 1). Defaults to 0.5.

    Returns:
        pathlib.Path: The same path that was written.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    wave = (amplitude * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    if n_channels == 2:
        data = np.stack([wave, wave * 0.8], axis=1)  # (n_samples, 2)
    else:
        data = wave  # (n_samples,)

    sf.write(str(path), data, sample_rate)
    return path


# ── Test: too-short audio → DSDBAError(AUD-001) ───────────────────────────────


def test_too_short_audio(tmp_path: pathlib.Path) -> None:
    """A 0.4 s clip must raise DSDBAError with code AUD-001. [FR-AUD-005]"""
    wav = _write_wav(tmp_path / "short.wav", duration_sec=0.4)

    with pytest.raises(DSDBAError) as exc_info:
        preprocess_audio(wav)

    error: DSDBAError = exc_info.value
    assert error.code == ErrorCode.AUD_001, (
        f"Expected AUD-001, got {error.code}"
    )
    assert error.stage == "Audio DSP"


# ── Test: exact 2.0 s clip → no crop, no pad, correct shape ──────────────────


def test_exact_duration(tmp_path: pathlib.Path) -> None:
    """A 2.0 s clip must produce [3, 224, 224] float32 without crop or pad. [FR-AUD-004]"""
    wav = _write_wav(tmp_path / "exact.wav", duration_sec=2.0)
    tensor = preprocess_audio(wav)

    assert tensor.shape == torch.Size(_OUTPUT_SHAPE), (
        f"Shape {tensor.shape} != {_OUTPUT_SHAPE}"
    )
    assert tensor.dtype == torch.float32


# ── Test: long clip (3.5 s) → centre-crop → correct shape ────────────────────


def test_long_audio(tmp_path: pathlib.Path) -> None:
    """A 3.5 s clip must be centre-cropped to 2.0 s → [3, 224, 224]. [FR-AUD-004]"""
    wav = _write_wav(tmp_path / "long.wav", duration_sec=3.5)
    tensor = preprocess_audio(wav)

    assert tensor.shape == torch.Size(_OUTPUT_SHAPE), (
        f"Shape {tensor.shape} != {_OUTPUT_SHAPE}"
    )
    assert tensor.dtype == torch.float32


# ── Test: short-but-valid clip (1.0 s) → zero-pad right → correct shape ──────


def test_short_padded(tmp_path: pathlib.Path) -> None:
    """A 1.0 s clip must be zero-padded (right) to 2.0 s → [3, 224, 224]. [FR-AUD-004]"""
    wav = _write_wav(tmp_path / "padded.wav", duration_sec=1.0)
    tensor = preprocess_audio(wav)

    assert tensor.shape == torch.Size(_OUTPUT_SHAPE), (
        f"Shape {tensor.shape} != {_OUTPUT_SHAPE}"
    )
    assert tensor.dtype == torch.float32


# ── Test: stereo WAV → averaged to mono → no error ───────────────────────────


def test_stereo_to_mono(tmp_path: pathlib.Path) -> None:
    """A stereo WAV must be averaged to mono without raising any error. [FR-AUD-003]"""
    wav = _write_wav(tmp_path / "stereo.wav", duration_sec=2.0, n_channels=2)
    tensor = preprocess_audio(wav)

    assert tensor.shape == torch.Size(_OUTPUT_SHAPE), (
        f"Stereo→mono produced wrong shape: {tensor.shape}"
    )
    assert tensor.dtype == torch.float32


# ── Test: output contract ─────────────────────────────────────────────────────


def test_output_contract(tmp_path: pathlib.Path) -> None:
    """Valid 2.0 s input must return torch.Tensor [3, 224, 224] float32. [FR-AUD-008]"""
    wav = _write_wav(tmp_path / "contract.wav", duration_sec=2.0)
    tensor = preprocess_audio(wav)

    assert isinstance(tensor, torch.Tensor), "Return type must be torch.Tensor"
    assert tensor.shape == torch.Size(_OUTPUT_SHAPE), (
        f"Shape contract violated: {tensor.shape}"
    )
    assert tensor.dtype == torch.float32, (
        f"Dtype contract violated: {tensor.dtype}"
    )
    # Values must be finite (no NaN / Inf from log-dB or normalisation)
    assert torch.isfinite(tensor).all(), "Tensor contains NaN or Inf values"


# ── Test: latency ≤ 500 ms ────────────────────────────────────────────────────


@pytest.mark.timeout(_LATENCY_BUDGET_S + 2.0)  # test-level timeout with 2 s headroom
def test_latency(tmp_path: pathlib.Path) -> None:
    """preprocess_audio() on a 2.0 s WAV must complete in ≤ 500 ms. [NFR-Performance]"""
    wav = _write_wav(tmp_path / "latency.wav", duration_sec=2.0)

    t0 = time.perf_counter()
    _ = preprocess_audio(wav)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms <= _LATENCY_BUDGET_MS, (
        f"Latency {elapsed_ms:.1f}ms exceeded {_LATENCY_BUDGET_MS}ms budget. "
        "NFR-Performance violation."
    )


# ── Test: unsupported format → DSDBAError(AUD-002) ───────────────────────────


def test_unsupported_format(tmp_path: pathlib.Path) -> None:
    """A .txt file must raise DSDBAError with code AUD-002. [FR-AUD-001]"""
    bad_file = tmp_path / "audio.txt"
    bad_file.write_text("not audio")

    with pytest.raises(DSDBAError) as exc_info:
        preprocess_audio(bad_file)

    error: DSDBAError = exc_info.value
    assert error.code == ErrorCode.AUD_002, (
        f"Expected AUD-002, got {error.code}"
    )


# ── Test: batch preprocessing preserves order and shape ──────────────────────


def test_batch_preprocess_order(tmp_path: pathlib.Path) -> None:
    """preprocess_batch() must return results in the same order as inputs. [FR-AUD-009]"""
    paths = [
        _write_wav(tmp_path / f"batch_{i}.wav", duration_sec=2.0)
        for i in range(3)
    ]
    tensors = preprocess_batch(paths)

    assert len(tensors) == 3, f"Expected 3 tensors, got {len(tensors)}"
    for i, t in enumerate(tensors):
        assert t.shape == torch.Size(_OUTPUT_SHAPE), (
            f"Batch item {i} has wrong shape: {t.shape}"
        )
        assert t.dtype == torch.float32, (
            f"Batch item {i} has wrong dtype: {t.dtype}"
        )


# ── Unit tests: internal helpers ──────────────────────────────────────────────


def test_to_mono_stereo_averages() -> None:
    """to_mono() must average two channels correctly. [FR-AUD-003]"""
    ch1 = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    ch2 = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    stereo = np.stack([ch1, ch2], axis=0)  # (2, 3)
    mono = to_mono(stereo)

    expected_avg = 0.6
    np.testing.assert_allclose(mono / mono.max(), np.ones(3), atol=1e-5)
    assert mono.ndim == 1, "to_mono output must be 1D"
    # Check peak normalisation: peak must be ≤ 1.0
    assert float(np.abs(mono).max()) <= 1.0 + 1e-6


def test_fix_duration_pad() -> None:
    """fix_duration() must zero-pad a short waveform to n_samples. [FR-AUD-004]"""
    from src.audio.dsp import _CFG

    cfg = _CFG.audio
    short = np.ones(cfg.n_samples // 2, dtype=np.float32)
    fixed = fix_duration(short, cfg)

    assert len(fixed) == cfg.n_samples, (
        f"Padded length {len(fixed)} != target {cfg.n_samples}"
    )
    assert fixed[cfg.n_samples // 2 :].sum() == 0.0, (
        "Right-padded region should be all zeros"
    )


def test_fix_duration_crop() -> None:
    """fix_duration() must centre-crop a long waveform to n_samples. [FR-AUD-004]"""
    from src.audio.dsp import _CFG

    cfg = _CFG.audio
    long_wave = np.arange(cfg.n_samples * 2, dtype=np.float32)
    fixed = fix_duration(long_wave, cfg)

    assert len(fixed) == cfg.n_samples, (
        f"Cropped length {len(fixed)} != target {cfg.n_samples}"
    )
    # Centre crop: start = (2*n - n) // 2 = n // 2
    expected_start = cfg.n_samples // 2
    np.testing.assert_array_equal(
        fixed, long_wave[expected_start : expected_start + cfg.n_samples]
    )
