"""
Module: dsp
SRS Reference: FR-AUD-001, FR-AUD-002, FR-AUD-003, FR-AUD-004, FR-AUD-005,
               FR-AUD-006, FR-AUD-007, FR-AUD-008, FR-AUD-009, FR-AUD-010, FR-AUD-011
SDLC Phase: Phase 4 — Sprint A (full implementation)
Sprint: A
Pipeline Stage: Audio DSP
Interface Contract:
  Input:  pathlib.Path — absolute path to WAV or FLAC file (FR-AUD-001)
  Output: torch.Tensor shape=[3, 224, 224], dtype=torch.float32 (FR-AUD-008)
Latency Target: ≤ 500 ms on CPU per NFR-Performance
Open Questions Resolved: Q5 — Mel filter bank mapping via librosa.mel_frequencies
                                (n_mels=128, fmin=0.0, fmax=8000.0) [ADR-0005]
Open Questions Blocking: None
MCP Tools Used: context7-mcp (librosa 0.10.x) — API signatures verified
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 4 — Sprint A — Pending V.E.R.I.F.Y.]

from __future__ import annotations

import pathlib
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.utils.config import AudioConfig, DSDBAConfig, load_config
from src.utils.errors import DSDBAError, ErrorCode
from src.utils.logger import get_logger, log_error, log_info

# ── Module-level singletons (immutable after import — NFR-Maintainability) ────

_CONFIG_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "config.yaml"
)
_CFG: DSDBAConfig = load_config(_CONFIG_PATH)
_log = get_logger(__name__)

# ── Internal helpers ──────────────────────────────────────────────────────────


def load_audio(
    file_path: pathlib.Path,
    cfg: AudioConfig,
) -> tuple[np.ndarray, int]:
    """Load an audio file and return the waveform and native sample rate.

    Accepts WAV and FLAC (SHALL per FR-AUD-001) via soundfile.
    Accepts MP3 and OGG (MAY per FR-AUD-011) via torchaudio.
    Raises DSDBAError(AUD-002) for any other extension.

    Audio bytes are never written to disk beyond the original file.
    Processing is performed on in-memory arrays. [NFR-Security]

    Args:
        file_path (pathlib.Path): Path to audio file.
        cfg (AudioConfig): Audio configuration from config.yaml.

    Returns:
        tuple[np.ndarray, int]:
            - waveform: shape=(n_channels, n_samples), dtype=float32.
            - sample_rate: native sample rate of the file (Hz).

    Raises:
        DSDBAError: code=AUD-002 if format not in supported or optional formats.
                    SRS: FR-AUD-001.
        OSError: if file cannot be opened (re-raised from soundfile/torchaudio).
    """
    suffix = file_path.suffix.lower().lstrip(".")
    all_formats: list[str] = cfg.supported_formats + cfg.optional_formats

    if suffix not in all_formats:
        raise DSDBAError(
            code=ErrorCode.AUD_002,
            message=(
                f"Unsupported audio format. Accepted: {cfg.supported_formats} "
                f"(required), {cfg.optional_formats} (optional)."
            ),
            stage="Audio DSP",
        )

    if suffix in cfg.supported_formats:
        # WAV / FLAC — soundfile (FR-AUD-001 SHALL path)
        data, sample_rate = sf.read(str(file_path), always_2d=True, dtype="float32")
        # soundfile returns (n_samples, n_channels) → transpose to (n_channels, n_samples)
        waveform: np.ndarray = data.T
    else:
        # MP3 / OGG — torchaudio (FR-AUD-011 MAY path)
        import torchaudio  # deferred: only needed for optional formats

        tensor, sample_rate = torchaudio.load(str(file_path))
        waveform = tensor.numpy().astype(np.float32)  # (n_channels, n_samples)

    return waveform, int(sample_rate)


def validate_duration(
    waveform: np.ndarray,
    sample_rate: int,
    cfg: AudioConfig,
) -> None:
    """Reject audio clips shorter than the minimum accepted duration.

    Duration is computed from native sample count before resampling,
    which is equivalent (resampling preserves duration).

    Args:
        waveform (np.ndarray): Waveform of shape (n_channels, n_samples).
        sample_rate (int): Native sample rate in Hz.
        cfg (AudioConfig): Audio configuration with min_duration_sec.

    Raises:
        DSDBAError: code=AUD-001 if duration < cfg.min_duration_sec (0.5 s).
                    SRS: FR-AUD-005.
    """
    n_samples: int = waveform.shape[-1]
    duration: float = n_samples / sample_rate

    if duration < cfg.min_duration_sec:
        raise DSDBAError(
            code=ErrorCode.AUD_001,
            message=(
                f"Audio clip is too short ({duration:.2f}s). "
                f"Minimum accepted duration is {cfg.min_duration_sec}s."
            ),
            stage="Audio DSP",
        )


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    cfg: AudioConfig,
) -> np.ndarray:
    """Resample the waveform to the target sample rate using Kaiser-best window.

    If the waveform is already at the target rate, returns it unchanged.
    librosa.resample supports multi-dimensional arrays (last axis = time).

    Args:
        waveform (np.ndarray): Waveform of shape (n_channels, n_samples).
        orig_sr (int): Original (native) sample rate in Hz.
        cfg (AudioConfig): Audio configuration with sample_rate and
                           resampling_method ('kaiser_best' per FR-AUD-002).

    Returns:
        np.ndarray: Resampled waveform, shape (n_channels, n_samples_new),
                    dtype float32. n_samples_new = round(n_samples * target/orig).

    Latency note: resampling is the most compute-intensive step; Kaiser-best
    quality is required by FR-AUD-002 despite higher CPU cost than soxr_hq.
    [TECH DEBT: resampy==0.4.2 compatibility unverified — see session-cheatsheet.md]
    """
    if orig_sr == cfg.sample_rate:
        return waveform

    resampled: np.ndarray = librosa.resample(
        waveform,
        orig_sr=orig_sr,
        target_sr=cfg.sample_rate,
        res_type=cfg.resampling_method,
    )
    return resampled.astype(np.float32)


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Average multi-channel audio to mono and normalise peak amplitude.

    Averaging all channels satisfies FR-AUD-003 (mono conversion).
    Peak normalisation to [-1.0, 1.0] satisfies FR-AUD-003 (amplitude range).
    A silent clip (all-zero) is returned unchanged to avoid division by zero.

    Args:
        waveform (np.ndarray): Waveform of shape (n_channels, n_samples)
                               or (n_samples,) for already-mono input.

    Returns:
        np.ndarray: Mono waveform, shape (n_samples,), dtype float32.
                    Peak amplitude ≤ 1.0. SRS: FR-AUD-003.
    """
    if waveform.ndim == 1:
        mono: np.ndarray = waveform.astype(np.float32)
    else:
        mono = waveform.mean(axis=0).astype(np.float32)

    peak = float(np.abs(mono).max())
    if peak > 0.0:
        mono = mono / peak

    return mono


def fix_duration(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Pad or trim the waveform to exactly cfg.n_samples (32,000 samples).

    - Longer clips: centre-crop to preserve the most informative region.
    - Shorter clips: zero-pad on the right (end-of-clip silence).
    - Exact length: returned unchanged.

    Args:
        waveform (np.ndarray): Mono waveform, shape (n_samples,), float32.
        cfg (AudioConfig): Audio configuration with n_samples target (32000).

    Returns:
        np.ndarray: Waveform of shape (cfg.n_samples,), dtype float32.
                    Exactly 2.0 seconds at 16,000 Hz. SRS: FR-AUD-004.
    """
    n: int = len(waveform)
    target: int = cfg.n_samples

    if n == target:
        return waveform

    if n > target:
        start: int = (n - target) // 2
        return waveform[start : start + target]

    pad_width: int = target - n
    return np.pad(waveform, (0, pad_width), mode="constant").astype(np.float32)


def extract_mel_spectrogram(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Compute a Mel-scaled power spectrogram from a mono waveform.

    Parameters sourced exclusively from config.yaml:audio. [NFR-Maintainability]
    Uses librosa 0.10.x API (verified via context7-mcp).

    Args:
        waveform (np.ndarray): Mono waveform, shape (n_samples,), float32.
                               Expected to be exactly cfg.n_samples samples.
        cfg (AudioConfig): Audio config with n_mels=128, n_fft=2048,
                           hop_length=512, window='hann', fmin=0.0,
                           fmax=8000.0. SRS: FR-AUD-006.

    Returns:
        np.ndarray: Power Mel spectrogram, shape (n_mels, T) = (128, frames),
                    dtype float32. T ≈ 63 for 32,000-sample input at hop=512.
    """
    mel_spec: np.ndarray = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        fmin=float(cfg.fmin),
        fmax=float(cfg.fmax),
    )
    return mel_spec.astype(np.float32)


def normalise_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Convert power spectrogram to log-dB scale and min-max normalise to [0, 1].

    Step 1: librosa.power_to_db(ref=np.max) — dB conversion [FR-AUD-007].
    Step 2: min-max normalisation to [0, 1] for EfficientNet-B4 input range.

    A flat spectrogram (silence / constant signal) returns an all-zero array
    to avoid division by zero artefacts.

    Args:
        spec (np.ndarray): Power Mel spectrogram, shape (n_mels, T), float32.

    Returns:
        np.ndarray: Normalised spectrogram in [0, 1], shape (n_mels, T),
                    dtype float32. SRS: FR-AUD-007.
    """
    spec_db: np.ndarray = librosa.power_to_db(spec, ref=np.max)

    s_min: float = float(spec_db.min())
    s_max: float = float(spec_db.max())
    span: float = s_max - s_min

    if span < 1e-6:
        return np.zeros_like(spec_db, dtype=np.float32)

    spec_norm: np.ndarray = ((spec_db - s_min) / span).astype(np.float32)
    return spec_norm


def to_tensor(spec: np.ndarray, cfg: AudioConfig) -> torch.Tensor:
    """Resize the spectrogram to 224×224 and replicate to 3 channels.

    Uses bilinear interpolation to match EfficientNet-B4's ImageNet input
    expectations. Three identical channels simulate an RGB image from the
    monochrome spectrogram. [FR-AUD-008]

    Args:
        spec (np.ndarray): Normalised spectrogram, shape (n_mels, T),
                           values in [0, 1], dtype float32.
        cfg (AudioConfig): Config providing output_tensor_shape=[3,224,224].

    Returns:
        torch.Tensor: shape=[3, 224, 224], dtype=torch.float32.
                      All three channels are identical (grayscale replicated).
                      SRS: FR-AUD-008.

    Raises:
        DSDBAError: code=CV-001 if output shape or dtype assertion fails.
    """
    out_h: int = cfg.output_tensor_shape[1]
    out_w: int = cfg.output_tensor_shape[2]

    # (n_mels, T) → (1, 1, n_mels, T)
    t: torch.Tensor = (
        torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    )

    # Bilinear resize → (1, 1, 224, 224)
    t_resized: torch.Tensor = F.interpolate(
        t,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )

    # Expand 1 channel → 3 channels, remove batch dim → (3, 224, 224)
    tensor_3ch: torch.Tensor = t_resized.squeeze(0).expand(3, -1, -1).contiguous()

    expected_shape = torch.Size(cfg.output_tensor_shape)
    if tensor_3ch.shape != expected_shape or tensor_3ch.dtype != torch.float32:
        raise DSDBAError(
            code=ErrorCode.CV_001,
            message=(
                f"Output tensor contract violated: "
                f"shape={tensor_3ch.shape} dtype={tensor_3ch.dtype}. "
                f"Expected {expected_shape} float32."
            ),
            stage="Audio DSP",
        )

    return tensor_3ch


# ── Public API ────────────────────────────────────────────────────────────────


def preprocess_audio(file_path: pathlib.Path) -> torch.Tensor:
    """Load, validate, and transform an audio file into a [3, 224, 224] float32 tensor.

    This is the sole public entry point for the Audio DSP stage. It composes
    all sub-steps in the order mandated by the Phase 2 interface contract
    (ADR-0007). All parameters come from config.yaml via the module-level _CFG.

    Pipeline steps (all mandatory, FR-IDs in parentheses):
        1.  Validate file extension                         [FR-AUD-001]
        2.  Load audio into memory (soundfile / torchaudio) [FR-AUD-001, FR-AUD-011]
        3.  Log metadata: sample_rate, duration, channels   [FR-AUD-010]
        4.  Validate minimum duration ≥ 0.5 s               [FR-AUD-005]
        5.  Resample to 16,000 Hz (Kaiser-best)             [FR-AUD-002]
        6.  Convert to mono + peak-normalise to [-1, 1]     [FR-AUD-003]
        7.  Pad (zero-right) or centre-crop to 32,000 samples [FR-AUD-004]
        8.  Compute 128-band Mel power spectrogram          [FR-AUD-006]
        9.  Log-dB conversion + min-max normalise [0, 1]   [FR-AUD-007]
        10. Bilinear resize to 224×224, replicate to 3 ch  [FR-AUD-008]
        11. Log processing latency                          [FR-AUD-010]

    Security: audio data stays in NumPy arrays — zero disk writes. [NFR-Security]
    All config values from config.yaml — no magic numbers. [NFR-Maintainability]

    Args:
        file_path (pathlib.Path): Path to audio file. Must be WAV or FLAC
                                  (SHALL, FR-AUD-001) or MP3/OGG (MAY,
                                  FR-AUD-011). File must exist and be readable.

    Returns:
        torch.Tensor: shape=[3, 224, 224], dtype=torch.float32.
                      Three identical channels (grayscale spectrogram replicated).
                      Values in [0.0, 1.0] after min-max normalisation.

    Raises:
        DSDBAError: code=AUD-001 if duration < 0.5 s [FR-AUD-005].
        DSDBAError: code=AUD-002 if file format unsupported [FR-AUD-001].
        DSDBAError: code=CV-001 if output tensor shape/dtype violated [FR-AUD-008].
        OSError: if file cannot be opened (propagated from soundfile/torchaudio).

    Latency Target: ≤ 500 ms on CPU per NFR-Performance.
    MCP Tools Used: context7-mcp (librosa 0.10.x)
    """
    t_start: float = time.perf_counter()
    cfg: AudioConfig = _CFG.audio

    # Steps 1–2: load (includes format validation)
    waveform, orig_sr = load_audio(file_path, cfg)

    # Step 3: log metadata [FR-AUD-010]
    n_channels: int = waveform.shape[0]
    duration_native: float = waveform.shape[1] / orig_sr
    log_info(
        stage="Audio DSP",
        message="Audio file loaded",
        srs_ref="FR-AUD-010",
        data={
            "file": file_path.name,
            "sample_rate_native": orig_sr,
            "duration_sec": round(duration_native, 4),
            "channels": n_channels,
        },
    )

    # Step 4: validate minimum duration [FR-AUD-005]
    validate_duration(waveform, orig_sr, cfg)

    # Step 5: resample to 16,000 Hz [FR-AUD-002]
    waveform = resample_audio(waveform, orig_sr, cfg)

    # Step 6: mono + peak-normalise [FR-AUD-003]
    waveform_mono: np.ndarray = to_mono(waveform)

    # Step 7: pad or centre-crop to n_samples [FR-AUD-004]
    waveform_fixed: np.ndarray = fix_duration(waveform_mono, cfg)

    # Step 8: Mel power spectrogram [FR-AUD-006]
    mel_spec: np.ndarray = extract_mel_spectrogram(waveform_fixed, cfg)

    # Step 9: log-dB + min-max normalise [FR-AUD-007]
    mel_norm: np.ndarray = normalise_spectrogram(mel_spec)

    # Step 10: resize + 3-channel tensor [FR-AUD-008]
    tensor: torch.Tensor = to_tensor(mel_norm, cfg)

    # Step 11: log latency [FR-AUD-010]
    latency_ms: float = (time.perf_counter() - t_start) * 1000.0
    log_info(
        stage="Audio DSP",
        message="Preprocessing complete",
        srs_ref="FR-AUD-010",
        data={
            "latency_ms": round(latency_ms, 2),
            "output_shape": list(tensor.shape),
            "output_dtype": str(tensor.dtype),
        },
    )

    target_ms: int = _CFG.performance_targets.audio_dsp_ms
    if latency_ms > target_ms:
        from src.utils.logger import log_warning

        log_warning(
            stage="Audio DSP",
            message=f"Latency {latency_ms:.1f}ms exceeded target {target_ms}ms.",
            srs_ref="NFR-Performance",
            data={"latency_ms": round(latency_ms, 2), "target_ms": target_ms},
        )

    return tensor


def preprocess_batch(file_paths: list[pathlib.Path]) -> list[torch.Tensor]:
    """Apply preprocess_audio() to an ordered list of audio files.

    Fail-fast: raises DSDBAError on the first file that fails validation.
    Preserves order of input paths in the output list.

    Args:
        file_paths (list[pathlib.Path]): Ordered list of audio file paths.
                                         Each must pass preprocess_audio()
                                         validation.

    Returns:
        list[torch.Tensor]: One [3, 224, 224] float32 tensor per input path,
                            in the same order as file_paths.

    Raises:
        DSDBAError: propagated from preprocess_audio() on first failure.

    Latency Target: ≤ 500 ms × len(file_paths). SRS: FR-AUD-009 (SHOULD).
    """
    results: list[torch.Tensor] = []
    for path in file_paths:
        results.append(preprocess_audio(path))
    return results
