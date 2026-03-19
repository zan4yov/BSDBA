"""
Module: config
SRS Reference: NFR-Maintainability (all hyperparameters via config.yaml)
SDLC Phase: Phase 3 — Environment Setup & MCP Configuration
Sprint: N/A (utility — shared across all sprints)
Pipeline Stage: All stages (utility)
Interface Contract:
  Input:  pathlib.Path — path to config.yaml
  Output: DSDBAConfig — Pydantic-validated configuration object
Latency Target: N/A (called once at module import)
Open Questions Resolved: N/A
Open Questions Blocking: None
MCP Tools Used: None
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 3 — Sprint N/A — Pending V.E.R.I.F.Y.]

from __future__ import annotations

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class AudioConfig(BaseModel):
    """Audio DSP hyperparameters. SRS: FR-AUD-001–011."""

    sample_rate: int = Field(16000, description="Resample target in Hz [FR-AUD-002]")
    duration_sec: float = Field(2.0, description="Fixed clip duration [FR-AUD-004]")
    n_samples: int = Field(32000, description="16000 Hz × 2.0 s [FR-AUD-004]")
    n_mels: int = Field(128, description="Mel filter bank count [FR-AUD-006]")
    n_fft: int = Field(2048, description="FFT window size [FR-AUD-006]")
    hop_length: int = Field(512, description="STFT hop length [FR-AUD-006]")
    window: str = Field("hann", description="STFT window function [FR-AUD-006]")
    min_duration_sec: float = Field(0.5, description="Minimum clip length [FR-AUD-005]")
    max_file_size_mb: int = Field(20, description="Maximum upload size [FR-DEP-002]")
    output_tensor_shape: list[int] = Field([3, 224, 224], description="Output shape [FR-AUD-008]")
    output_dtype: str = Field("float32", description="Output tensor dtype [FR-AUD-008]")
    error_code_too_short: str = Field("AUD-001", description="Error code: too short [FR-AUD-005]")
    error_code_bad_format: str = Field("AUD-002", description="Error code: bad format [FR-AUD-001]")
    supported_formats: list[str] = Field(["wav", "flac"], description="SHALL formats [FR-AUD-001]")
    optional_formats: list[str] = Field(["mp3", "ogg"], description="MAY formats [FR-AUD-011]")
    resampling_method: str = Field("kaiser_best", description="Resampling quality [FR-AUD-002]")
    fmin: float = Field(0.0, description="Mel fmin Hz [FR-AUD-006, Q5 RESOLVED]")
    fmax: int = Field(8000, description="Mel fmax Hz = sr/2 [FR-AUD-006, Q5 RESOLVED]")


class ModelConfig(BaseModel):
    """CV model hyperparameters. SRS: FR-CV-001–009."""

    backbone: str = Field("efficientnet_b4", description="Model backbone [FR-CV-001]")
    pretrained_weights: str = Field("imagenet1k", description="Pretrained weights [FR-CV-001]")
    num_classes: int = Field(2, description="bonafide=0, spoof=1 [FR-CV-002]")
    head_type: str = Field("linear", description="Classification head type [FR-CV-002]")
    frozen_epochs: int = Field(5, description="Frozen backbone epochs [FR-CV-003]")
    finetune_lr: float = Field(1.0e-4, description="Fine-tuning LR [FR-CV-003]")
    decision_threshold: float = Field(0.5, description="Bonafide if score >= 0.5 [FR-CV-004]")
    output_activation: str = Field("sigmoid", description="Output activation [FR-CV-004]")
    loss_function: str = Field("bce_weighted", description="Loss function [FR-CV-005]")
    checkpoint_max_mb: int = Field(250, description="Max .pth size [NFR-Performance]")
    hf_model_repo: str = Field("", description="HF model repo ID [FR-CV-007]")


class AugmentationConfig(BaseModel):
    """SpecAugment configuration. SRS: FR-CV-006 (SHOULD)."""

    specaugment_enabled: bool = True
    time_mask_pct: float = 0.10
    freq_mask_pct: float = 0.10
    time_shift_sec: float = 0.1
    gaussian_noise_snr_db: int = 20


class TrainingConfig(BaseModel):
    """Training loop configuration. SRS: FR-CV-003–008."""

    batch_size: int = Field(16, description="Colab T4 batch size [Q3 RESOLVED]")
    gradient_checkpointing: bool = Field(True, description="VRAM safety [Q3]")
    mixed_precision: bool = Field(True, description="fp16 training [T4 efficiency]")
    num_workers: int = Field(2, description="DataLoader workers [Colab CPU]")
    max_epochs: int = Field(30, description="Max training epochs [FR-CV-003]")
    early_stopping_patience: int = Field(5, description="Early stop patience [FR-CV-003]")
    save_every_epoch: bool = Field(True, description="Checkpoint per epoch [NFR-Reliability]")
    dataset_split: str = Field("for_2sec", description="FoR variant [Q2 RESOLVED]")
    dataset_source: str = Field("kaggle", description="Dataset origin")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class AcceptanceCriteriaConfig(BaseModel):
    """Model accuracy targets. SRS: FR-CV-008, NFR-Accuracy."""

    target_eer: float = 0.10
    target_auc_roc: float = 0.90
    target_f1_macro: float = 0.88
    target_overall_accuracy: float = 0.90
    target_ece: float = 0.05
    target_xai_faithfulness: float = 0.65


class BandHzConfig(BaseModel):
    """Frequency band boundaries in Hz. SRS: FR-CV-013."""

    low: list[int] = Field([0, 500])
    low_mid: list[int] = Field([500, 2000])
    high_mid: list[int] = Field([2000, 4000])
    high: list[int] = Field([4000, 8000])


class GradcamConfig(BaseModel):
    """Grad-CAM / XAI configuration. SRS: FR-CV-010–016."""

    target_layer: str = Field(
        "model.features[7][-1]",
        description="Final MBConv Stage 7 [Q4 RESOLVED, ADR-0004]",
    )
    library: str = Field("pytorch_grad_cam", description="jacobgil/pytorch-grad-cam [FR-CV-011]")
    colormap: str = Field("jet", description="Overlay colormap [FR-CV-012]")
    overlay_alpha: float = Field(0.5, description="Blending alpha [FR-CV-012]")
    output_format: str = Field("png", description="Heatmap format [FR-CV-012]")
    band_hz: BandHzConfig = Field(default_factory=BandHzConfig)
    band_normalisation: str = Field("softmax", description="Band normalisation [FR-CV-014]")
    latency_target_ms: int = Field(3000, description="≤ 3000 ms CPU [FR-CV-015]")
    expose_raw_saliency: bool = Field(True, description="JSON dev endpoint [FR-CV-016]")


class NLPCachingConfig(BaseModel):
    """NLP response caching. SRS: FR-NLP-008 (SHOULD)."""

    enabled: bool = True
    cache_key_fields: list[str] = ["label", "confidence_bucket", "top_band"]
    confidence_buckets: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class NLPConfig(BaseModel):
    """NLP / Qwen 2.5 configuration. SRS: FR-NLP-001–009."""

    primary_provider: str = Field("qwen_2.5", description="Primary LLM [FR-NLP-002, Q1]")
    fallback_provider: str = Field("gemma_3", description="Fallback LLM [FR-NLP-007]")
    final_fallback: str = Field("rule_based", description="Always-available [FR-NLP-003]")
    timeout_sec: int = Field(30, description="API timeout [FR-NLP-002]")
    explanation_min_sentences: int = Field(3, description="Min sentences [FR-NLP-001]")
    explanation_max_sentences: int = Field(5, description="Max sentences [FR-NLP-001]")
    output_language: str = Field("english", description="Output language [FR-NLP-004]")
    warning_badge_text: str = Field("AI explanation unavailable", description="UI badge [FR-NLP-003]")
    api_key_env_var: str = Field("QWEN_API_KEY", description="Secret name — NOT the value [FR-NLP-005]")
    hf_token_env_var: str = Field("HF_TOKEN", description="HF token secret name")
    caching: NLPCachingConfig = Field(default_factory=NLPCachingConfig)


class DemoSamplesConfig(BaseModel):
    """Demo sample counts. SRS: FR-DEP-008."""

    bonafide: int = 2
    spoof: int = 2


class DeploymentConfig(BaseModel):
    """Deployment / UI configuration. SRS: FR-DEP-001–010."""

    framework: Literal["gradio"] = Field(
        "gradio",
        description="LOCKED: Gradio 4.x [Q6 RESOLVED, ADR-0006]",
    )
    gradio_version: str = Field("4", description="Gradio major version [FR-DEP-001]")
    auth_required: bool = Field(False, description="Public access [FR-DEP-001]")
    max_upload_mb: int = Field(20, description="Max audio file [FR-DEP-002]")
    onnx_enabled: bool = Field(True, description="ONNX mandatory [FR-DEP-010]")
    onnx_execution_providers: list[str] = Field(
        ["CPUExecutionProvider"],
        description="HF Spaces CPU-only [FR-DEP-010]",
    )
    onnx_latency_target_ms: int = Field(1500, description="ONNX latency [FR-DEP-010]")
    onnx_equivalence_tolerance: float = Field(1.0e-5, description="|ONNX-PyTorch| < 1e-5 [FR-DEP-010]")
    e2e_latency_target_ms: int = Field(15000, description="E2E wall time [FR-DEP-007]")
    cold_start_target_sec: int = Field(30, description="HF Spaces cold start [NFR-Reliability]")
    hf_space_repo: str = Field("", description="HF Space ID [FR-DEP-001]")
    demo_samples: DemoSamplesConfig = Field(default_factory=DemoSamplesConfig)
    about_section: bool = Field(True, description="About panel [FR-DEP-009]")
    dataset_citation: str = Field(
        "Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.",
        description="Dataset attribution",
    )


class PerformanceTargetsConfig(BaseModel):
    """NFR performance targets."""

    audio_dsp_ms: int = 500
    cv_inference_no_onnx_ms: int = 3000
    cv_inference_onnx_ms: int = 1500
    gradcam_ms: int = 3000
    nlp_api_ms: int = 8000
    e2e_ms: int = 15000
    training_throughput_samples_per_sec: int = 60
    checkpoint_max_mb: int = 250


# ── Root config model ──────────────────────────────────────────────────────────

class DSDBAConfig(BaseModel):
    """Root Pydantic model for config.yaml.

    RULE: No hyperparameter may appear hardcoded in any Python source file.
    This model is the single source of truth for all pipeline configuration.
    NFR-Maintainability.
    """

    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    acceptance_criteria: AcceptanceCriteriaConfig = Field(
        default_factory=AcceptanceCriteriaConfig
    )
    gradcam: GradcamConfig = Field(default_factory=GradcamConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    performance_targets: PerformanceTargetsConfig = Field(
        default_factory=PerformanceTargetsConfig
    )

    @field_validator("audio")
    @classmethod
    def validate_n_samples(cls, v: AudioConfig) -> AudioConfig:
        """Assert n_samples == sample_rate × duration_sec. SRS: FR-AUD-004."""
        expected = int(v.sample_rate * v.duration_sec)
        if v.n_samples != expected:
            raise ValueError(
                f"audio.n_samples {v.n_samples} != "
                f"sample_rate × duration_sec ({expected})"
            )
        return v

    @field_validator("audio")
    @classmethod
    def validate_fmax(cls, v: AudioConfig) -> AudioConfig:
        """Assert fmax == sample_rate / 2 per Nyquist. SRS: FR-AUD-006, Q5."""
        expected = v.sample_rate // 2
        if v.fmax != expected:
            raise ValueError(
                f"audio.fmax {v.fmax} != sample_rate/2 ({expected}). "
                "Mel fmax must equal Nyquist frequency."
            )
        return v


# ── Loader function ────────────────────────────────────────────────────────────

def load_config(config_path: pathlib.Path = pathlib.Path("config.yaml")) -> DSDBAConfig:
    """Load and validate config.yaml into a DSDBAConfig Pydantic model.

    Args:
        config_path (pathlib.Path): Path to config.yaml.
                                    Defaults to ``config.yaml`` relative to CWD.

    Returns:
        DSDBAConfig: Fully validated configuration object.

    Raises:
        FileNotFoundError: if config.yaml does not exist at the given path.
        pydantic.ValidationError: if any config value fails validation.

    Latency Target: N/A (called once at module import)
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path.resolve()}. "
            "Ensure the working directory is the project root."
        )

    raw: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return DSDBAConfig.model_validate(raw)
