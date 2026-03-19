"""
Module: train
SRS Reference: FR-CV-003, FR-CV-004, FR-CV-005, FR-CV-006, FR-CV-007, FR-CV-008
SDLC Phase: Phase 4 — Sprint B (full implementation)
Sprint: B
Pipeline Stage: CV Inference (training only — no inference logic here)
Interface Contract:
  Input:  torch.nn.Module (EfficientNet-B4), DataLoader (train + val), config
  Output: DSDBAModel — best-checkpoint model after two-phase training
Latency Target: ≥ 60 samples/s throughput on T4 GPU per NFR-Performance
Open Questions Resolved: Q3 — batch_size=16 feasible on T4 (ADR-0008)
Open Questions Blocking: None
MCP Tools Used: context7-mcp (PyTorch 2.x training APIs),
                huggingface-mcp (HF Hub checkpoint upload)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 4 — Sprint B — Pending V.E.R.I.F.Y.]
# Module boundary: training loop + checkpointing ONLY. No inference. [.cursorrules]

from __future__ import annotations

import pathlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from src.audio.dsp import preprocess_audio
from src.cv.model import DSDBAModel, build_model
from src.utils.config import DSDBAConfig, load_config
from src.utils.logger import get_logger, log_info, log_warning

# ── Module-level singletons ───────────────────────────────────────────────────

_CONFIG_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "config.yaml"
)
_CFG: DSDBAConfig = load_config(_CONFIG_PATH)
_log = get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────


class FoRDataset(Dataset):
    """Fake-or-Real dataset wrapping preprocessed audio tensors.

    Each sample is loaded via src/audio/dsp.preprocess_audio() and cached
    in memory after first access.  Labels: bonafide=0, spoof=1 [FR-CV-002].

    Args:
        file_paths (list[pathlib.Path]): Audio file paths.
        labels (list[int]): 0 (bonafide) or 1 (spoof) per file.
        transform (callable | None): Optional augmentation applied to tensor.
    """

    def __init__(
        self,
        file_paths: list[pathlib.Path],
        labels: list[int],
        transform: Any | None = None,
    ) -> None:
        if len(file_paths) != len(labels):
            raise ValueError("file_paths and labels must have same length")
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx not in self._cache:
            self._cache[idx] = preprocess_audio(self.file_paths[idx])
        tensor = self._cache[idx].clone()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, self.labels[idx]


class TensorDatasetCV(Dataset):
    """Lightweight dataset from pre-computed tensor list + labels.

    Used when tensors are already preprocessed (e.g. loaded from .pt files).
    """

    def __init__(
        self,
        tensors: list[torch.Tensor],
        labels: list[int],
        transform: Any | None = None,
    ) -> None:
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = self.tensors[idx].clone()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, self.labels[idx]


# ── SpecAugment Transform ────────────────────────────────────────────────────


class SpecAugment(nn.Module):
    """SpecAugment data augmentation applied on [C, H, W] spectrogram tensors.

    Implements time masking, frequency masking, random time shift, and
    additive Gaussian noise. All parameters from config.yaml:training.augmentation.
    SRS: FR-CV-006 (SHOULD).

    Args:
        time_mask_pct (float): Max fraction of width to mask.
        freq_mask_pct (float): Max fraction of height to mask.
        time_shift_pixels (int): Max circular shift in pixels along width.
        noise_std (float): Gaussian noise standard deviation.
    """

    def __init__(
        self,
        time_mask_pct: float,
        freq_mask_pct: float,
        time_shift_pixels: int,
        noise_std: float,
    ) -> None:
        super().__init__()
        self.time_mask_pct = time_mask_pct
        self.freq_mask_pct = freq_mask_pct
        self.time_shift_pixels = time_shift_pixels
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a [C, H, W] tensor (training only)."""
        _, h, w = x.shape

        # Time masking: zero-out a random contiguous block of columns
        t_width = int(torch.randint(0, max(1, int(w * self.time_mask_pct)), (1,)).item())
        if t_width > 0:
            t_start = torch.randint(0, w - t_width + 1, (1,)).item()
            x[:, :, t_start : t_start + t_width] = 0.0

        # Frequency masking: zero-out a random contiguous block of rows
        f_height = int(torch.randint(0, max(1, int(h * self.freq_mask_pct)), (1,)).item())
        if f_height > 0:
            f_start = torch.randint(0, h - f_height + 1, (1,)).item()
            x[:, f_start : f_start + f_height, :] = 0.0

        # Time shift: circular roll along width
        if self.time_shift_pixels > 0:
            shift = torch.randint(
                -self.time_shift_pixels, self.time_shift_pixels + 1, (1,)
            ).item()
            x = torch.roll(x, shifts=int(shift), dims=2)

        # Gaussian noise [FR-CV-006: SNR ≥ 20 dB]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x


# ── Public Functions ──────────────────────────────────────────────────────────


def get_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss.

    Args:
        labels (list[int]): All training labels (0 or 1).
        num_classes (int): Number of classes. Default 2.

    Returns:
        torch.Tensor: Class weights of shape [num_classes], float32.
                      Higher weight for the minority class. SRS: FR-CV-005.
    """
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    counts = counts.float().clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_augmentations(cfg: DSDBAConfig) -> SpecAugment | None:
    """Build SpecAugment transform from config.yaml:training.augmentation.

    Args:
        cfg (DSDBAConfig): Full configuration.

    Returns:
        SpecAugment | None: Transform if augmentation is enabled, else None.
        SRS: FR-CV-006 (SHOULD).
    """
    aug_cfg = cfg.training.augmentation
    if not aug_cfg.specaugment_enabled:
        return None

    img_w: int = cfg.audio.output_tensor_shape[2]
    duration: float = cfg.audio.duration_sec
    shift_pixels = int(aug_cfg.time_shift_sec / duration * img_w)

    snr_linear = 10.0 ** (aug_cfg.gaussian_noise_snr_db / 10.0)
    noise_std = 1.0 / (snr_linear ** 0.5)

    return SpecAugment(
        time_mask_pct=aug_cfg.time_mask_pct,
        freq_mask_pct=aug_cfg.freq_mask_pct,
        time_shift_pixels=shift_pixels,
        noise_std=noise_std,
    )


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate from true labels and prediction scores.

    EER is the point where False Positive Rate equals False Negative Rate.
    SRS: FR-CV-008. Q7 (EER scoring protocol) is OPEN for Phase 7 — this
    implementation uses sklearn.metrics.roc_curve as the interim method.

    Args:
        y_true (np.ndarray): Binary labels (0=bonafide, 1=spoof).
        y_scores (np.ndarray): Model confidence scores for class 1 (spoof).

    Returns:
        float: Equal Error Rate in [0.0, 1.0].
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def train_epoch(
    model: DSDBAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Run one training epoch with optional mixed precision.

    Args:
        model: DSDBAModel in train mode.
        loader: Training DataLoader yielding (tensor, label) batches.
        optimizer: Optimizer (Adam).
        criterion: Loss function (CrossEntropyLoss with class weights).
        device: Target device (cuda / cpu).
        scaler: GradScaler for mixed precision. None disables AMP.

    Returns:
        dict with keys: "loss" (epoch mean), "samples_per_sec" (throughput).
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    t_start = time.perf_counter()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        n_samples += batch_x.size(0)

    elapsed = time.perf_counter() - t_start
    return {
        "loss": running_loss / max(n_samples, 1),
        "samples_per_sec": n_samples / max(elapsed, 1e-6),
    }


@torch.no_grad()
def validate_epoch(
    model: DSDBAModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation: compute loss, AUC-ROC, and EER on the full val set.

    Args:
        model: DSDBAModel in eval mode (set internally).
        loader: Validation DataLoader.
        device: Target device.

    Returns:
        dict with keys: "auc_roc", "eer", "loss".
        SRS: FR-CV-008.
    """
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    running_loss = 0.0
    n_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device, dtype=torch.long, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        running_loss += loss.item() * batch_x.size(0)
        n_samples += batch_x.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
        all_labels.extend(batch_y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)

    auc = float(roc_auc_score(y_true, y_scores)) if len(set(all_labels)) > 1 else 0.0
    eer = compute_eer(y_true, y_scores) if len(set(all_labels)) > 1 else 1.0

    return {
        "auc_roc": auc,
        "eer": eer,
        "loss": running_loss / max(n_samples, 1),
    }


def run_training(
    cfg: DSDBAConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: pathlib.Path,
    device: torch.device | None = None,
) -> tuple[DSDBAModel, pathlib.Path]:
    """Execute the full two-phase EfficientNet-B4 training protocol.

    Phase 1 — Frozen backbone: train only the classification head for
    cfg.model.frozen_epochs epochs [FR-CV-003].
    Phase 2 — Fine-tune: unfreeze top 3 backbone stages, train at
    lr ≤ cfg.model.finetune_lr with early stopping [FR-CV-003].

    Checkpoint every epoch. Upload best to HF Hub if repo configured.
    SRS: FR-CV-003–008.

    Args:
        cfg (DSDBAConfig): Full configuration.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        output_dir (pathlib.Path): Directory for .pth checkpoints.
        device (torch.device | None): GPU/CPU. Auto-detects if None.

    Returns:
        tuple[DSDBAModel, pathlib.Path]: (trained model, best checkpoint path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=True)
    model = model.to(device)

    # Phase 1: frozen backbone [FR-CV-003]
    model.freeze_backbone()

    class_weights = get_class_weights(
        [label for _, label in train_loader.dataset], cfg.model.num_classes  # type: ignore[arg-type]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.model.finetune_lr,
    )

    scaler: torch.amp.GradScaler | None = None
    if cfg.training.mixed_precision and device.type == "cuda":
        scaler = torch.amp.GradScaler(device.type)

    best_auc: float = 0.0
    best_path: pathlib.Path = output_dir / "best_model.pth"
    patience_counter: int = 0

    total_epochs = cfg.training.max_epochs
    frozen_epochs = cfg.model.frozen_epochs

    for epoch in range(1, total_epochs + 1):
        # Phase transition: unfreeze after frozen_epochs [FR-CV-003]
        if epoch == frozen_epochs + 1:
            model.unfreeze_top_n(3)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.model.finetune_lr,
            )
            log_info(
                stage="CV Training",
                message=f"Phase 2: unfroze top 3 backbone stages at epoch {epoch}",
                srs_ref="FR-CV-003",
            )

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_epoch(model, val_loader, device)

        log_info(
            stage="CV Training",
            message=f"Epoch {epoch}/{total_epochs}",
            srs_ref="FR-CV-008",
            data={
                "train_loss": round(train_metrics["loss"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_auc_roc": round(val_metrics["auc_roc"], 4),
                "val_eer": round(val_metrics["eer"], 4),
                "samples_per_sec": round(train_metrics["samples_per_sec"], 1),
                "phase": "frozen" if epoch <= frozen_epochs else "finetune",
            },
        )

        # Checkpoint every epoch [NFR-Reliability]
        if cfg.training.save_every_epoch:
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)

        # Track best model by AUC-ROC [FR-CV-008]
        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            log_info(
                stage="CV Training",
                message=f"New best AUC-ROC: {best_auc:.4f} — saved {best_path.name}",
                srs_ref="FR-CV-007",
            )
        else:
            patience_counter += 1

        # Early stopping [FR-CV-003]
        if patience_counter >= cfg.training.early_stopping_patience:
            log_info(
                stage="CV Training",
                message=f"Early stopping at epoch {epoch} (patience={cfg.training.early_stopping_patience})",
                srs_ref="FR-CV-003",
            )
            break

    # Upload best checkpoint to HF Hub [FR-CV-007]
    hf_repo = cfg.training.hf_model_repo
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(best_path),
                path_in_repo="best_model.pth",
                repo_id=hf_repo,
            )
            log_info(
                stage="CV Training",
                message=f"Best checkpoint uploaded to HF Hub: {hf_repo}",
                srs_ref="FR-CV-007",
            )
        except Exception as exc:
            log_warning(
                stage="CV Training",
                message=f"HF Hub upload failed: {type(exc).__name__}",
                srs_ref="FR-CV-007",
            )
    else:
        log_warning(
            stage="CV Training",
            message="HF model repo not configured — skipping upload",
            srs_ref="FR-CV-007",
        )

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    return model, best_path
