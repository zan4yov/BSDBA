"""
Module: errors
SRS Reference: FR-AUD-001, FR-AUD-005
SDLC Phase: Phase 3 — Environment Setup & MCP Configuration
Sprint: N/A (utility — no sprint boundary)
Pipeline Stage: Audio DSP (error codes), CV Inference (error codes)
Interface Contract:
  Input:  N/A — defines exception types and error code constants
  Output: N/A — imported by dsp.py, infer.py, gradcam.py
Latency Target: N/A (exception handling only)
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

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    """SRS-defined error codes for structured error reporting.

    All codes are defined in config.yaml and cross-referenced to SRS FRs.
    String inheritance allows direct use in f-strings and JSON serialisation.
    """

    AUD_001 = "AUD-001"
    """Audio clip duration < min_duration_sec (0.5 s). SRS: FR-AUD-005."""

    AUD_002 = "AUD-002"
    """Unsupported audio file format. SRS: FR-AUD-001."""

    CV_001 = "CV-001"
    """Tensor shape or dtype contract violated at CV module entry. SRS: FR-AUD-008."""

    CV_002 = "CV-002"
    """Grad-CAM saliency map is all-zero — target layer misconfigured. SRS: FR-CV-010."""


@dataclass
class DSDBAError(Exception):
    """Structured exception for DSDBA pipeline errors.

    Carries an SRS-traceable error code, a human-readable message, and the
    pipeline stage where the error originated. Used by all src/ modules to
    raise consistent, loggable exceptions.

    Security note: the ``message`` field MUST contain only generic information
    safe to surface to the Gradio UI. Detailed diagnostics belong in structured
    logs (src/utils/logger.py) only. [NFR-Security]

    Args:
        code (ErrorCode): SRS-derived error code from the ``ErrorCode`` enum.
        message (str): Generic error message safe for UI display. No file paths,
                       no internal state details, no API keys.
        stage (str): Pipeline stage that raised the error (e.g. "Audio DSP",
                     "CV Inference", "XAI Grad-CAM"). For structured logging.

    Example:
        >>> raise DSDBAError(
        ...     code=ErrorCode.AUD_001,
        ...     message="Audio clip is too short. Minimum duration is 0.5 s.",
        ...     stage="Audio DSP",
        ... )
    """

    code: ErrorCode | str
    message: str
    stage: str

    def __str__(self) -> str:
        code_str = self.code.value if isinstance(self.code, ErrorCode) else str(self.code)
        return f"[{code_str}] {self.stage}: {self.message}"

    def to_dict(self) -> dict[str, str]:
        """Serialise to a JSON-safe dict for structured logging.

        Returns:
            dict[str, str]: Keys are "code", "stage", "message".
        """
        return {
            "code": self.code.value if isinstance(self.code, ErrorCode) else str(self.code),
            "stage": self.stage,
            "message": self.message,
        }
