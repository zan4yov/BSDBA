"""
Module: logger
SRS Reference: NFR-Security (no print() in any module), NFR-Maintainability
SDLC Phase: Phase 4 — Sprint A (enhanced with helper functions per chain-05)
Sprint: N/A (utility — shared across all sprints)
Pipeline Stage: All stages (utility)
Interface Contract:
  Input:  N/A — exposes get_logger() factory + log_info/log_warning/log_error helpers
  Output: logging.Logger — configured structured JSON logger
Latency Target: N/A (logging overhead negligible)
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

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


class _StructuredJSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Each record is serialised to: {"timestamp": ..., "level": ...,
    "logger": ..., "message": ..., [optional extra fields]}.

    Captured extra fields: stage, srs_ref, latency_ms, error_code, data.

    Security rule: API key values must NEVER appear in any log field.
    The formatter does not redact automatically — callers are responsible.
    [FR-NLP-005, NFR-Security]
    """

    _EXTRA_KEYS: tuple[str, ...] = ("stage", "srs_ref", "latency_ms", "error_code", "data")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = traceback.format_exception(*record.exc_info)

        for key in self._EXTRA_KEYS:
            if hasattr(record, key):
                value = getattr(record, key)
                if value is not None:
                    payload[key] = value

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Return a named structured JSON logger for a pipeline module.

    Emits structured JSON records to stdout. One logger per module —
    pass ``__name__`` as the ``name`` argument.

    Security: NEVER log API key values. Log the env variable **name** only.
    Example::

        logger.info("Qwen API key resolved", extra={"stage": "NLP"})
        # NOT: logger.info(f"Key = {api_key}")  ← violates FR-NLP-005

    Args:
        name (str): Logger name. Convention: pass ``__name__`` from the
                    calling module (e.g. ``"src.audio.dsp"``).

    Returns:
        logging.Logger: Configured logger with structured JSON output.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_StructuredJSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger


# ── Module-level pipeline logger (shared helper functions) ────────────────────

_pipeline_logger: logging.Logger = get_logger("dsdba.pipeline")


def log_info(
    stage: str,
    message: str,
    data: dict[str, Any] | None = None,
    srs_ref: str = "",
) -> None:
    """Emit a structured INFO record from a pipeline stage.

    Produces JSON: {timestamp, level, stage, message, [data], [srs_ref]}.

    Security: never pass API key values in ``data``. [FR-NLP-005]

    Args:
        stage (str): Pipeline stage name (e.g. "Audio DSP", "CV Inference").
        message (str): Human-readable log message. No sensitive data.
        data (dict | None): Optional structured payload for machine parsing.
                            Must be JSON-serialisable. Default None.
        srs_ref (str): Optional SRS FR ID reference (e.g. "FR-AUD-010").
    """
    _pipeline_logger.info(
        message,
        extra={"stage": stage, "srs_ref": srs_ref or None, "data": data},
    )


def log_warning(
    stage: str,
    message: str,
    data: dict[str, Any] | None = None,
    srs_ref: str = "",
) -> None:
    """Emit a structured WARNING record from a pipeline stage.

    Args:
        stage (str): Pipeline stage name.
        message (str): Warning message. No sensitive data.
        data (dict | None): Optional structured payload.
        srs_ref (str): Optional SRS FR ID reference.
    """
    _pipeline_logger.warning(
        message,
        extra={"stage": stage, "srs_ref": srs_ref or None, "data": data},
    )


def log_error(
    stage: str,
    message: str,
    error_code: str = "",
    data: dict[str, Any] | None = None,
) -> None:
    """Emit a structured ERROR record from a pipeline stage.

    Security: error messages MUST be generic — no file paths, internal state,
    or API credentials. [NFR-Security]

    Args:
        stage (str): Pipeline stage name.
        message (str): Generic error message safe for structured logging.
        error_code (str): SRS error code (e.g. "AUD-001"). Default "".
        data (dict | None): Optional structured diagnostic payload.
    """
    _pipeline_logger.error(
        message,
        extra={"stage": stage, "error_code": error_code or None, "data": data},
    )
