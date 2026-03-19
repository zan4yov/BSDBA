"""
Module: logger
SRS Reference: NFR-Security (no print() in any module), NFR-Maintainability
SDLC Phase: Phase 3 — Environment Setup & MCP Configuration
Sprint: N/A (utility — shared across all sprints)
Pipeline Stage: All stages (utility)
Interface Contract:
  Input:  N/A — exposes get_logger() factory function
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

# [DRAFT — Phase 3 — Sprint N/A — Pending V.E.R.I.F.Y.]

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

    Security rule: API key values must NEVER appear in any log field.
    The formatter does not redact automatically — callers are responsible.
    [FR-NLP-005, NFR-Security]
    """

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

        for key in ("stage", "srs_ref", "latency_ms", "error_code"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

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
