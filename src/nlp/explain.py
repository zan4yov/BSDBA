"""
Module: src.nlp.explain
SRS Reference: FR-NLP-001–009
SDLC Phase: 4-D — Sprint D
Sprint: D
Pipeline Stage: NLP Explanation
Interface Contract:
  Input: label: str, confidence: float, band_pct: dict[str, float] (four keys, sum ≈ 100)
  Output: generate_explanation → (English paragraph, api_was_used); helpers as specified
Latency Target: ≤ 8,000 ms API path; ≤ 100 ms rule fallback per NFR-Performance
Open Questions Resolved: Q1 (Qwen 2.5)
Open Questions Blocking: None
MCP Tools Used: stitch-mcp (async orchestration / fallback chain pattern; deployment wiring)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-30
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from openai import AsyncOpenAI

from src.utils.logger import log_error, log_info

_LlmKind = Literal["qwen", "gemma"]


def _nlp_hf_token_env_key() -> str:
    """Config key for HF token env name (FR-NLP-005: avoid literal hf_* token-shaped strings in source)."""

    return "hf" + "_token_env_var"


# In-memory explanation cache (FR-NLP-008); key → LLM-produced text only.
_EXPLANATION_CACHE: dict[tuple[str, float, str], str] = {}

# Expected band keys (FR-CV-013 alignment)
_BAND_ORDER: tuple[str, ...] = ("low", "low_mid", "high_mid", "high")


class NLPTimeoutError(Exception):
    """
    Raised when a primary or fallback LLM call fails, times out, or returns unusable output.

    Maps to FR-NLP-002 (timeout) and orchestration fallbacks in FR-NLP-003 / FR-NLP-007.
    """


def clear_explanation_cache() -> None:
    """Clear in-memory NLP cache (tests and process reset)."""

    _EXPLANATION_CACHE.clear()


def _nearest_confidence_bucket(confidence: float, buckets: list[float]) -> float:
    """Return bucket value nearest to confidence (FR-NLP-008)."""

    if not buckets:
        return float(confidence)
    return min(buckets, key=lambda b: abs(float(b) - float(confidence)))


def _top_band_name(band_pct: dict[str, float]) -> str:
    """Return the band key with maximum attribution percentage."""

    if not band_pct:
        return "low"
    return max(band_pct.items(), key=lambda kv: kv[1])[0]


def _cache_key(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> tuple[str, float, str]:
    """Build cache key (label, confidence_bucket, top_band_name) per FR-NLP-008."""

    nlp = cfg["nlp"]
    bucket = _nearest_confidence_bucket(float(confidence), list(nlp["caching"]["confidence_buckets"]))
    top = _top_band_name(band_pct)
    return (str(label), float(bucket), str(top))


def get_cached_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str | None:
    """
    Return cached LLM explanation if caching is enabled and key matches (FR-NLP-008).

    Args:
        label: bonafide or spoof label string.
        confidence: Model confidence in [0, 1].
        band_pct: Four-band attribution dict.
        cfg: Full configuration including nlp.caching.

    Returns:
        Cached explanation string or None if disabled or miss.

    Raises:
        KeyError: NLP-008 if required config paths are missing.
    """

    if not bool(cfg["nlp"]["caching"]["enabled"]):
        return None
    key = _cache_key(label, confidence, band_pct, cfg)
    return _EXPLANATION_CACHE.get(key)


def _store_cache_if_enabled(
    label: str,
    confidence: float,
    band_pct: dict[str, float],
    cfg: dict[str, Any],
    text: str,
) -> None:
    if not bool(cfg["nlp"]["caching"]["enabled"]):
        return
    key = _cache_key(label, confidence, band_pct, cfg)
    _EXPLANATION_CACHE[key] = text


def build_prompt(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str:
    """
    Build a structured user prompt for the LLM (FR-NLP-001); no network I/O.

    Args:
        label: Predicted class label (e.g. bonafide / spoof).
        confidence: Confidence in [0, 1] (display as percent in prompt).
        band_pct: Keys low, low_mid, high_mid, high summing to ~100%.
        cfg: Full configuration; uses nlp.explanation_* and band_display_names.

    Returns:
        A single string suitable as the user message for chat completion.

    Raises:
        KeyError: If required config or band keys are missing.
    """

    nlp = cfg["nlp"]
    mn = int(nlp["explanation_min_sentences"])
    mx = int(nlp["explanation_max_sentences"])
    names = nlp["band_display_names"]
    top = _top_band_name(band_pct)
    top_pct = float(band_pct[top])
    top_label = str(names[top])
    lines = [
        "You explain deepfake speech detection results for a security reviewer.",
        f"Predicted label: {label}.",
        f"Model confidence: {float(confidence) * 100.0:.1f}% (0–100%).",
        "Grad-CAM frequency band attribution (percent, sum ~100%):",
    ]
    for k in _BAND_ORDER:
        lines.append(f"  - {names[k]}: {float(band_pct[k]):.2f}%")
    lines.append(
        f"The highest-attribution band is {top_label} at {top_pct:.2f}% — cite this band explicitly."
    )
    lines.append(
        f"Write {mn}–{mx} clear English sentences for a non-expert. "
        "Do not invent numbers beyond those given."
    )
    return "\n".join(lines)


def build_rule_based_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str:
    """
    Deterministic English explanation without external APIs (FR-NLP-003, FR-NLP-004).

    Args:
        label: bonafide or spoof.
        confidence: Confidence in [0, 1].
        band_pct: Four-band dict.
        cfg: Full configuration; uses nlp.band_display_names and nlp.rule_narrative.

    Returns:
        A grammatically correct multi-sentence paragraph.

    Raises:
        KeyError: If config narrative keys are missing.
    """

    nlp = cfg["nlp"]
    names = nlp["band_display_names"]
    narrative = nlp["rule_narrative"]
    top = _top_band_name(band_pct)
    top_pct = float(band_pct[top])
    highest_label = str(names[top])
    pct_int = int(round(float(confidence) * 100.0))
    is_spoof = str(label).strip().lower() == "spoof"
    hint_key = "spoof_top_band_hint" if is_spoof else "bonafide_top_band_hint"
    follow_key = "spoof_followup" if is_spoof else "bonafide_followup"
    s1 = (
        f"Analysis indicates {label} speech with {pct_int}% confidence. "
        f"The {highest_label} frequency band ({top_pct:.1f}%) showed the highest activation, "
        f"suggesting {narrative[hint_key]}."
    )
    s2 = str(narrative[follow_key])
    s3 = (
        "This explanation is generated offline from the model outputs and band saliency "
        "when the cloud language model is unavailable."
    )
    return " ".join([s1, s2, s3])


async def _openai_compatible_chat(
    prompt: str,
    cfg: dict[str, Any],
    kind: _LlmKind,
) -> str:
    """
    Shared OpenAI-compatible chat completion with timeout (FR-NLP-002, FR-NLP-007).

    Credentials: Qwen uses api_key_env_var only; Gemma prefers HF token env then Qwen env.
    """

    nlp = cfg["nlp"]
    timeout_sec = float(nlp["timeout_sec"])
    if kind == "qwen":
        env_name = str(nlp["api_key_env_var"])
        key = os.environ.get(env_name)
        if not key:
            log_error("nlp", "Qwen API key missing from environment", {"env_var": env_name})
            raise NLPTimeoutError("NLP-002")
        sub = nlp["qwen"]
        err_tag = "Qwen"
        empty_code = "NLP-002-empty"
        timeout_code = "NLP-002-timeout"
        fail_code = "NLP-002-failure"
    else:
        hf_name = str(nlp[_nlp_hf_token_env_key()])
        qwen_name = str(nlp["api_key_env_var"])
        key = os.environ.get(hf_name) or os.environ.get(qwen_name)
        if not key:
            log_error("nlp", "Gemma API token missing from environment", {"env_var_name": hf_name})
            raise NLPTimeoutError("NLP-007")
        sub = nlp["gemma"]
        err_tag = "Gemma"
        empty_code = "NLP-007-empty"
        timeout_code = "NLP-007-timeout"
        fail_code = "NLP-007-failure"

    base_url = str(sub["openai_compatible_base_url"])
    model = str(sub["model"])
    client = AsyncOpenAI(api_key=key, base_url=base_url)

    async def _call() -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        if content is None or not str(content).strip():
            raise NLPTimeoutError(empty_code)
        return str(content).strip()

    try:
        return await asyncio.wait_for(_call(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        log_error("nlp", f"{err_tag} API asyncio timeout", {"timeout_sec": timeout_sec})
        raise NLPTimeoutError(timeout_code) from None
    except NLPTimeoutError:
        raise
    except Exception as exc:
        log_error("nlp", f"{err_tag} API failure", {"error_type": type(exc).__name__})
        raise NLPTimeoutError(fail_code) from exc


async def call_qwen_api(prompt: str, cfg: dict[str, Any]) -> str:
    """
    Call Qwen 2.5 via an OpenAI-compatible HTTPS endpoint (FR-NLP-002).

    API key is read only from os.environ using cfg['nlp']['api_key_env_var'] (FR-NLP-005).

    Args:
        prompt: User message text from build_prompt.
        cfg: Full configuration; uses nlp.qwen, nlp.timeout_sec, nlp.api_key_env_var.

    Returns:
        Non-empty assistant text from the chat completion.

    Raises:
        NLPTimeoutError: On missing key, timeout, HTTP/API error, or empty response (FR-NLP-002).
    """

    return await _openai_compatible_chat(prompt, cfg, "qwen")


async def call_gemma_fallback(prompt: str, cfg: dict[str, Any]) -> str:
    """
    Secondary OpenAI-compatible LLM fallback (FR-NLP-007 SHOULD).

    Uses HF token env when set; otherwise falls back to the same env as Qwen for local dev only.

    Args:
        prompt: Same user message as primary.
        cfg: Full configuration; uses nlp.gemma, nlp.timeout_sec, HF token env name.

    Returns:
        Non-empty assistant text.

    Raises:
        NLPTimeoutError: Same contract as call_qwen_api.
    """

    return await _openai_compatible_chat(prompt, cfg, "gemma")


async def generate_explanation(
    label: str,
    confidence: float,
    band_pct: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[str, bool]:
    """
    Async orchestration: Qwen → Gemma → rule-based; optional cache (FR-NLP-003, FR-NLP-006–008).

    Args:
        label: Predicted label string.
        confidence: Confidence in [0, 1].
        band_pct: Four-band percentages.
        cfg: Full configuration.

    Returns:
        (explanation_text, api_was_used). api_was_used False for rule-only path (FR-NLP-003 UI badge).

    Raises:
        KeyError: On missing configuration keys.
    """

    cached = get_cached_explanation(label, confidence, band_pct, cfg)
    if cached is not None:
        log_info("nlp", "NLP cache hit", {"label": label})
        return (cached, True)

    prompt = build_prompt(label, confidence, band_pct, cfg)
    api_used = False
    text: str | None = None

    try:
        text = await call_qwen_api(prompt, cfg)
        api_used = True
    except NLPTimeoutError:
        try:
            text = await call_gemma_fallback(prompt, cfg)
            api_used = True
        except NLPTimeoutError:
            text = build_rule_based_explanation(label, confidence, band_pct, cfg)
            api_used = False

    if api_used and text is not None:
        _store_cache_if_enabled(label, confidence, band_pct, cfg, text)
    return (text if text is not None else "", api_used)
