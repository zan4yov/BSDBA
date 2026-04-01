"""
Module: src.tests.test_nlp
SRS Reference: FR-NLP-001–009
Phase: 4-D — Sprint D
V.E.R.I.F.Y. Level: 3
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

import src.nlp.explain as explain


def _load_cfg() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture()
def cfg() -> dict[str, Any]:
    return _load_cfg()


@pytest.fixture(autouse=True)
def _clear_cache_each_test() -> None:
    explain.clear_explanation_cache()


def test_build_prompt_contains_all_fields(cfg: dict[str, Any]) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 50.0, "high": 20.0}
    prompt = explain.build_prompt("bonafide", 0.7, band_pct, cfg)

    assert "bonafide" in prompt
    assert "70.0%" in prompt
    for band_key in ("low", "low_mid", "high_mid", "high"):
        assert f"- {band_key} (" in prompt


def test_rule_based_fallback_always_returns(cfg: dict[str, Any]) -> None:
    band_pct = {"low": 25.0, "low_mid": 25.0, "high_mid": 25.0, "high": 25.0}
    text = explain.build_rule_based_explanation("spoof", 0.8, band_pct, cfg)
    assert isinstance(text, str)
    assert text.strip() != ""


def test_rule_based_grammar(cfg: dict[str, Any]) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 60.0, "high": 10.0}
    text = explain.build_rule_based_explanation("bonafide", 0.6, band_pct, cfg)

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    assert len(sentences) >= 3


@pytest.mark.asyncio
async def test_qwen_timeout_triggers_fallback(cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 50.0, "high": 20.0}

    async def _timeout_qwen(*_: Any, **__: Any) -> str:
        raise explain.NLPTimeoutError("NLP-002-timeout")

    async def _timeout_gemma(*_: Any, **__: Any) -> str:
        raise explain.NLPTimeoutError("NLP-007-timeout")

    monkeypatch.setattr(explain, "call_qwen_api", _timeout_qwen)
    monkeypatch.setattr(explain, "call_gemma_fallback", _timeout_gemma)

    text, api_used = await explain.generate_explanation("spoof", 0.8, band_pct, cfg)

    expected = explain.build_rule_based_explanation("spoof", 0.8, band_pct, cfg)
    assert text == expected
    assert api_used is False


@pytest.mark.asyncio
async def test_warning_flag_on_fallback(cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 50.0, "high": 20.0}

    async def _timeout_qwen(*_: Any, **__: Any) -> str:
        raise explain.NLPTimeoutError("NLP-002-timeout")

    async def _timeout_gemma(*_: Any, **__: Any) -> str:
        raise explain.NLPTimeoutError("NLP-007-timeout")

    monkeypatch.setattr(explain, "call_qwen_api", _timeout_qwen)
    monkeypatch.setattr(explain, "call_gemma_fallback", _timeout_gemma)

    _, api_used = await explain.generate_explanation("spoof", 0.8, band_pct, cfg)
    assert api_used is False


@pytest.mark.asyncio
async def test_cv_result_independent_of_nlp(cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 50.0, "high": 20.0}

    async def _slow_qwen(*_: Any, **__: Any) -> str:
        await asyncio.sleep(0.05)
        return "API explanation text"

    monkeypatch.setattr(explain, "call_qwen_api", _slow_qwen)
    monkeypatch.setattr(explain, "call_gemma_fallback", _slow_qwen)

    task = asyncio.create_task(explain.generate_explanation("bonafide", 0.8, band_pct, cfg))

    async def _mock_cv_display() -> str:
        await asyncio.sleep(0)
        return "cv displayed"

    ui_result = await _mock_cv_display()
    assert ui_result == "cv displayed"
    assert not task.done()

    await task


def test_no_api_key_in_source() -> None:
    explain_path = Path(__file__).resolve().parents[2] / "src" / "nlp" / "explain.py"
    source = explain_path.read_text(encoding="utf-8", errors="ignore")

    forbidden = ("hf_", "sk-", "DASH", "Bearer")
    for pat in forbidden:
        assert pat not in source


@pytest.mark.asyncio
async def test_cache_hit_skips_api(cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    band_pct = {"low": 10.0, "low_mid": 20.0, "high_mid": 50.0, "high": 20.0}
    calls: list[int] = []

    async def _qwen_ok(*_: Any, **__: Any) -> str:
        calls.append(1)
        return "cached api explanation"

    monkeypatch.setattr(explain, "call_qwen_api", _qwen_ok)
    monkeypatch.setattr(explain, "call_gemma_fallback", _qwen_ok)

    text1, api_used1 = await explain.generate_explanation("bonafide", 0.8, band_pct, cfg)
    assert text1 == "cached api explanation"
    assert api_used1 is True

    text2, api_used2 = await explain.generate_explanation("bonafide", 0.8, band_pct, cfg)
    assert text2 == "cached api explanation"
    assert api_used2 is True
    assert len(calls) == 1

