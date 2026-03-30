"""NLP explanation tests (FR-NLP-001–009, V.E.R.I.F.Y. L3)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
import pytest
import yaml

from src.nlp.explain import (
    NLPTimeoutError,
    build_prompt,
    build_rule_based_explanation,
    call_gemma_fallback,
    call_qwen_api,
    clear_explanation_cache,
    generate_explanation,
)


def _load_cfg() -> dict:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def cfg() -> dict:
    return _load_cfg()


@pytest.fixture(autouse=True)
def _reset_nlp_cache() -> None:
    clear_explanation_cache()
    yield
    clear_explanation_cache()


def _bands() -> dict[str, float]:
    return {"low": 10.0, "low_mid": 20.0, "high_mid": 30.0, "high": 40.0}


def test_build_prompt_contains_all_fields(cfg: dict) -> None:
    band = _bands()
    p = build_prompt("spoof", 0.87, band, cfg)
    assert "spoof" in p
    assert "87.0%" in p or "87" in p
    for k in ("low", "low_mid", "high_mid", "high"):
        assert k in p or cfg["nlp"]["band_display_names"][k] in p
        assert f"{band[k]:.2f}" in p


def test_rule_based_fallback_always_returns(cfg: dict) -> None:
    out = build_rule_based_explanation("bonafide", 0.91, _bands(), cfg)
    assert isinstance(out, str)
    assert len(out) > 40


def test_rule_based_grammar(cfg: dict) -> None:
    out = build_rule_based_explanation("spoof", 0.55, _bands(), cfg)
    parts = [s.strip() for s in out.replace("?", ".").split(".") if s.strip()]
    assert len(parts) >= 3


@pytest.mark.asyncio
async def test_qwen_timeout_triggers_fallback(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_qwen(*args: object, **kwargs: object) -> str:
        raise NLPTimeoutError("t")

    async def fail_gemma(*args: object, **kwargs: object) -> str:
        raise NLPTimeoutError("t")

    monkeypatch.setattr("src.nlp.explain.call_qwen_api", fail_qwen)
    monkeypatch.setattr("src.nlp.explain.call_gemma_fallback", fail_gemma)

    text, api_used = await generate_explanation("spoof", 0.7, _bands(), cfg)
    assert not api_used
    assert "Analysis indicates" in text


@pytest.mark.asyncio
async def test_warning_flag_on_fallback(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_qwen(*args: object, **kwargs: object) -> str:
        raise NLPTimeoutError("t")

    async def fail_gemma(*args: object, **kwargs: object) -> str:
        raise NLPTimeoutError("t")

    monkeypatch.setattr("src.nlp.explain.call_qwen_api", fail_qwen)
    monkeypatch.setattr("src.nlp.explain.call_gemma_fallback", fail_gemma)

    _, api_used = await generate_explanation("bonafide", 0.66, _bands(), cfg)
    assert api_used is False


@pytest.mark.asyncio
async def test_cv_result_independent_of_nlp(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_qwen(*args: object, **kwargs: object) -> str:
        await asyncio.sleep(0.35)
        return "nlp done"

    monkeypatch.setattr("src.nlp.explain.call_qwen_api", slow_qwen)

    band = _bands()
    cv_panel = {"label": "spoof", "confidence": 0.9}
    nlp_task = asyncio.create_task(generate_explanation("spoof", 0.9, band, cfg))
    assert cv_panel["label"] == "spoof"
    assert not nlp_task.done()
    out, used = await asyncio.wait_for(nlp_task, timeout=3.0)
    assert out == "nlp done"
    assert used is True


def test_no_api_key_in_source() -> None:
    """Leaked-secret shapes (long hf_/sk-) plus forbidden literals per FR-NLP-005 gate checklist."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "src" / "nlp" / "explain.py").read_text(encoding="utf-8")
    assert re.search(r"hf_[A-Za-z0-9]{20,}", text) is None
    assert re.search(r"sk-[A-Za-z0-9]{20,}", text) is None
    assert "DASH" not in text
    assert "Bearer" not in text


@pytest.mark.asyncio
async def test_cache_hit_skips_api(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def count_qwen(*args: object, **kwargs: object) -> str:
        calls["n"] += 1
        return "cached-llm-text"

    monkeypatch.setattr("src.nlp.explain.call_qwen_api", count_qwen)

    band = _bands()
    a, u1 = await generate_explanation("spoof", 0.8, band, cfg)
    b, u2 = await generate_explanation("spoof", 0.8, band, cfg)
    assert calls["n"] == 1
    assert a == b == "cached-llm-text"
    assert u1 and u2


@pytest.mark.asyncio
async def test_call_qwen_raises_without_env_key(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg["nlp"]["api_key_env_var"], raising=False)
    with pytest.raises(NLPTimeoutError):
        await call_qwen_api("hello", cfg)


@pytest.mark.asyncio
async def test_call_gemma_raises_without_token(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg["nlp"]["hf_token_env_var"], raising=False)
    monkeypatch.delenv(cfg["nlp"]["api_key_env_var"], raising=False)
    with pytest.raises(NLPTimeoutError):
        await call_gemma_fallback("hello", cfg)
