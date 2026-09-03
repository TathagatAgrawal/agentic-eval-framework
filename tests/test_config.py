"""Tests for Settings loading and the Gemini LLM factory.

The live-call test hits the real Gemini API and is skipped by default -- run it
explicitly with `RUN_LIVE_TESTS=1 pytest tests/test_config.py` once a valid
GOOGLE_API_KEY is configured in `.env`.
"""

import os

import pytest

from finance_qna.config import get_llm, get_settings

requires_live_api = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run tests that call the real Gemini API",
)


def test_settings_load_model_defaults() -> None:
    """Settings must load from .env and fall back to the configured model defaults."""
    settings = get_settings()
    assert settings.agent_model == "gemini-3.5-flash-lite"
    assert settings.classifier_model == "gemini-3.5-flash-lite"
    assert settings.google_api_key.get_secret_value() != ""


@requires_live_api
def test_get_llm_can_reach_gemini() -> None:
    """get_llm must return a client that can successfully call the real Gemini API."""
    llm = get_llm(get_settings().classifier_model)
    response = llm.invoke("Reply with exactly the word: pong")
    assert "pong" in str(response.content).lower()
